import torch
from torch import nn
import torch.nn.functional as F

from plyfile import PlyData, PlyElement
import tqdm

import sdfoam
from sdfoam_model.render import TraceRays
from sdfoam_model.utils import *

from sdfoam_model.sdfnet import *


class SDFoamScene(torch.nn.Module):

    def __init__(
        self,
        args,
        points=None,
        points_colors=None,
        cameras=None,
        device=torch.device("cuda"),
        attr_dtype=torch.float32,
    ):
        super().__init__()

        self.device = device
        self.attr_dtype = attr_dtype
        if cameras is not None:
            self.cameras = cameras.to(device)
        else:
            self.cameras = None
        self.sh_degree = args.sh_degree
        self.num_init_points = args.init_points
        self.num_final_points = args.final_points
        self.activation_scale = args.activation_scale
        self.use_sdfoam = args.use_sdfoam
        if self.use_sdfoam:
            self.variance_net = SingleVarianceNetwork(init_val=0.3, device=device) # -0.3
            #use the pointcloud to initialize the SDF network with a geometric initialization
            self.sdf_network = SDFNetwork(
                d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,), multires=6, bias=0.50, scale=1.0, geometric_init=True, weight_norm=True, point_cloud=points
            ).to(self.device)
            # self.sdf_network = SDFNetwork(
            #     d_in=3, d_out=1, d_hidden=64, n_layers=8, skip_in=(4,), multires=6, bias=0.50, scale=1.0, geometric_init=True, weight_norm=True
            # ).to(self.device)

        if points is not None:
            self.initialize_from_pcd(points, points_colors)
        else:
            self.random_initialize()

        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )
        self.att_sh = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1),
                device=device,
                dtype=self.attr_dtype,
            )
        )
        self.alpha = torch.tensor((self.primal_points.shape[0],1),device=self.device, dtype=self.attr_dtype)
        self.pipeline = sdfoam.create_pipeline(
            self.sh_degree, str(self.attr_dtype).replace("torch.", "")
        )

    def random_initialize(self):
        primal_points = (
            torch.randn(self.num_init_points, 3, device=self.device) * 25
        )
        self.triangulation = sdfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        self.att_dc = nn.Parameter(
            torch.zeros(
                self.num_init_points,
                3,
                device=self.device,
                dtype=self.attr_dtype,
            )
        )

        if not self.use_sdfoam:
            density = torch.zeros(self.num_init_points, 1, device=self.device, dtype=self.attr_dtype)
            self.density = nn.Parameter(density[perm])
        
    def initialize_from_pcd(self, points, points_colors):
        points = points.to(self.device)
        points_colors = points_colors.to(self.device)

        # Remove statistical outliers from COLMAP points
        with torch.no_grad():
            from sklearn.neighbors import NearestNeighbors
            import numpy as np

            nbrs = NearestNeighbors(n_neighbors=8, algorithm='kd_tree').fit(points.cpu().numpy())
            distances, _ = nbrs.kneighbors(points.cpu().numpy())
            mean_dist = distances[:, 1:].mean(axis=1)
            std_dist = mean_dist.std()
            mean_dist_mean = mean_dist.mean()
            inlier_mask = mean_dist < (mean_dist_mean + 2 * std_dist)

            points = points[inlier_mask]
            points_colors = points_colors[inlier_mask]


        num_random = 5_000
        if self.use_sdfoam:
            random = (torch.rand([num_random, 3], device=self.device) * 2 - 1)
            # compute the mean center of the points
            center = points.mean(dim=0, keepdim=True)
            random = center + random
            
        else:
            random = (torch.randn([num_random, 3], device=self.device) * 10)

        num_samples = int(0.9 * points.shape[0])
        print(
            f"Starting with {num_samples} points from {points.shape[0]} COLMAP points"
        )
        points_idx = torch.randint(0, points.shape[0], (num_samples,))
        samp_points = points[points_idx]
        samp_points += torch.randn_like(samp_points) * 1e-2
        samp_colors = points_colors[points_idx]

        primal_points = torch.cat([samp_points, random], dim=0)

        if not self.use_sdfoam:
            primal_density = torch.cat(
                [
                    torch.rand(samp_colors.shape[0], 1, dtype=self.attr_dtype),
                    -0.5 * torch.ones(num_random, 1, dtype=self.attr_dtype),
                ],
                dim=0,
            ).to(self.device)

        torch.cuda.empty_cache()

        self.triangulation = sdfoam.Triangulation(primal_points)
        perm = self.triangulation.permutation().to(torch.long)
        primal_points = primal_points[perm]

        self.primal_points = nn.Parameter(primal_points)
        self.faces = None

        self.update_triangulation(rebuild=False)

        if not self.use_sdfoam:
            self.density = nn.Parameter(primal_density)

        self.num_init_points = self.primal_points.shape[0]

    def permute_points(self, permutation):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["inv_s", "sharpness", "sdf"]:
                continue  # skip scalar params
            if "env" not in group["name"]:
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][
                        permutation
                    ]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][
                        permutation
                    ]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][permutation].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][permutation].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc        = optimizable_tensors["att_dc"]
        self.att_sh        = optimizable_tensors["att_sh"]
        if not self.use_sdfoam:
            self.density       = optimizable_tensors["density"]


    def update_triangulation(self, rebuild=True, incremental=False):
        if not self.primal_points.isfinite().all():
            raise RuntimeError("NaN in points")

        needs_permute = False
        perturbation = 1e-6
        del_points = self.primal_points
        failures = 0
        while rebuild:
            if failures > 25:
                raise RuntimeError("aborted triangulation after 25 attempts")
            try:
                needs_permute = self.triangulation.rebuild(
                    del_points, incremental=incremental
                )
                break
            except sdfoam.TriangulationFailedError as e:
                print("caught: ", e)
                perturbation *= 2
                failures += 1
                incremental = False
                with torch.no_grad():
                    del_points = (
                        self.primal_points
                        + perturbation * torch.randn_like(self.primal_points)
                    )

        if failures > 5:
            with torch.no_grad():
                self.primal_points.copy_(del_points)

        if needs_permute:
            perm = self.triangulation.permutation().to(torch.long)
            self.permute_points(perm)

        self.aabb_tree = sdfoam.build_aabb_tree(self.primal_points)

        self.point_adjacency = self.triangulation.point_adjacency()
        self.point_adjacency_offsets = (
            self.triangulation.point_adjacency_offsets()
        )

    def get_primal_density(self):
        return self.activation_scale * F.softplus(self.density, beta=10)

    def get_primal_sdf(self):
        return self.sdf_network(self.primal_points)
        
    def get_inv_s(self):
        # return self.variance_net(torch.zeros(1, device=self.device, dtype=torch.float32))
        return self.variance_net(torch.zeros([1, 3], device=self.device, dtype=torch.float32))[:, :1].clip(1e-6, 1e6).expand(self.primal_points.shape[0], 1)

    def get_primal_attributes(self):
        return torch.cat([self.att_dc, self.att_sh], dim=-1)

    def get_trace_data(self):
        points = self.primal_points
        # Last channel: SDF when SDFoam is enabled, density otherwise
        last_chan = self.get_primal_sdf() if self.use_sdfoam else self.get_primal_density()

        attributes = torch.cat([self.get_primal_attributes(), last_chan], dim=-1).to(self.attr_dtype).contiguous()
        return points, attributes, self.point_adjacency, self.point_adjacency_offsets

    def show(self, loop_fn=lambda v: None, iterations=None, **viewer_kwargs):
        sdfoam.run_with_viewer(
            self.pipeline, loop_fn, total_iterations=iterations, **viewer_kwargs
        )

    def get_starting_point(self, rays, points, aabb_tree):
        with torch.no_grad():
            camera_origins = rays[..., :3]
            unique_cameras, inverse_indices = torch.unique(
                camera_origins, dim=0, return_inverse=True
            )

            nn_inds = sdfoam.nn(points, aabb_tree, unique_cameras).long()

            start_point = nn_inds[inverse_indices]
            return start_point.type(torch.uint32)

    def forward(
        self,
        rays,
        start_point=None,
        depth_quantiles=None,
        return_contribution=True,
    ):
        if self.use_sdfoam:
            inv_s = self.get_inv_s().to(dtype=self.attr_dtype).contiguous()
        else:
            inv_s = torch.ones([1, 1], device=self.device, dtype=self.attr_dtype).contiguous()


        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )

        if start_point is None:
            start_point = self.get_starting_point(rays, points, self.aabb_tree)
        else:
            start_point = torch.broadcast_to(start_point, rays.shape[:-1])
        
        #alpha_output= torch.tensor((points.shape[0], 1),device=self.device, dtype=self.attr_dtype)

        # Order must match TraceRays.forward signature.
        return TraceRays.apply(
            self.pipeline,                # pipeline
            points,                       # _points
            attributes,                   # _attributes
            point_adjacency,              # _point_adjacency
            point_adjacency_offsets,      # _point_adjacency_offsets
            rays,                         # rays
            start_point,                  # start_point
            depth_quantiles,              # depth_quantiles
            bool(return_contribution),    # return_contribution
            #alpha_output,
            bool(self.use_sdfoam),          # use_sdfoam
            inv_s,                        # inv_s
            1e-6,                         # eps
            int(attributes.shape[-1] - 1) if self.use_sdfoam else -1,  # sdf_channel
        )

    def update_viewer(self, viewer):
        points, attributes, point_adjacency, point_adjacency_offsets = (
            self.get_trace_data()
        )

        num_points = points.shape[0]
        if self.use_sdfoam:
            invs_expanded = (
                self.get_inv_s()
                .to(device=self.device, dtype=torch.float32)
                .expand(num_points, 1)
                .contiguous()
            )
        else:
            invs_expanded = torch.ones(
                num_points, 1, device=self.device, dtype=torch.float32
            )

        viewer.update_scene(
            points,
            attributes,
            point_adjacency,
            point_adjacency_offsets,
            self.aabb_tree,
            invs_expanded,
        )

    def declare_optimizer(self, args, warmup, max_iterations):
        params = [
            {
                "params": self.primal_points,
                "lr": args.points_lr_init,
                "name": "primal_points",
            },
            {
                "params": self.att_dc,
                "lr": args.attributes_lr_init,
                "name": "att_dc",
            },
            {
                "params": self.att_sh,
                "lr": args.attributes_lr_init,
                "name": "att_sh",
            },
        ]
        if self.use_sdfoam:
            params.append(
                {
                    "params": self.sdf_network.parameters(), 
                    "lr": args.sdf_lr_init, 
                    "name": "sdf",
                },
            )
            params.append(
                {
                    "params": self.variance_net.parameters(), 
                    "lr": args.sharpness_lr_init,
                    "name": "inv_s",
                },
            )
        else:
            params.append(
                {
                    "params": self.density,
                    "lr": args.density_lr_init,
                    "name": "density",
                }
            )

        self.optimizer = torch.optim.Adam(params, eps=1e-15)
        self.xyz_scheduler_args = get_cosine_lr_func(
            lr_init=args.points_lr_init,
            lr_final=args.points_lr_final,
            max_steps=args.freeze_points,
        )
        self.attr_dc_scheduler_args = get_cosine_lr_func(
            lr_init=args.attributes_lr_init,
            lr_final=args.attributes_lr_final,
            max_steps=max_iterations,
        )
        self.attr_rest_scheduler_args = get_cosine_lr_func(
            lr_init=args.sh_factor * args.attributes_lr_init,
            lr_final=args.sh_factor * args.attributes_lr_final,
            warmup_steps=max_iterations // 5,
            max_steps=max_iterations,
        )
        if self.use_sdfoam:
            self.sdf_scheduler_args = get_cosine_lr_func(
                lr_init=args.sdf_lr_init,
                lr_final=args.sdf_lr_final,
                # warmup_steps=warmup,
                max_steps=max_iterations,
            )
            self.sharpness_scheduler_args = get_cosine_lr_func(
                lr_init=args.sharpness_lr_init,
                lr_final=args.sharpness_lr_final,
                #warmup_steps=max_iterations // 5,
                max_steps=max_iterations,
            )
        else:
            self.den_scheduler_args = get_cosine_lr_func(
                lr_init=args.density_lr_init,
                lr_final=args.density_lr_final,
                warmup_steps=warmup,
                max_steps=max_iterations,
            )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "primal_points":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "density":
                lr = self.den_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "sdf":
                lr = self.sdf_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_dc":
                lr = self.attr_dc_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "att_sh":
                lr = self.attr_rest_scheduler_args(iteration)
                param_group["lr"] = lr
            elif param_group["name"] == "inv_s":
                lr = self.sharpness_scheduler_args(iteration)
                param_group["lr"] = lr


    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["inv_s", "sharpness", "sdf"]:
                continue  # skip scalar params
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, prune_mask):
        valid_points_mask = ~prune_mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        if not self.use_sdfoam:
            self.density = optimizable_tensors["density"]

    def cat_tensors_to_optimizer(self, new_params):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["inv_s", "sharpness", "sdf"]:
                continue  # skip scalar params
            if group["name"] in new_params.keys():
                assert len(group["params"]) == 1
                stored_tensor = group["params"][0]
                extension_tensor = new_params[group["name"]]
                stored_state = self.optimizer.state.get(
                    group["params"][0], None
                )
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (
                            stored_state["exp_avg"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (
                            stored_state["exp_avg_sq"],
                            torch.zeros_like(extension_tensor),
                        ),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (stored_tensor, extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def densification_postfix(self, new_params):
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        self.primal_points = optimizable_tensors["primal_points"]
        self.att_dc = optimizable_tensors["att_dc"]
        self.att_sh = optimizable_tensors["att_sh"]
        if not self.use_sdfoam:
            self.density = optimizable_tensors["density"]

    def prune_and_densify(
        self, point_error, point_contribution, upsample_factor=1.2
    ):
        with torch.no_grad():
            num_curr_points = self.primal_points.shape[0]
            num_new_points = int((upsample_factor - 1) * num_curr_points)

            primal_error_accum = point_error.clip(min=0).squeeze()
            points, _, point_adjacency, point_adjacency_offsets = (
                self.get_trace_data()
            )
            ################### Farthest neighbor ###################
            farthest_neighbor, cell_radius = sdfoam.farthest_neighbor(
                points,
                point_adjacency,
                point_adjacency_offsets,
            )
            farthest_neighbor = farthest_neighbor.long()

            ######################## Pruning ########################
            self_mask = point_contribution > 1e-2
            neighbor_mask = self_mask.long()[point_adjacency.long()]
            neighbor_mask = torch.cat(
                [neighbor_mask, torch.zeros_like(neighbor_mask[:1])], dim=0
            )
            nsum = torch.cumsum(neighbor_mask, dim=0)

            offsets = point_adjacency_offsets.long()
            n_masked_adj = nsum[offsets[1:]] - nsum[offsets[:-1]]

            contrib_mask = ((n_masked_adj == 0) & ~self_mask).squeeze()
            cell_size_mask = cell_radius < 1e-3
            prune_mask = contrib_mask * cell_size_mask

            ######################## Random sampling ########################
            primal_contribution_accum = point_contribution.squeeze()
            mask = primal_contribution_accum < 1e-3
            if not self.use_sdfoam:
                self.density[mask] = -1

            perturbation = 0.25 * (points[farthest_neighbor] - points)
            delta = torch.randn_like(perturbation)
            delta /= delta.norm(dim=-1, keepdim=True)
            perturbation += (
                0.1 * perturbation.norm(dim=-1, keepdim=True) * delta
            )

            num_sample_points = num_new_points

            sampled_inds = torch.multinomial(
                primal_error_accum * cell_radius,
                num_sample_points,
                replacement=False,
            )

            sampled_points = (points + perturbation)[sampled_inds]

            new_params = {
                "primal_points": sampled_points,
                "att_dc": self.att_dc[sampled_inds],
                "att_sh": self.att_sh[sampled_inds],
            }
            if not self.use_sdfoam:
                new_params["density"] = self.density[sampled_inds]

            prune_mask = torch.cat(
                (
                    prune_mask,
                    torch.zeros(
                        sampled_points.shape[0],
                        device=prune_mask.device,
                        dtype=bool,
                    ),
                )
            )

            self.densification_postfix(new_params)
            self.prune_points(prune_mask)

    def collect_error_map(self, data_handler, white_bkg=True, downsample=2):
        rays, rgbs = data_handler.rays, data_handler.rgbs

        points, _, _, _ = self.get_trace_data()
        start_points = self.get_starting_point(
            rays[:, 0, 0].cuda(), points, self.aabb_tree
        )

        ray_batch_fetcher = sdfoam.BatchFetcher(
            rays, batch_size=1, shuffle=False
        )
        rgb_batch_fetcher = sdfoam.BatchFetcher(
            rgbs, batch_size=1, shuffle=False
        )

        point_error_accum = torch.zeros_like(self.primal_points[..., 0:1])
        point_contribution_accum = torch.zeros_like(
            self.primal_points[..., 0:1]
        )
        rgb_loss = nn.L1Loss(reduction="none")

        for i in range(rays.shape[0]):
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()

            d = torch.randint(0, downsample, (2,))
            ray_batch = ray_batch[:, d[0] :: downsample, d[1] :: downsample, :]
            rgb_batch = rgb_batch[:, d[0] :: downsample, d[1] :: downsample, :]

            rgba_output, _, contribution, _, errbox, *_ = self.forward(
                ray_batch, start_points[i], return_contribution=True
            )

            opacity = rgba_output[..., -1:]
            if white_bkg:
                rgb_output = rgba_output[..., :3] + (1 - opacity)
            else:
                rgb_output = rgba_output[..., :3]

            color_loss = rgb_loss(rgb_batch, rgb_output).mean(dim=-1)

            color_loss.sum().backward()
            point_error_accum += self.primal_points.grad.norm(
                dim=-1, keepdim=True
            ).detach()
            point_contribution_accum = torch.maximum(
                point_contribution_accum, contribution.detach()
            )
            torch.cuda.synchronize()

            self.optimizer.zero_grad(set_to_none=True)

        return point_error_accum, point_contribution_accum

    def save_ply(self, ply_path):
        points = self.primal_points.detach().float().cpu().numpy()
        if self.use_sdfoam:
            density = self.get_primal_sdf().detach().float().cpu().numpy()
        color_attributes = (
            self.get_primal_attributes().detach().float().cpu().numpy()
        )
        adjacency = self.point_adjacency.cpu().numpy()
        adjacency_offsets = self.point_adjacency_offsets.cpu().numpy()

        C0 = 0.28209479177387814
        r = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 0]), 0, 255),
            dtype=np.uint8,
        )
        g = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 1]), 0, 255),
            dtype=np.uint8,
        )
        b = np.array(
            np.clip(255 * (0.5 + C0 * color_attributes[:, 2]), 0, 255),
            dtype=np.uint8,
        )

        vertex_data = []
        for i in tqdm.trange(points.shape[0]):
            vertex_data.append(
                (
                    points[i, 0],
                    points[i, 1],
                    points[i, 2],
                    r[i],
                    g[i],
                    b[i],
                    density[i, 0] if self.use_sdfoam else torch.ones(1).item(),
                    adjacency_offsets[i + 1],
                    *[
                        color_attributes[i, 3 + j]
                        for j in range(color_attributes.shape[1] - 3)
                    ],
                )
            )

        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("density", np.float32),
            ("adjacency_offset", np.uint32),
        ]

        for i in range(self.att_sh.shape[1]):
            dtype.append(("color_sh_{}".format(i), np.float32))

        vertex_data = np.array(vertex_data, dtype=dtype)
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        adjacency_data = np.array(adjacency, dtype=[("adjacency", np.uint32)])
        adjacency_element = PlyElement.describe(adjacency_data, "adjacency")

        PlyData([vertex_element, adjacency_element]).write(ply_path)

    def save_pt(self, pt_path):
        points = self.primal_points.detach().float().cpu()
        if not self.use_sdfoam:
            density = self.density.detach().float().cpu()
        else:
            sdf_network = self.sdf_network.state_dict()
            variance_net = self.variance_net.state_dict()
        
        alpha = self.alpha.detach().float().cpu()

        color_dc = self.att_dc.detach().float().cpu()
        color_sh = self.att_sh.detach().float().cpu()
        adjacency = self.point_adjacency.cpu()
        adjacency_offsets = self.point_adjacency_offsets.cpu()

        scene_data = {
            "xyz": points,
            "color_dc": color_dc,
            "color_sh": color_sh,
            "adjacency": adjacency.long(),
            "adjacency_offsets": adjacency_offsets.long(),
            "alpha": alpha,
        }
        if self.use_sdfoam:
            scene_data["sdf"] = sdf_network
            scene_data["variance"] = variance_net
        else:
            scene_data["density"] = density
        torch.save(scene_data, pt_path)

    def load_pt(self, pt_path):
        
        scene_data = torch.load(pt_path, map_location="cpu")

        def _strip_module_prefix(sd):
            """
            Gestisce checkpoint salvati con wrapper che aggiunge il prefisso 'module.'
            (es: TimedModule, DataParallel, DDP).
            """
            if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
                return {k[len("module."):]: v for k, v in sd.items()}
            return sd

        self.primal_points = nn.Parameter(scene_data["xyz"].to(self.device))

        if not self.use_sdfoam:
            self.density = nn.Parameter(scene_data["density"].to(self.device))
        else:
            self.sdf_network = SDFNetwork(
                d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,),
                multires=6, bias=0.5, scale=1.0, geometric_init=True,
                weight_norm=True, point_cloud=self.primal_points
            ).to(self.device)

            sdf_sd = _strip_module_prefix(scene_data["sdf"])
            self.sdf_network.load_state_dict(sdf_sd, strict=True)

            self.variance_net = SingleVarianceNetwork(init_val=0.3, device=self.device)
            var_sd = _strip_module_prefix(scene_data["variance"])
            self.variance_net.load_state_dict(var_sd, strict=True)

        self.att_dc = nn.Parameter(
            scene_data["color_dc"].to(self.attr_dtype).to(self.device)
        )

        exp_sh_coeffs = 3 * ((1 + self.sh_degree) * (1 + self.sh_degree) - 1)
        got_sh_coeffs = scene_data["color_sh"].shape[-1]
        assert exp_sh_coeffs == got_sh_coeffs, (
            f"Expected {exp_sh_coeffs} SH coeffs per-point, got {got_sh_coeffs}"
        )

        self.att_sh = nn.Parameter(
            scene_data["color_sh"].to(self.attr_dtype).to(self.device)
        )

        self.point_adjacency = scene_data["adjacency"].to(self.device).to(torch.uint32)
        self.point_adjacency_offsets = scene_data["adjacency_offsets"].to(self.device).to(torch.uint32)

        self.aabb_tree = sdfoam.build_aabb_tree(self.primal_points)
        self.alpha = scene_data["alpha"].to(self.device).to(self.attr_dtype)
