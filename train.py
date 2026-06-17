import os
import yaml
import gc
import numpy as np
from PIL import Image
import configargparse
import tqdm
import warnings
import logging
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
from torch import nn
import torch.nn.functional as F

import wandb

from data_loader import DataHandler
from configs import *
from sdfoam_model.scene import SDFoamScene
from sdfoam_model.utils import psnr
import sdfoam

from torchmetrics.functional import structural_similarity_index_measure as ssim
from lpips import LPIPS

os.environ["TBB_NUM_THREADS"] = "8"


# ----------------------------- Logging utils -----------------------------

def setup_logger(out_dir: str, verbose: bool) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("sdfoam.train")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.debug("Logger initialized")
    return logger

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)


# ----------------------------- Geometry helpers -----------------------------

def compute_bbox(c2ws, points3D, device):
    cam_centers = c2ws[:, :3, 3]
    pts  = torch.as_tensor(points3D, device=device, dtype=torch.float32)
    cams = torch.as_tensor(cam_centers, device=device, dtype=torch.float32)
    X = torch.cat([pts, cams], dim=0)
    aabb_min = X.min(dim=0).values
    aabb_max = X.max(dim=0).values
    eps = 1e-6
    aabb_max = torch.maximum(aabb_max, aabb_min + eps)
    return aabb_min, aabb_max

def sample_uniform_bbox(aabb_min, aabb_max, n, device):
    u = torch.rand(n, 3, device=device)
    return aabb_min + u * (aabb_max - aabb_min)


# ----------------------------- TRAIN -----------------------------

def train(args, pipeline_args, model_args, optimizer_args, dataset_args):
    device = torch.device(model_args.device)

    unique_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_mode = "SDFOAM" if model_args.use_sdfoam else "RADFOAM"
    run_mode += "_NVS" if dataset_args.eval else "_GEOM"
    group_scene = dataset_args.scene
    out_dir = f"output/{group_scene}_{run_mode}@{unique_str}"

    os.makedirs(out_dir, exist_ok=True)

    def represent_list_inline(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )

    yaml.add_representer(list, represent_list_inline)
    with open(f"{out_dir}/config.yaml", "w") as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

    wandb.init(
        project="SDFoam",
        group=group_scene,
        job_type=run_mode,
        name=f"{group_scene}_{run_mode}_{unique_str}",
        config=vars(args),
    )

    logger = setup_logger(out_dir, verbose=not pipeline_args.debug)
    logger.info(f"Experiment dir: {out_dir}")
    logger.info(f"W&B run: {wandb.run.name}")
    logger.info(f"Device: {device}, CUDA: {torch.cuda.is_available()}, torch={torch.__version__}")

    # Dataset
    iter2downsample = dict(zip(dataset_args.downsample_iterations, dataset_args.downsample))
    train_data_handler = DataHandler(dataset_args, rays_per_batch=1_000_000, device=device)
    downsample = iter2downsample[0]
    train_data_handler.reload(split="train", downsample=downsample)

    bbox_min, bbox_max = compute_bbox(train_data_handler.c2ws, train_data_handler.points3D, device)
    extent = bbox_max - bbox_min
    bbox_diag = extent.norm().item()
    tau_surface = 0.02 * bbox_diag
    surface_jitter_sigma = 0.005 * bbox_diag
    fixed_bbox_random_samples = 8192
    fixed_surface_random_samples = 2048
    logger.info(f"AABB extent: {extent.tolist()} | diag={bbox_diag:.4f}")

    test_data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))
    test_ray_batch_fetcher = sdfoam.BatchFetcher(test_data_handler.rays, batch_size=1, shuffle=False)
    test_rgb_batch_fetcher = sdfoam.BatchFetcher(test_data_handler.rgbs, batch_size=1, shuffle=False)

    viewer_options = {
        "camera_pos": train_data_handler.viewer_pos,
        "camera_up": train_data_handler.viewer_up,
        "camera_forward": train_data_handler.viewer_forward,
        "use_sdfoam": model_args.use_sdfoam,
    }

    rgb_loss = nn.SmoothL1Loss(reduction="none")

    # Model
    model = SDFoamScene(
        args=model_args,
        device=device,
        points=train_data_handler.points3D,
        points_colors=train_data_handler.points3D_colors,
    )

    model.declare_optimizer(args=optimizer_args, warmup=pipeline_args.densify_from, max_iterations=pipeline_args.iterations)
    logger.info(f"Model initialized: points={model.primal_points.shape[0]}, use_sdfoam={model_args.use_sdfoam}")

    lpips_fn = LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # ------------------------------ Test render ------------------------------

    def test_render(test_data_handler, ray_batch_fetcher, rgb_batch_fetcher, debug=False):
        logger.info("[test] rendering...")
        rays = test_data_handler.rays
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(rays[:, 0, 0].to(device), points, model.aabb_tree)

        os.makedirs(f"{out_dir}/test", exist_ok=True)

        psnr_list, ssim_list, lpips_list = [], [], []
        images_for_table = []

        with torch.no_grad():
            for i in range(rays.shape[0]):
                ray_batch = ray_batch_fetcher.next()[0]
                rgb_batch = rgb_batch_fetcher.next()[0]

                output, _, _, _, _, _, *_ = model(ray_batch, start_points[i])

                opacity = output[..., -1:]
                rgb_output = output[..., :3] + (1 - opacity)
                rgb_output = rgb_output.reshape(*rgb_batch.shape).clip(0, 1)

                img_psnr = psnr(rgb_output, rgb_batch).mean().item()
                img_ssim = ssim(
                    rgb_output.permute(2, 0, 1).unsqueeze(0),
                    rgb_batch.permute(2, 0, 1).unsqueeze(0),
                ).item()

                img1_lpips = (rgb_output.permute(2, 0, 1).unsqueeze(0) * 2 - 1).to(device)
                img2_lpips = (rgb_batch.permute(2, 0, 1).unsqueeze(0) * 2 - 1).to(device)
                img_lpips = lpips_fn(img1_lpips, img2_lpips).mean().item()

                psnr_list.append(img_psnr)
                ssim_list.append(img_ssim)
                lpips_list.append(img_lpips)

                if not debug:
                    error = np.uint8((rgb_output - rgb_batch).cpu().abs() * 255)
                    rgb_u8 = np.uint8(rgb_output.cpu() * 255)
                    gt_u8  = np.uint8(rgb_batch.cpu() * 255)
                    concat_img = np.concatenate([rgb_u8, gt_u8, error], axis=1)
                    im = Image.fromarray(concat_img)

                    save_path = f"{out_dir}/test/rgb_{i:03d}_psnr_{img_psnr:.3f}_ssim_{img_ssim:.3f}_lpips_{img_lpips:.3f}.png"
                    im.save(save_path)

                    caption = f"Frame {i:03d} | PSNR={img_psnr:.3f} | SSIM={img_ssim:.3f} | LPIPS={img_lpips:.3f}"
                    images_for_table.append([i, wandb.Image(concat_img, caption=caption), img_psnr, img_ssim, img_lpips])

        average_psnr = float(sum(psnr_list) / max(1, len(psnr_list)))
        average_ssim = float(sum(ssim_list) / max(1, len(ssim_list)))
        average_lpips = float(sum(lpips_list) / max(1, len(lpips_list)))

        logger.info(f"[test] PSNR={average_psnr:.3f}, SSIM={average_ssim:.3f}, LPIPS={average_lpips:.3f}")

        image_table = wandb.Table(columns=["Frame", "Image", "PSNR", "SSIM", "LPIPS"], data=images_for_table) if len(images_for_table) else None
        log_dict = {
            "test/psnr": average_psnr,
            "test/ssim": average_ssim,
            "test/lpips": average_lpips,
        }
        if image_table is not None:
            log_dict["test/images_table"] = image_table
        wandb.log(log_dict)
        return average_psnr, average_ssim, average_lpips

    # ------------------------------ Train loop -------------------------------

    def train_loop(viewer):
        logger.info("Training start")

        if device.type == "cuda":
            torch.cuda.set_device(0)
            _ = torch.empty(1, device=device)

        data_iterator = train_data_handler.get_iter()
        ray_batch, rgb_batch, alpha_batch = next(data_iterator)

        triangulation_update_period = 1
        iters_since_update = 1
        iters_since_densification = 0
        next_densification_after = 1

        with tqdm.trange(pipeline_args.iterations) as train_tqdm:
            for i in train_tqdm:
                # viewer
                if viewer is not None:
                    try:
                        model.update_viewer(viewer)
                        viewer.step(i)
                    except Exception as e:
                        logger.exception(f"[viewer] step failed at iter {i}: {e}")

                # dynamic downsample
                if i in iter2downsample and i:
                    downsample = iter2downsample[i]
                    train_data_handler.reload(split="train", downsample=downsample)
                    data_iterator = train_data_handler.get_iter()
                    ray_batch, rgb_batch, alpha_batch = next(data_iterator)

                # ---------------- quantiles ----------------
                depth_quantiles = torch.rand(
                    *ray_batch.shape[:-1], 2, device=device
                ).sort(dim=-1, descending=True).values

                # ---------------- model forward ----------------
                rgba_output, depth, _, _, _, alpha, *_ = model(
                    ray_batch, depth_quantiles=depth_quantiles
                )

                model.alpha = alpha

                # ---------------- losses (render) ----------------
                opacity = rgba_output[..., -1:]
                if pipeline_args.white_background:
                    rgb_output = rgba_output[..., :3] + (1 - opacity)
                else:
                    rgb_output = rgba_output[..., :3]

                color_loss = rgb_loss(rgb_batch, rgb_output)
                opacity_loss = ((alpha_batch - opacity) ** 2).mean()

                valid_depth_mask = (depth > 0).all(dim=-1)
                quant_loss = (depth[..., 0] - depth[..., 1]).abs()
                quant_loss = (quant_loss * valid_depth_mask).mean()
                w_depth = pipeline_args.quantile_weight * min(
                    2 * i / pipeline_args.iterations, 1
                )

                # ---------------- SDFoam loss ----------------
                loss_eikonal = None
                mask_loss = None
                w_mask = 0.0

                if model_args.use_sdfoam:
                    # ---------------- Sampling ----------------
                    x_primal = model.primal_points
                    x_bbox = sample_uniform_bbox(
                        bbox_min,
                        bbox_max,
                        fixed_bbox_random_samples,
                        device,
                    )

                    x_surface = torch.empty((0, 3), device=device)
                    if i >= 7000:
                        with torch.no_grad():
                            sdf_primal = model.sdf_network(x_primal.detach()).squeeze(-1)
                            surf_idx = torch.where(sdf_primal.abs() < tau_surface)[0]

                        if surf_idx.numel() > 0:
                            jitter_idx = surf_idx[
                                torch.randint(
                                    0,
                                    surf_idx.numel(),
                                    (fixed_surface_random_samples,),
                                    device=device,
                                )
                            ]
                            x_surface = x_primal.detach()[jitter_idx] + torch.randn(
                                (fixed_surface_random_samples, 3), device=device
                            ) * surface_jitter_sigma
                            x_surface = torch.clamp(x_surface, bbox_min, bbox_max)

                    x_eik = torch.cat([x_primal, x_bbox, x_surface], dim=0)

                    # Eikonal loss
                    grads = model.sdf_network.gradient(x_eik)
                    loss_eikonal = model.sdf_network.eikonal_loss(grads)

                    # masked config
                    if "_masked" in os.path.basename(args.config):
                        mask_loss = F.binary_cross_entropy(
                            opacity.clamp(0, 1),
                            (alpha_batch > 0.5).float()
                        )
                        w_mask = 0.01 * min(i / 5000, 1.0)

                # ---------------- loss composition ----------------
                loss_render = color_loss.mean() + opacity_loss
                if not model_args.use_sdfoam:
                    loss_render = loss_render + w_depth * quant_loss

                loss_sdf = None
                if model_args.use_sdfoam:
                    loss_sdf = 0.01 * loss_eikonal
                    if mask_loss is not None:
                        loss_sdf = loss_sdf + w_mask * mask_loss
                    loss_total = loss_render + loss_sdf
                else:
                    loss_total = loss_render

                model.optimizer.zero_grad(set_to_none=True)
                loss_total.backward()

                ray_batch, rgb_batch, alpha_batch = next(data_iterator)

                model.optimizer.step()
                model.update_learning_rate(i)

                train_tqdm.set_description(f"Iter {i} | loss={loss_total.item():.4f}")

                # logging every 100
                if i % 100 == 99:
                    log_dict = {
                        "train/rgb_loss": float(color_loss.mean().item()),
                        "train/opacity_loss": float(opacity_loss.item()),
                        "train/quantile_loss": float(quant_loss.item()),
                        "train/total_loss": float(loss_total.item()),
                        "train/w_depth": float(w_depth),
                        "train/num_points": int(model.primal_points.shape[0]),
                    }

                    if model_args.use_sdfoam:
                        log_dict["train/eikonal_loss"] = float(loss_eikonal.item())
                        if mask_loss is not None:
                            log_dict["train/mask_loss"] = float(mask_loss.item())
                            log_dict["train/w_mask"] = float(w_mask)

                        if hasattr(model, "variance_net"):
                            inv_s = model.variance_net(
                                torch.zeros([1, 3], device=model.device)
                            )[:, :1].clip(1e-6, 1e6)
                            log_dict["train/inv_s"] = float(inv_s.mean().item())

                    test_psnr, average_ssim, average_lpips = test_render(
                        test_data_handler, test_ray_batch_fetcher, test_rgb_batch_fetcher, True
                    )
                    log_dict["test/psnr"] = float(test_psnr)
                    log_dict["test/ssim"] = float(average_ssim)
                    log_dict["test/lpips"] = float(average_lpips)

                    wandb.log(log_dict, step=i)

                if iters_since_update >= triangulation_update_period:
                    model.update_triangulation(incremental=True)
                    iters_since_update = 0
                    if triangulation_update_period < 100:
                        triangulation_update_period += 2
                iters_since_update += 1

                # densification
                if i + 1 >= pipeline_args.densify_from:
                    iters_since_densification += 1

                if (iters_since_densification == next_densification_after
                    and model.primal_points.shape[0] < 0.9 * model.num_final_points):

                    point_error, point_contribution = model.collect_error_map(
                        train_data_handler, pipeline_args.white_background
                    )
                    model.prune_and_densify(
                        point_error,
                        point_contribution,
                        pipeline_args.densify_factor,
                    )

                    model.update_triangulation(incremental=False)

                    triangulation_update_period = 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    iters_since_densification = 0
                    next_densification_after = int(
                        ((pipeline_args.densify_factor - 1) * model.primal_points.shape[0]
                        * (pipeline_args.densify_until - pipeline_args.densify_from))
                        / max(1, (model.num_final_points - model.num_init_points))
                    )
                    next_densification_after = max(next_densification_after, 100)

                if i == optimizer_args.freeze_points:
                    model.update_triangulation(incremental=False)

                if viewer is not None and viewer.is_closed():
                    break

        # Save
        model.save_pt(f"{out_dir}/model.pt")
        model.save_ply(f"{out_dir}/scene.ply")
        logger.info("Training complete.")
        wandb.save(f"{out_dir}/model.pt")
        wandb.save(f"{out_dir}/scene.ply")


    # run
    if pipeline_args.viewer:
        model.show(train_loop, iterations=pipeline_args.iterations, **viewer_options)
    else:
        train_loop(viewer=None)

    test_render(test_data_handler, test_ray_batch_fetcher, test_rgb_batch_fetcher, pipeline_args.debug)
    wandb.finish()


# ----------------------------- MAIN -----------------------------

def main():
    parser = configargparse.ArgParser(default_config_files=["arguments/mipnerf360_outdoor_config.yaml"])
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)
    dataset_params = DatasetParams(parser)

    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    args = parser.parse_args()

    train(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
