#headless script to extract the voronoi mesh from a trained SDFoam model, no need of the viewer
import itertools
from pathlib import Path

import configargparse
import numpy as np
import torch
import trimesh
from scipy.spatial import Voronoi
from tqdm import tqdm

from configs import DatasetParams, ModelParams, OptimizationParams, PipelineParams
from sdfoam_model.scene import SDFoamScene


def save_mesh_with_colors(path_base, verts, faces, face_colors):
    ply_path = path_base + ".ply"
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        for i, face in enumerate(faces):
            r, g, b = (face_colors[i] * 255).astype(np.uint8)
            f.write(
                f"{len(face)} {' '.join(str(int(x)) for x in face)} {r} {g} {b}\n"
            )

    obj_path = path_base + ".obj"
    mtl_path = path_base + ".mtl"

    with open(mtl_path, "w") as mtl:
        for i, col in enumerate(face_colors):
            r, g, b = col
            mtl.write(f"newmtl m{i}\n")
            mtl.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")
            mtl.write("Ka 0 0 0\nKs 0 0 0\n\n")

    with open(obj_path, "w") as obj:
        obj.write(f"mtllib {mtl_path.split('/')[-1]}\n")
        for v in verts:
            obj.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for i, face in enumerate(faces):
            obj.write(f"usemtl m{i}\n")
            f_idx = [idx + 1 for idx in face]
            obj.write(f"f {' '.join(str(v) for v in f_idx)}\n")

    print(f"Saved Voronoi mesh: {path_base}.[ply|obj|mtl]")


def sanitize_val(v):
    s = f"{v:.4f}"
    return s.replace("-", "m").replace(".", "p")


def make_bounded_voronoi(all_points_np: np.ndarray, pad_ratio: float = 0.25):
    pmin = all_points_np.min(axis=0)
    pmax = all_points_np.max(axis=0)
    span = pmax - pmin
    pad = pad_ratio * float(np.linalg.norm(span) + 1e-9)

    lo = pmin - pad
    hi = pmax + pad

    corners = np.array(
        list(itertools.product([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]])),
        dtype=np.float64,
    )

    pts_bounded = np.vstack([all_points_np, corners])
    vor = Voronoi(pts_bounded)

    return vor, all_points_np.shape[0]


def reconstruct_voronoi_mesh_from_indices(
    vor: Voronoi,
    region_indices: np.ndarray,
    seed_colors: np.ndarray,
):
    all_verts = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for seed_idx in tqdm(region_indices, desc="Building Voronoi mesh"):
        region = vor.regions[vor.point_region[seed_idx]]
        if not region or (-1 in region):
            continue

        try:
            hull = trimesh.convex.convex_hull(vor.vertices[region])
        except Exception:
            continue

        hv = np.asarray(hull.vertices)
        hf = np.asarray(hull.faces, dtype=np.int32) + vertex_offset

        all_verts.append(hv)
        all_faces.append(hf)
        all_colors.extend([seed_colors[seed_idx]] * len(hull.faces))
        vertex_offset += hv.shape[0]

    if not all_verts:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3)),
        )

    return np.vstack(all_verts), np.vstack(all_faces), np.vstack(all_colors)


def extract_voronoi_mesh(
    model,
    sdf_min=-0.02,
    sdf_max=0.05,
    alpha_min=0.096,
    alpha_max=1.0,
    pad_ratio=0.25,
    out_base="voronoi",
):
    model.eval()

    with torch.no_grad():
        points = model.primal_points.detach().cpu().numpy()
        num_points = points.shape[0]
        sdf = (
            model.sdf_network(model.primal_points)
            .view(-1)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        alpha = model.alpha.view(-1).detach().float().cpu().numpy()
        colors = (model.att_dc[:, :3] + 0.5).clamp(0, 1).detach().float().cpu().numpy()

    mask = (sdf > sdf_min) & (sdf < sdf_max)
    mask &= (alpha > alpha_min) & (alpha < alpha_max)

    selected = np.where(mask)[0]
    print(f"Selected {len(selected):,}/{num_points:,} seeds")

    vor, n_real = make_bounded_voronoi(points.astype(np.float64), pad_ratio)
    selected = selected[selected < n_real]

    verts, faces, face_colors = reconstruct_voronoi_mesh_from_indices(
        vor, selected, colors
    )
    save_mesh_with_colors(out_base, verts, faces, face_colors)


def main():
    parser = configargparse.ArgParser()
    model_params = ModelParams(parser)
    dataset_params = DatasetParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)

    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )
    parser.add_argument("--sdf_min", type=float, default=-0.02)
    parser.add_argument("--sdf_max", type=float, default=0.05)
    parser.add_argument("--alpha_min", type=float, default=0.096)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--pad_ratio", type=float, default=0.25)
    parser.add_argument("--out_base", type=str, default=None)

    args = parser.parse_args()
    model_args = model_params.extract(args)
    _ = dataset_params.extract(args)
    _ = pipeline_params.extract(args)
    _ = optimization_params.extract(args)

    checkpoint = Path(args.config).parent
    device = torch.device(model_args.device)

    print("Loading model...")
    model = SDFoamScene(args=model_args, device=device, attr_dtype=torch.float16)
    model.load_pt(str(checkpoint / "model.pt"))
    model.eval()
    print("Model loaded.")

    out_base = args.out_base
    if out_base is None:
        out_base = (
            f"voronoi_"
            f"sdf_{sanitize_val(args.sdf_min)}_{sanitize_val(args.sdf_max)}_"
            f"alpha_{sanitize_val(args.alpha_min)}_{sanitize_val(args.alpha_max)}"
        )

    #change parameters to define the intervals of extraction
    extract_voronoi_mesh(
        model,
        sdf_min=args.sdf_min,
        sdf_max=args.sdf_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        pad_ratio=args.pad_ratio,
        out_base=out_base,
    )


if __name__ == "__main__":
    main()
