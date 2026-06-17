# ================================================================
#  GLOBAL VORONOI + DC COLOR EXTRACTION — FULL SCRIPT
#  SDF + ALPHA FILTERING, UNLIT VIEWER, COLORED MESH EXPORT
# ================================================================

import numpy as np
import torch
import open3d as o3d
import trimesh
from scipy.spatial import Voronoi
from tqdm import tqdm
import itertools
from pathlib import Path


# ===================================================================
# Utilities
# ===================================================================

def save_mesh_with_colors(path_base, verts, faces, face_colors):
    """
    Save Voronoi mesh with real colors:
    - <path_base>.ply  (with per-face color)
    - <path_base>.obj + <path_base>.mtl (with per-material colors)
    """

    # -----------------------------------------------------------
    # 1) Save PLY with per-face color
    # -----------------------------------------------------------
    ply_path = path_base + ".ply"
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        # vertices
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # faces with color
        for i, face in enumerate(faces):
            r, g, b = (face_colors[i] * 255).astype(np.uint8)
            f.write(f"{len(face)} {' '.join(str(int(x)) for x in face)} {r} {g} {b}\n")

    print(f"💾 Saved PLY with per-face colors: {ply_path}")

    # -----------------------------------------------------------
    # 2) Save OBJ + MTL with face materials
    # -----------------------------------------------------------
    obj_path = path_base + ".obj"
    mtl_path = path_base + ".mtl"

    with open(mtl_path, "w") as mtl:
        for i, col in enumerate(face_colors):
            r, g, b = col
            mtl.write(f"newmtl m{i}\n")
            mtl.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")  # diffuse color
            mtl.write("Ka 0 0 0\nKs 0 0 0\n\n")

    with open(obj_path, "w") as obj:
        obj.write(f"mtllib {mtl_path.split('/')[-1]}\n")

        for v in verts:
            obj.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for i, face in enumerate(faces):
            obj.write(f"usemtl m{i}\n")
            f_idx = [idx + 1 for idx in face]
            obj.write(f"f {' '.join(str(v) for v in f_idx)}\n")

    print(f"💾 Saved OBJ+MTL with per-face materials: {obj_path}")


def sanitize_val(v):
    """Used for file naming."""
    s = f"{v:.4f}"
    return s.replace('-', 'm').replace('.', 'p')


# ===================================================================
# GLOBAL BOUNDED VORONOI
# ===================================================================

def make_bounded_voronoi(all_points_np: np.ndarray, pad_ratio: float = 0.25):
    """
    Build a *global* bounded 3D Voronoi diagram:
    - Add 8 bounding-box corner 'ghost points'
    - Ensures all Voronoi regions are finite
    """
    pmin = all_points_np.min(axis=0)
    pmax = all_points_np.max(axis=0)
    span = pmax - pmin
    pad = pad_ratio * float(np.linalg.norm(span) + 1e-9)

    lo = pmin - pad
    hi = pmax + pad

    corners = np.array(list(itertools.product(
        [lo[0], hi[0]],
        [lo[1], hi[1]],
        [lo[2], hi[2]]
    )), dtype=np.float64)

    pts_bounded = np.vstack([all_points_np, corners])
    vor = Voronoi(pts_bounded)

    n_real = all_points_np.shape[0]
    return vor, n_real


# ===================================================================
# RECONSTRUCT SELECTED VORONOI REGIONS INTO TRIANGLE MESH
# ===================================================================

def reconstruct_voronoi_mesh_from_indices(
    vor: Voronoi,
    region_indices: np.ndarray,
    seed_colors: np.ndarray,
    show_progress: bool = True
):
    """
    Turn selected Voronoi cells (via region_indices) into:
        verts: (V,3)
        faces: (F,3)
        face_colors: (F,3)

    Colors are per-face, taken from seed_colors[seed_idx]
    """

    all_verts = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    iterator = tqdm(region_indices, desc="🔧 Building Voronoi mesh") if show_progress else region_indices

    for seed_idx in iterator:

        r_idx = vor.point_region[seed_idx]
        region = vor.regions[r_idx]

        # Skip infinite / invalid cells
        if not region or (-1 in region):
            continue

        # Extract region vertices
        region_verts = vor.vertices[region]

        # Triangulate via convex hull
        try:
            hull = trimesh.convex.convex_hull(region_verts)
        except Exception:
            continue

        hull_verts = np.asarray(hull.vertices)
        hull_faces = np.asarray(hull.faces, dtype=np.int32) + vertex_offset

        # Store geometry
        all_verts.append(hull_verts)
        all_faces.append(hull_faces)

        # Assign DC color to each face of this region
        color = seed_colors[seed_idx]
        for _ in range(len(hull.faces)):
            all_colors.append(color)

        vertex_offset += hull_verts.shape[0]

    if len(all_verts) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32),
                np.zeros((0, 3)))

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    face_colors = np.vstack(all_colors)

    return verts, faces, face_colors


# ================================================================
#  OPEN3D VIEWER — SDF + ALPHA FILTERS + DC COLORS + VORONOI MESH
# ================================================================

def show_sdf_pointcloud_o3d(model,
                             init_sdf_min=-0.02,   # good SDF range near surface
                             init_sdf_max=+0.02,
                             window_title="SDF + Alpha Filter (Open3D)"):

    from open3d.visualization import gui, rendering

    # -----------------------------
    # Load primal points + SDF + alpha + DC
    # -----------------------------
    with torch.no_grad():
        pts = model.primal_points.detach().cpu().numpy()    # [N,3]

        # Detect SDF network
        has_sdf = hasattr(model, "sdf_network") and (model.sdf_network is not None)

        if has_sdf:
            sdf = model.sdf_network(model.primal_points) \
                        .view(-1).detach().cpu().numpy()
        else:
            sdf = None

        has_alpha = hasattr(model, "alpha") and model.alpha is not None
        if has_alpha:
            alpha = model.alpha.detach().view(-1).cpu().numpy()
        else:
            alpha = None

        # ---- DC colors ----
        if hasattr(model, "att_dc"):
            dc_raw = model.att_dc[:, :3].detach().cpu().float()
            # RadiantFoam convention: DC ≈ [-0.5, +0.5] → map to [0,1]
            dc = (dc_raw + 0.5).clamp(0.0, 1.0).numpy()
        else:
            dc = np.ones((pts.shape[0], 3), dtype=np.float32)

    N = pts.shape[0]

    # Alpha initial range = full dynamic range (if present)
    if has_alpha:
        a_lo = float(alpha.min())
        a_hi = float(alpha.max())
        if abs(a_hi - a_lo) < 1e-6:
            a_lo, a_hi = a_lo - 0.5, a_hi + 0.5
    else:
        a_lo, a_hi = -1.0, 1.0   # unused

    # -----------------------------
    # Build global-bounded Voronoi
    # -----------------------------
    pts64 = pts.astype(np.float64)
    vor_full, n_real = make_bounded_voronoi(pts64, pad_ratio=0.25)

    # -----------------------------
    # Init Open3D GUI
    # -----------------------------
    gui.Application.instance.initialize()
    w = gui.Application.instance.create_window(window_title, 1280, 800)

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(w.renderer)
    scene.scene.set_background([0, 0, 0, 1])

    mat_points = rendering.MaterialRecord()
    mat_points.shader = "defaultUnlit"
    mat_points.point_size = 3.0      # default size

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultUnlit"
    mat_mesh.base_color = (1, 1, 1, 1)

    # -----------------------------
    # Point cloud (dynamic)
    # -----------------------------
    pcd = o3d.geometry.PointCloud()

    current = {
        "pts":   np.zeros((0, 3)),
        "mask":  np.zeros(N, dtype=bool),
        "smin":  float(init_sdf_min),
        "smax":  float(init_sdf_max),
        "amin":  float(a_lo),
        "amax":  float(a_hi),
    }

    em = w.theme.font_size

    # -----------------------------
    # Side panel
    # -----------------------------
    panel = gui.Vert(0.5 * em,
                     gui.Margins(1 * em, 1 * em, 1 * em, 1 * em))

    title = gui.Label("SDF + alpha range filters")
    title.text_color = gui.Color(0.8, 0.8, 0.8)
    panel.add_child(title)

    # -----------------------------
    # SDF MIN controls
    # -----------------------------
    if has_sdf:
        row_sdf_min = gui.Horiz(0.2 * em)
        row_sdf_min.add_child(gui.Label("SDF min"))

        s_min = gui.Slider(gui.Slider.DOUBLE)
        s_min.set_limits(-1.0, 1.0)
        s_min.double_value = float(init_sdf_min)
        row_sdf_min.add_child(s_min)

        ne_smin = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne_smin.set_limits(-1.0, 1.0)
        ne_smin.double_value = float(init_sdf_min)
        row_sdf_min.add_child(ne_smin)

        panel.add_child(row_sdf_min)

        # -----------------------------
        # SDF MAX controls
        # -----------------------------
        row_sdf_max = gui.Horiz(0.2 * em)
        row_sdf_max.add_child(gui.Label("SDF max"))

        s_max = gui.Slider(gui.Slider.DOUBLE)
        s_max.set_limits(-1.0, 1.0)
        s_max.double_value = float(init_sdf_max)
        row_sdf_max.add_child(s_max)

        ne_smax = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne_smax.set_limits(-1.0, 1.0)
        ne_smax.double_value = float(init_sdf_max)
        row_sdf_max.add_child(ne_smax)

        panel.add_child(row_sdf_max)

    # -----------------------------
    # ALPHA controls (only if model has alpha)
    # -----------------------------
    if has_alpha:
        row_a_min = gui.Horiz(0.2 * em)
        row_a_min.add_child(gui.Label("alpha min"))

        a_min_slider = gui.Slider(gui.Slider.DOUBLE)
        # use expanded dynamic range of alpha
        a_range = max(1e-6, a_hi - a_lo)
        a_min_lim = a_lo - 0.1 * a_range
        a_max_lim = a_hi + 0.1 * a_range
        a_min_slider.set_limits(a_min_lim, a_max_lim)
        a_min_slider.double_value = a_lo
        row_a_min.add_child(a_min_slider)

        ne_amin = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne_amin.set_limits(a_min_lim, a_max_lim)
        ne_amin.double_value = a_lo
        row_a_min.add_child(ne_amin)
        panel.add_child(row_a_min)

        row_a_max = gui.Horiz(0.2 * em)
        row_a_max.add_child(gui.Label("alpha max"))

        a_max_slider = gui.Slider(gui.Slider.DOUBLE)
        a_max_slider.set_limits(a_min_lim, a_max_lim)
        a_max_slider.double_value = a_hi
        row_a_max.add_child(a_max_slider)

        ne_amax = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        ne_amax.set_limits(a_min_lim, a_max_lim)
        ne_amax.double_value = a_hi
        row_a_max.add_child(ne_amax)
        panel.add_child(row_a_max)
    else:
        a_min_slider = a_max_slider = ne_amin = ne_amax = None

    # -----------------------------
    # Status text
    # -----------------------------
    count_label = gui.Label("")
    panel.add_child(count_label)

    # Buttons
    row_btn = gui.Horiz(0.2 * em)
    btn_reset = gui.Button("Reset")
    btn_fit   = gui.Button("Fit Camera")
    row_btn.add_child(btn_reset)
    row_btn.add_child(btn_fit)
    panel.add_child(row_btn)

    # ------------------------------------
    # POINT SIZE SLIDER
    # ------------------------------------
    row_ps = gui.Horiz(0.2 * em)
    row_ps.add_child(gui.Label("Point Size"))

    s_ps = gui.Slider(gui.Slider.DOUBLE)
    s_ps.set_limits(1.0, 20.0)
    s_ps.double_value = 3.0
    row_ps.add_child(s_ps)

    ne_ps = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    ne_ps.set_limits(1.0, 20.0)
    ne_ps.double_value = 3.0
    row_ps.add_child(ne_ps)

    panel.add_child(row_ps)

    # slider sync
    def on_ps_change(val):
        size = float(val)
        mat_points.point_size = size

        # Must re-add geometry for size change to apply
        if scene.scene.has_geometry("points"):
            scene.scene.remove_geometry("points")
        scene.scene.add_geometry("points", pcd, mat_points)

    s_ps.set_on_value_changed(on_ps_change)
    ne_ps.set_on_value_changed(on_ps_change)


    # Extraction button
    row_extract = gui.Horiz(0.2 * em)
    btn_extract = gui.Button("Extract Voronoi Mesh")
    row_extract.add_child(btn_extract)
    panel.add_child(row_extract)

    status_label = gui.Label("")
    panel.add_child(status_label)

    w.add_child(scene)
    w.add_child(panel)

    # -----------------------------
    # Layout
    # -----------------------------
    def on_layout(ctx):
        r = w.content_rect
        panel_w = int(20 * em)
        panel.frame = gui.Rect(r.x, r.y, panel_w, r.height)
        scene.frame = gui.Rect(r.x + panel_w, r.y,
                               r.width - panel_w, r.height)

    w.set_on_layout(on_layout)

    # -----------------------------
    # Fit camera
    # -----------------------------
    def frame_full_cloud():
        if pts64.shape[0] == 0:
            return
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(pts64)
        )
        scene.setup_camera(60.0, bbox, bbox.get_center())

    # -----------------------------
    # Coloring (DC)
    # -----------------------------
    def get_colors(dc_slice):
        return dc_slice.astype(np.float64)

    # -----------------------------
    # Redraw filtered point cloud
    # -----------------------------
    def apply_filter_and_redraw(smin, smax, amin, amax):
        mask = np.ones(N, dtype=bool)

        if has_sdf:
            mask &= (sdf > smin) & (sdf < smax)

        if has_alpha:
            mask &= (alpha > amin) & (alpha < amax)

        fpts = pts64[mask]
        fdc  = dc[mask]

        current["pts"]  = fpts
        current["mask"] = mask
        current["smin"] = float(smin)
        current["smax"] = float(smax)
        current["amin"] = float(amin)
        current["amax"] = float(amax)

        msg = f"SDF in ({smin:.4f}, {smax:.4f})"
        if has_alpha:
            msg += f" | alpha in ({amin:.4f}, {amax:.4f})"

        count_label.text = (f"Showing {len(fpts):,}/{N:,} points | " + msg)

        pcd.points = o3d.utility.Vector3dVector(fpts)
        pcd.colors = o3d.utility.Vector3dVector(get_colors(fdc))

        name = "points"
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, pcd, mat_points)

    # -----------------------------
    # Slider sync logic
    # -----------------------------
    eps = 1e-4
    _sync_sdf = {"flag": False}
    _sync_alpha = {"flag": False}

    def sync_sdf_min(val):
        if _sync_sdf["flag"]:
            return
        _sync_sdf["flag"] = True

        vmin = float(val)
        vmax = s_max.double_value
        if vmin >= vmax - eps:
            vmax = vmin + eps
            s_max.double_value = vmax
            ne_smax.double_value = vmax

        s_min.double_value = vmin
        ne_smin.double_value = vmin

        apply_filter_and_redraw(
            vmin, s_max.double_value,
            current["amin"], current["amax"]
        )
        _sync_sdf["flag"] = False

    def sync_sdf_max(val):
        if _sync_sdf["flag"]:
            return
        _sync_sdf["flag"] = True

        vmax = float(val)
        vmin = s_min.double_value
        if vmax <= vmin + eps:
            vmin = vmax - eps
            s_min.double_value = vmin
            ne_smin.double_value = vmin

        s_max.double_value = vmax
        ne_smax.double_value = vmax

        apply_filter_and_redraw(
            s_min.double_value, vmax,
            current["amin"], current["amax"]
        )
        _sync_sdf["flag"] = False

    if has_sdf:
        s_min.set_on_value_changed(sync_sdf_min)
        ne_smin.set_on_value_changed(sync_sdf_min)
        s_max.set_on_value_changed(sync_sdf_max)
        ne_smax.set_on_value_changed(sync_sdf_max)

    # Alpha sync
    if has_alpha:
        def sync_alpha_min(val):
            if _sync_alpha["flag"]:
                return
            _sync_alpha["flag"] = True

            vmin = float(val)
            vmax = a_max_slider.double_value
            if vmin >= vmax - eps:
                vmax = vmin + eps
                a_max_slider.double_value = vmax
                ne_amax.double_value = vmax

            a_min_slider.double_value = vmin
            ne_amin.double_value = vmin

            apply_filter_and_redraw(
                current["smin"], current["smax"],
                vmin, a_max_slider.double_value
            )
            _sync_alpha["flag"] = False

        def sync_alpha_max(val):
            if _sync_alpha["flag"]:
                return
            _sync_alpha["flag"] = True

            vmax = float(val)
            vmin = a_min_slider.double_value
            if vmax <= vmin + eps:
                vmin = vmax - eps
                a_min_slider.double_value = vmin
                ne_amin.double_value = vmin

            a_max_slider.double_value = vmax
            ne_amax.double_value = vmax

            apply_filter_and_redraw(
                current["smin"], current["smax"],
                a_min_slider.double_value, vmax
            )
            _sync_alpha["flag"] = False

        a_min_slider.set_on_value_changed(sync_alpha_min)
        ne_amin.set_on_value_changed(sync_alpha_min)
        a_max_slider.set_on_value_changed(sync_alpha_max)
        ne_amax.set_on_value_changed(sync_alpha_max)

    # -----------------------------
    # Button logic
    # -----------------------------
    def on_reset():
        # Reset SDF
        if has_sdf:
            s_min.double_value = init_sdf_min
            ne_smin.double_value = init_sdf_min
            s_max.double_value = init_sdf_max
            ne_smax.double_value = init_sdf_max

        # Reset alpha to full range (if present)
        if has_alpha:
            a_min_slider.double_value = a_lo
            ne_amin.double_value = a_lo
            a_max_slider.double_value = a_hi
            ne_amax.double_value = a_hi

        apply_filter_and_redraw(init_sdf_min, init_sdf_max,
                                a_lo if has_alpha else current["amin"],
                                a_hi if has_alpha else current["amax"])

    def on_fit():
        frame_full_cloud()

    btn_reset.set_on_clicked(on_reset)
    btn_fit.set_on_clicked(on_fit)

    # -----------------------------
    # Mesh extraction
    # -----------------------------
    def on_extract():
        mask = current["mask"]
        smin, smax = current["smin"], current["smax"]
        amin, amax = current["amin"], current["amax"]

        sel_idx = np.where(mask[:n_real])[0]

        if sel_idx.size == 0:
            status_label.text = "❌ No cells selected."
            return

        status_label.text = "⏳ Extracting mesh..."

        try:
            region_indices = np.array(
                [vor_full.point_region[i] for i in sel_idx],
                dtype=np.int32
            )

            verts, faces, face_colors = reconstruct_voronoi_mesh_from_indices(
                vor_full, sel_idx, seed_colors=dc, show_progress=False
            )
        except Exception as e:
            status_label.text = f"❌ Extraction failed: {e}"
            return

        if verts.shape[0] == 0:
            status_label.text = "❌ No valid Voronoi cells."
            return

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(faces.astype(np.int32))
        )
        mesh.compute_vertex_normals()

        # Per-vertex color fallback (Open3D <=0.17)
        vertex_colors = np.zeros((verts.shape[0], 3))
        for fi, face in enumerate(faces):
            col = face_colors[fi]
            for vid in face:
                vertex_colors[vid] = col
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        name = "voronoi_mesh"
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, mesh, mat_mesh)

        parts = []

        if has_sdf:
            parts.append(f"sdf_{sanitize_val(smin)}_{sanitize_val(smax)}")

        if has_alpha:
            parts.append(f"alpha_{sanitize_val(amin)}_{sanitize_val(amax)}")

        base = "voronoi_" + "_".join(parts)


        save_mesh_with_colors(base, verts, faces, face_colors)
        status_label.text = f"✅ Saved mesh as {base}.ply and {base}.obj"

    btn_extract.set_on_clicked(on_extract)

    # -----------------------------
    # Initial draw
    # -----------------------------
    apply_filter_and_redraw(init_sdf_min, init_sdf_max, a_lo, a_hi)
    frame_full_cloud()
    gui.Application.instance.run()


# ================================================================
#  PIPELINE GLUE — LAUNCH VIEWER WITH MODEL + CONFIG
# ================================================================

from data_loader import DataHandler
from configs import *
from sdfoam_model.scene import SDFoamScene


def viewer(args, pipeline_args, model_args, optimizer_args, dataset_args):
    """
    Wrapper that loads the model & dataset parameters,
    then launches the Open3D SDF+Alpha-colored Voronoi viewer.
    """

    # Resolve checkpoint path (folder)
    checkpoint = Path(args.config).parent
    device = torch.device(model_args.device)

    # Load test dataset for camera configuration
    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(
        split="test",
        downsample=min(dataset_args.downsample)
    )

    # Camera view (from dataset) — not used directly here, but kept if you
    # later want to sync Open3D camera with RF viewer.
    viewer_options = {
        "camera_pos":     test_data_handler.viewer_pos,
        "camera_up":      test_data_handler.viewer_up,
        "camera_forward": test_data_handler.viewer_forward,
        "use_neus":       getattr(model_args, "use_neus", False),
    }

    # ---------------------------------------------------------------
    # Load RadiantFoam / SDFoam model
    # ---------------------------------------------------------------
    print("📦 Loading model...")
    model = SDFoamScene(
        args=model_args,
        device=device,
        attr_dtype=torch.float16,
    )

    model.load_pt(str(checkpoint / "model.pt"))
    print("✔ Model loaded.")

    # =======================================================
    # Print SDF & alpha statistics
    # =======================================================
    with torch.no_grad():
        if hasattr(model, "sdf_network"):
            sdf_values = model.sdf_network(model.primal_points).view(-1)
            sdf_np = sdf_values.cpu().numpy()
            print(f"[SDF] min={sdf_np.min():.6f}, max={sdf_np.max():.6f}, mean={sdf_np.mean():.6f}")
            print(f"[SDF] percentiles: 1%={np.percentile(sdf_np,1):.6f}, "
                f"5%={np.percentile(sdf_np,5):.6f}, "
                f"95%={np.percentile(sdf_np,95):.6f}, 99%={np.percentile(sdf_np,99):.6f}")

        if hasattr(model, "alpha") and model.alpha is not None:
            alpha_np = model.alpha.view(-1).detach().cpu().numpy()
            print(f"[alpha]   min={alpha_np.min():.6f}, max={alpha_np.max():.6f}, mean={alpha_np.mean():.6f}")
            print(f"[alpha]   percentiles: 1%={np.percentile(alpha_np,1):.6f}, "
                  f"5%={np.percentile(alpha_np,5):.6f}, "
                  f"95%={np.percentile(alpha_np,95):.6f}, 99%={np.percentile(alpha_np,99):.6f}")

    # ---------------------------------------------------------------
    # Launch Open3D interactive viewer
    # ---------------------------------------------------------------
    show_sdf_pointcloud_o3d(
        model,
        init_sdf_min=-0.02,     # good default around surface
        init_sdf_max=+0.02,
        window_title="SDFoam Voronoi Viewer (DC Colors + SDF + alpha Filters)"
    )


# ================================================================
#  MAIN ENTRY POINT
# ================================================================

def main():
    import configargparse
    from configs import ModelParams, DatasetParams, PipelineParams, OptimizationParams

    parser = configargparse.ArgParser()

    model_params        = ModelParams(parser)
    dataset_params      = DatasetParams(parser)
    pipeline_params     = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)

    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    args = parser.parse_args()

    viewer(
        args,
        pipeline_params.extract(args),
        model_params.extract(args),
        optimization_params.extract(args),
        dataset_params.extract(args),
    )


if __name__ == "__main__":
    main()
