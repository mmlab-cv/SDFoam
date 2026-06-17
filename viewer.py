import numpy as np
from PIL import Image
import configargparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

from data_loader import DataHandler
from configs import *
from sdfoam_model.scene import SDFoamScene

from sdfoam_model.sdfnet import SDFNetwork, SingleVarianceNetwork
from skimage import measure

import numpy as np
import torch
import trimesh
from scipy.spatial import Voronoi
#from typing import Optional, Tuple
from tqdm import tqdm
#import itertools

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

import torch
import numpy as np
from scipy.spatial import Voronoi
import trimesh

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def select_centroids_in_sdf_range(
    sdf_values,
    sdf_exact_mode=False,
    sdf_value=0.0,
    sdf_tolerance=0.1,
    max_sdf=2.0
):

    sdf_values = torch.as_tensor(sdf_values).view(-1)

    if sdf_exact_mode:
        tol = max(sdf_tolerance, 0.0)
        mask = torch.abs(sdf_values - sdf_value) <= tol
    else:
        tau = max(max_sdf, 1e-6)
        mask = torch.abs(sdf_values) <= tau

    selected_ids = torch.nonzero(mask, as_tuple=False).view(-1)

    print(f"✅ Selezionati {selected_ids.numel()} centroidi su {len(sdf_values)} "
          f"({100 * selected_ids.numel() / len(sdf_values):.2f}% del totale)")

    return selected_ids


def extract_voronoi_cells(selected_indices, model):
    
    primal_points = model.primal_points.detach()
    
    vertex_cache = {}
    vertex_list = []
    
    cell_to_vertices = {}
    cell_to_faces = {}
    all_faces = []
    
    def coords_to_key(coords, tolerance=1e-6):
        rounded = np.round(coords / tolerance) * tolerance
        return tuple(rounded)
    
    # tqdm qui
    for idx in tqdm(selected_indices, desc="Extracting Voronoi cells"):
        idx_item = idx.item()
        center = primal_points[idx_item].cpu().numpy()
        
        start = model.point_adjacency_offsets[idx_item]
        end = model.point_adjacency_offsets[idx_item + 1]
        neighbors = model.point_adjacency[start:end]
        
        if len(neighbors) < 3:
            continue
        
        local_points = [center]
        neighbor_items = []
        
        for neighbor in neighbors:
            neighbor_item = neighbor.item()
            neighbor_point = primal_points[neighbor_item].cpu().numpy()
            local_points.append(neighbor_point)
            neighbor_items.append(neighbor_item)
        
        local_points = np.array(local_points)
        
        try:
            vor = Voronoi(local_points)
        except Exception as e:
            print(f"Warning: Could not compute Voronoi for cell {idx_item}: {e}")
            continue
        
        center_region_idx = vor.point_region[0]
        center_region = vor.regions[center_region_idx]
        
        if -1 in center_region or len(center_region) == 0:
            continue
        
        local_to_global = {}
        for local_vertex_idx in center_region:
            vertex_coords = vor.vertices[local_vertex_idx]
            vertex_key = coords_to_key(vertex_coords)
            
            if vertex_key not in vertex_cache:
                global_idx = len(vertex_list)
                vertex_tensor = torch.tensor(vertex_coords, device=model.device, dtype=torch.float32)
                vertex_list.append(vertex_tensor)
                vertex_cache[vertex_key] = global_idx
            else:
                global_idx = vertex_cache[vertex_key]
            
            local_to_global[local_vertex_idx] = global_idx
        
        cell_vertex_indices = [local_to_global[v] for v in center_region]
        cell_to_vertices[idx_item] = cell_vertex_indices
        
        cell_faces_list = []
        for i, neighbor_item in enumerate(neighbor_items):
            neighbor_local_idx = i + 1
            neighbor_region_idx = vor.point_region[neighbor_local_idx]
            neighbor_region = vor.regions[neighbor_region_idx]
            
            if -1 in neighbor_region or len(neighbor_region) == 0:
                continue
            
            shared_vertices_local = set(center_region) & set(neighbor_region)
            
            if len(shared_vertices_local) >= 3:
                face_vertex_indices = [local_to_global[v] for v in shared_vertices_local if v in local_to_global]
                
                if len(face_vertex_indices) >= 3:
                    face_vertices = torch.stack([vertex_list[i] for i in face_vertex_indices])
                    ordered_indices = order_face_vertices(
                        face_vertices,
                        face_vertex_indices,
                        primal_points[idx_item],
                        primal_points[neighbor_item]
                    )
                    
                    face_idx = len(all_faces)
                    all_faces.append(ordered_indices)
                    cell_faces_list.append(face_idx)
        
        cell_to_faces[idx_item] = cell_faces_list
    
    if len(vertex_list) > 0:
        vertices = torch.stack(vertex_list)
    else:
        vertices = torch.empty((0, 3), device=model.device)
    
    return vertices, all_faces, cell_to_vertices, cell_to_faces

def extract_voronoi_surface(selected_indices,
                            model,
                            sdf_threshold=0.02,
                            pointcloud_only=False,
                            mode="sdf",
                            density_threshold=0.001):
    
    primal_points = model.primal_points.detach()

    vertex_cache = {}
    vertex_list = []

    cell_to_vertices = {}
    cell_to_faces = {}
    all_faces = []

    def coords_to_key(coords, tolerance=1e-6):
        rounded = np.round(coords / tolerance) * tolerance
        return tuple(rounded)

    for idx in tqdm(selected_indices, desc="Extracting Voronoi cells"):
        idx_item = idx.item()
        center = primal_points[idx_item].cpu().numpy()

        start = model.point_adjacency_offsets[idx_item]
        end = model.point_adjacency_offsets[idx_item + 1]
        neighbors = model.point_adjacency[start:end]

        if len(neighbors) < 3:
            continue

        local_points = [center]
        neighbor_items = []

        for neighbor in neighbors:
            neighbor_item = neighbor.item()
            neighbor_point = primal_points[neighbor_item].cpu().numpy()
            local_points.append(neighbor_point)
            neighbor_items.append(neighbor_item)

        local_points = np.array(local_points)

        try:
            vor = Voronoi(local_points)
        except Exception as e:
            print(f"⚠️ Warning: Could not compute Voronoi for cell {idx_item}: {e}")
            continue

        center_region_idx = vor.point_region[0]
        center_region = vor.regions[center_region_idx]

        if -1 in center_region or len(center_region) == 0:
            continue

        # Costruisci i vertici globali unici
        local_to_global = {}
        for local_vertex_idx in center_region:
            vertex_coords = vor.vertices[local_vertex_idx]
            vertex_key = coords_to_key(vertex_coords)

            if vertex_key not in vertex_cache:
                global_idx = len(vertex_list)
                vertex_tensor = torch.tensor(vertex_coords, device=model.device, dtype=torch.float32)
                vertex_list.append(vertex_tensor)
                vertex_cache[vertex_key] = global_idx
            else:
                global_idx = vertex_cache[vertex_key]

            local_to_global[local_vertex_idx] = global_idx

        cell_vertex_indices = [local_to_global[v] for v in center_region]
        cell_to_vertices[idx_item] = cell_vertex_indices

        # Se siamo in modalità pointcloud, non costruiamo le facce
        if pointcloud_only:
            continue

        # Costruzione facce
        cell_faces_list = []
        for i, neighbor_item in enumerate(neighbor_items):
            neighbor_local_idx = i + 1
            neighbor_region_idx = vor.point_region[neighbor_local_idx]
            neighbor_region = vor.regions[neighbor_region_idx]

            if -1 in neighbor_region or len(neighbor_region) == 0:
                continue

            shared_vertices_local = set(center_region) & set(neighbor_region)

            if len(shared_vertices_local) < 3:
                continue

            face_vertex_indices = [local_to_global[v] for v in shared_vertices_local if v in local_to_global]
            if len(face_vertex_indices) < 3:
                continue

            # --- 🔍 FILTRAGGIO DELLE FACCE ---
            keep_face = True

            if mode == "sdf":
                # mantieni faccia solo se almeno 3 vertici hanno |SDF| < soglia
                face_vertices = torch.stack([vertex_list[k] for k in face_vertex_indices])
                with torch.no_grad():
                    sdf_vals = model.sdf_network(face_vertices).view(-1)
                    near_surface = (torch.abs(sdf_vals) < sdf_threshold)
                    if near_surface.sum().item() < 3:
                        keep_face = False

            elif mode == "density":
                # 🔹 mantieni faccia solo se cella corrente sopra soglia e vicina sotto soglia
                if hasattr(model, "density") and model.density is not None:
                    dens_i = model.density[idx_item].item()
                    dens_j = model.density[neighbor_item].item()
                else:
                    # fallback: valuta con density_network
                    with torch.no_grad():
                        dens_i = model.density_network(primal_points[idx_item].unsqueeze(0)).item()
                        dens_j = model.density_network(primal_points[neighbor_item].unsqueeze(0)).item()

                if not (dens_i > density_threshold and dens_j < density_threshold):
                    keep_face = False

            # se il filtro scarta la faccia, passa oltre
            if not keep_face:
                continue

            # --- ordina i vertici della faccia ---
            face_vertices = torch.stack([vertex_list[k] for k in face_vertex_indices])
            ordered_indices = order_face_vertices(
                face_vertices,
                face_vertex_indices,
                primal_points[idx_item],
                primal_points[neighbor_item]
            )

            face_idx = len(all_faces)
            all_faces.append(ordered_indices)
            cell_faces_list.append(face_idx)

        cell_to_faces[idx_item] = cell_faces_list

    # --- Costruzione tensore finale dei vertici ---
    if len(vertex_list) > 0:
        vertices = torch.stack(vertex_list)
    else:
        vertices = torch.empty((0, 3), device=model.device)

    # --- Filtraggio pointcloud-only ---
    if pointcloud_only:
        if vertices.shape[0] == 0:
            filtered_vertices = vertices
        else:
            if mode == "sdf":
                with torch.no_grad():
                    sdf_vals = model.sdf_network(vertices).view(-1)
                    mask = (torch.abs(sdf_vals) < sdf_threshold)
                    selected_count = int(mask.sum().item())
                    print(f"✅ Vertici totali Voronoi: {vertices.shape[0]}, dopo filtro SDF (<{sdf_threshold}): {selected_count}")
                    filtered_vertices = vertices[mask] if selected_count > 0 else torch.empty((0,3), device=model.device)
            elif mode == "density":
                if hasattr(model, "density_network"):
                    with torch.no_grad():
                        dens_vals = model.density_network(vertices).view(-1)
                        mask = (dens_vals > density_threshold)
                        selected_count = int(mask.sum().item())
                        print(f"✅ Vertici totali Voronoi: {vertices.shape[0]}, dopo filtro DENSITY (>{density_threshold}): {selected_count}")
                        filtered_vertices = vertices[mask] if selected_count > 0 else torch.empty((0,3), device=model.device)
                elif hasattr(model, "density") and model.density is not None:
                    # fallback: nearest primal density (approssimato)
                    dens_vals = model.density.mean().repeat(vertices.shape[0])
                    filtered_vertices = vertices
                else:
                    filtered_vertices = vertices
            else:
                filtered_vertices = vertices

        return filtered_vertices, [], {}, {}

    else:
        # comportamento completo con facce
        return vertices, all_faces, cell_to_vertices, cell_to_faces

def save_pointcloud_ply(file_path, vertices):
    """
    Salva un PLY ASCII contenente solo vertici (point cloud).
    vertices: torch.Tensor [N,3] (può essere su device GPU)
    """
    vertices_np = vertices.detach().cpu().numpy()
    with open(file_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for v in vertices_np:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
    print(f"✅ Point cloud PLY saved: {file_path} (points: {len(vertices_np)})")


def order_face_vertices(face_verts, face_indices, center_point, neighbor_point):

    if len(face_verts) < 3:
        return face_indices
    
    # Calcola il centroide della faccia
    centroid = face_verts.mean(dim=0)
    
    # Vettore normale alla faccia (direzione dalla cella al vicino)
    normal = neighbor_point - center_point
    normal = normal / torch.norm(normal)
    
    # Proietta i vertici sul piano della faccia e calcola gli angoli
    angles = []
    ref_vec = face_verts[0] - centroid
    ref_vec = ref_vec - torch.dot(ref_vec, normal) * normal
    ref_vec_norm = torch.norm(ref_vec)
    
    if ref_vec_norm < 1e-10:
        return face_indices
    
    ref_vec = ref_vec / ref_vec_norm
    
    for v in face_verts:
        vec = v - centroid
        vec = vec - torch.dot(vec, normal) * normal
        vec_norm = torch.norm(vec)
        
        if vec_norm < 1e-10:
            angles.append(0.0)
            continue
        
        vec = vec / vec_norm
        
        cos_angle = torch.dot(vec, ref_vec).clamp(-1, 1)
        cross = torch.cross(ref_vec, vec)
        sin_angle = torch.dot(cross, normal)
        
        angle = torch.atan2(sin_angle, cos_angle)
        angles.append(angle.item())
    
    # Ordina gli indici per angolo
    sorted_pairs = sorted(zip(angles, face_indices), key=lambda x: x[0])
    return [idx for _, idx in sorted_pairs]

def save_mesh_obj(file_path, vertices, faces):

    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")
        
        for face in faces:
            face_indices = [idx + 1 for idx in face]  # 1-based
            f.write("f " + " ".join(map(str, face_indices)) + "\n")
    
    print(f"✅ Mesh saved in OBJ: {file_path}")

def save_mesh_ply(file_path, vertices, faces):

    vertices = vertices.cpu().numpy()
    
    with open(file_path, 'w') as f:
        # Header PLY
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"{len(face)} " + " ".join(map(str, face)) + "\n")
    
    print(f"✅ Mesh saved in PLY: {file_path}")


# ============================================================================
# VIEWER FUNCTION
# ============================================================================

def viewer(args, pipeline_args, model_args, optimizer_args, dataset_args):
    checkpoint = Path(args.config).parent
    device = torch.device(args.device)

    test_data_handler = DataHandler(
        dataset_args, rays_per_batch=0, device=device
    )
    test_data_handler.reload(split="test", downsample=min(dataset_args.downsample))

    viewer_options = {
        "camera_pos": test_data_handler.viewer_pos,
        "camera_up": test_data_handler.viewer_up,
        "camera_forward": test_data_handler.viewer_forward,
        "use_sdfoam": True if args.use_sdfoam else False,
    }

    model = SDFoamScene(
        args=model_args, device=device, attr_dtype=torch.float16
    )

    model.load_pt(str(checkpoint / "model.pt"))

    
    
    # Opzionale: mostra viewer 3D
    def viewer_init(viewer):
        model.update_viewer(viewer)
    model.show(viewer_init, **viewer_options)
    quit()

    print("\n" + "="*80)
    print("EXTRACTING GEOMETRY ...")
    print("="*80 + "\n")

    print("📍 Loading primal points...")
    points = model.primal_points.to(device)

    mode = args.mode
    output_type = args.output_type.lower()  # "pcd", "mesh", or "both"

    # --- Extraction logic ---
    if mode == "sdf":
        print("🌀 Using SDF mode for Voronoi extraction")
        sdf_values = model.sdf_network(points).detach()
        N = points.shape[0]
        print(f"   Total points: {N}")

        sdf_exact_mode=True
        sdf_value=0.0
        sdf_tolerance=0.001
        max_sdf=0.01
        sdf_threshold=0.001

        selected_ids = select_centroids_in_sdf_range(
            sdf_values=sdf_values,
            sdf_exact_mode=sdf_exact_mode,
            sdf_value=sdf_value,
            sdf_tolerance=sdf_tolerance,
            max_sdf=max_sdf
        )

        if output_type in ["pcd", "both"]:
            vertices_pc, _, _, _ = extract_voronoi_surface(
                selected_ids, model, sdf_threshold=sdf_threshold,
                pointcloud_only=True, mode="sdf"
            )
            network = "sdfoam" if args.use_sdfoam else "baseline"

            save_pointcloud_ply(f"{network}_{args.scene}.ply", vertices_pc)

        if output_type in ["mesh", "both"]:
            vertices, all_faces, cell_to_vertices, cell_to_faces = extract_voronoi_surface(
                selected_ids, model, sdf_threshold=sdf_threshold,
                pointcloud_only=False, mode="sdf"
            )
            save_mesh_obj("sdfoam_mesh.obj", vertices, all_faces)
            save_mesh_ply("sdfoam_mesh.ply", vertices, all_faces)

    else:
        print("📦 Using DENSITY mode for Voronoi extraction")
        density = model.density
        dens_thr = 0.001
        selected_ids = torch.nonzero(density.view(-1) > dens_thr, as_tuple=False).view(-1)
        print(f"✅ Selezionate {len(selected_ids)} celle sopra soglia densità > {dens_thr}")

        if output_type in ["pcd", "both"]:
            vertices_pc, _, _, _ = extract_voronoi_surface(
                selected_ids, model,
                pointcloud_only=True,
                mode="density",
                density_threshold=dens_thr
            )
            network = "sdfoam" if args.use_sdfoam else "baseline"

            save_pointcloud_ply(f"{network}_{args.scene}.ply", vertices_pc)
            # save_pointcloud_ply("sdfoam_pointcloud.ply", vertices_pc)

        if output_type in ["mesh", "both"]:
            vertices, all_faces, cell_to_vertices, cell_to_faces = extract_voronoi_surface(
                selected_ids, model,
                pointcloud_only=False,
                mode="density",
                density_threshold=dens_thr
            )
            save_mesh_obj("sdfoam_mesh.obj", vertices, all_faces)
            save_mesh_ply("sdfoam_mesh.ply", vertices, all_faces)



    


def main():
    parser = configargparse.ArgParser()

    model_params = ModelParams(parser)
    dataset_params = DatasetParams(parser)
    pipeline_params = PipelineParams(parser)
    optimization_params = OptimizationParams(parser)

    parser.add_argument(
        "-c", "--config", is_config_file=True, help="Path to config file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="sdf",
        choices=["sdf", "density"],
        help="Extraction mode: 'sdf' or 'density'"
    )

    parser.add_argument(
        "--output_type",
        type=str,
        default="mesh",
        choices=["pcd", "mesh", "both"],
        help="Output type: 'pcd' (point cloud), 'mesh', or 'both'"
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
