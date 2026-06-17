import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap

import random
import math


def _resolve_pycolmap_attr(value):
    return value() if callable(value) else value


def get_world_from_cam_matrix(image):
    cam_from_world = _resolve_pycolmap_attr(image.cam_from_world)
    world_from_cam = cam_from_world.inverse()
    return world_from_cam.matrix()


def get_cam_ray_dirs(camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate(
        [ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1
    )
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


class COLMAPDataset:
    def __init__(self, datadir, eval=True, split="train", downsample=1):
        assert downsample in [1, 2, 4, 8]

        self.root_dir = datadir
        self.colmap_dir = os.path.join(datadir, "sparse/0/")
        self.eval = eval
        self.split = split
        self.downsample = downsample

        if downsample == 1:
            images_dir = os.path.join(datadir, "image")
        else:
            images_dir = os.path.join(datadir, f"image_{downsample}")

        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        #if len(self.reconstruction.cameras) > 1:
        #    raise ValueError("Multiple cameras are not supported")

        # names = sorted(im.name for im in self.reconstruction.images.values())
        # indices = np.arange(len(names))

        # if split == "train" and eval:
        #     names = list(np.array(names)[indices % 8 != 0])
        # elif split == "test" and eval:
        #     names = list(np.array(names)[indices % 8 == 0])
        # elif eval:
        #     raise ValueError(f"Invalid split: {split}")

        ####################################################•
        #Uniformly distribuited test samples from the dataset

        names = sorted(im.name for im in self.reconstruction.images.values())
        n_total = len(names)

        # 10%
        n_test = round(0.1 * n_total)
        step = max(1, n_total // n_test)

        #take every step‑th image for the test set, up to n_test images total
        test_indices = list(range(0, n_total, step))[:n_test]
        test_names = set(np.array(names)[test_indices])

        if eval:
            if split == "train":
                names = [n for n in names if n not in test_names]
            elif split == "test":
                names = [n for n in names if n in test_names]
            else:
                raise ValueError(f"Invalid split: {split}")

        names = list(str(name) for name in names)

        im = Image.open(os.path.join(images_dir, names[0]))
        self.img_wh = im.size
        im.close()

        # rescale all cameras to match the image resolution.  The COLMAP
        # reconstruction may contain multiple cameras (one per image), each
        # with its own intrinsic parameters.  Instead of using the first
        # camera for all images, rescale every camera once up front so that
        # their intrinsics are valid for the loaded images.
        for cam in self.reconstruction.cameras.values():
            cam.rescale(self.img_wh[0], self.img_wh[1])

        # Store a default focal length from the first camera for
        # compatibility with downstream code that expects `dataset.fx` and
        # `dataset.fy`.  Note that each image may have its own camera
        # intrinsics, so these values are representative rather than
        # authoritative.
        first_cam = list(self.reconstruction.cameras.values())[0]
        self.fx = first_cam.focal_length_x
        self.fy = first_cam.focal_length_y

        # We will compute per‑image ray directions below using the
        # appropriate camera for each image, so do not precompute
        # cam_ray_dirs here.
        cam_ray_dirs = None

        self.images = []
        for name in names:
            image = None
            for image_id in self.reconstruction.images:
                image = self.reconstruction.images[image_id]
                if image.name == name:
                    break

            if image is None:
                raise ValueError(
                    f"Image {name} not found in COLMAP reconstruction"
                )

            self.images.append(image)

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_alphas = []
        for image in tqdm(self.images):
            # Fetch the camera associated with this image.  Each image in
            # the reconstruction references a camera via its `camera_id`.
            camera = self.reconstruction.cameras[image.camera_id]
            # Generate ray directions in camera space using that camera's
            # intrinsics.  Since cameras have already been rescaled to
            # `self.img_wh`, the resulting rays correspond to the loaded
            # image resolution.
            cam_ray_dirs = get_cam_ray_dirs(camera)

            # Compute the camera‑to‑world transform.  PyCOLMAP returns a
            # 3×4 matrix for `Rigid3d.matrix()`, where the left 3×3 block
            # is the rotation and the last column is the translation.
            c2w = torch.tensor(get_world_from_cam_matrix(image), dtype=torch.float32)
            self.poses.append(c2w)

            # Rotate the ray directions into world space.  Note the use of
            # Einstein summation to perform a batch multiplication of the
            # rotation matrix with all ray directions.
            world_ray_dirs = torch.einsum(
                "ij,kj->ik",
                cam_ray_dirs,
                c2w[:, :3],
            )
            # Broadcast the camera origin to every ray.  Since c2w[:,3] is
            # a 3‑vector, broadcasting over the first dimension of
            # `cam_ray_dirs` yields an array of shape (N,3) where N is the
            # number of pixels.
            world_ray_origins = c2w[:, 3] + torch.zeros_like(cam_ray_dirs)
            # Concatenate origins and directions.  The resulting array
            # contains 6 values per pixel: (origin_x, origin_y, origin_z,
            # direction_x, direction_y, direction_z).
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            # Reshape to image height × width × 6 for downstream
            # processing.
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            im = Image.open(os.path.join(images_dir, image.name))
            if np.array(im).shape[-1] == 4:
                im = im.convert("RGBA")
                rgbas = torch.tensor(np.array(im), dtype=torch.float32) / 255.0
                alphas = rgbas[..., 3:4]
                rgbs = rgbas[..., :3] * alphas + (1 - alphas)
            else:
                im = im.convert("RGB")
                rgbs = torch.tensor(np.array(im), dtype=torch.float32) / 255.0
                alphas = torch.ones_like(rgbs[..., :1])

            im.close()

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)
            self.all_alphas.append(alphas)

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.stack(self.all_rays)
        self.all_rgbs = torch.stack(self.all_rgbs)
        self.all_alphas = torch.stack(self.all_alphas)

        self.points3D = []
        self.points3D_color = []
        for point in self.reconstruction.points3D.values():
            self.points3D.append(point.xyz)
            self.points3D_color.append(point.color)

        self.points3D = torch.tensor(
            np.array(self.points3D), dtype=torch.float32
        )
        self.points3D_color = torch.tensor(
            np.array(self.points3D_color), dtype=torch.float32
        )
        self.points3D_color = self.points3D_color / 255.0
