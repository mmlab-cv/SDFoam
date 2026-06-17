#include "pipeline_bindings.h"

#include "tracing/pipeline.h"
#include "viewer/viewer.h"

#include <vector>

namespace sdfoam_bindings {

std::vector<std::shared_ptr<Pipeline>> pipeline_registry;

void validate_scene_data(const Pipeline &pipeline,
                         torch::Tensor points,
                         torch::Tensor attributes,
                         torch::Tensor point_adjacency,
                         torch::Tensor point_adjacency_offsets) {

    if (points.size(-1) != 3) {
        throw std::runtime_error("points had dimension " +
                                 std::to_string(points.size(-1)) +
                                 " along axis -1, expected 3");
    }
    if (dtype_to_scalar_type(points.scalar_type()) != ScalarType::Float32) {
        throw std::runtime_error(
            "points had dtype " +
            std::string(c10::toString(points.scalar_type())) + ", expected " +
            std::string(scalar_to_string(ScalarType::Float32)));
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    uint32_t num_points = points.numel() / 3;

    if (attributes.size(-1) != pipeline.attribute_dim()) {
        throw std::runtime_error("attributes had dimension " +
                                 std::to_string(attributes.size(-1)) +
                                 " along axis -1, expected " +
                                 std::to_string(pipeline.attribute_dim()));
    }
    if (attributes.numel() / pipeline.attribute_dim() != num_points) {
        throw std::runtime_error("attributes must have the same number of "
                                 "rows as points");
    }
    if (dtype_to_scalar_type(attributes.scalar_type()) !=
        pipeline.attribute_type()) {
        throw std::runtime_error(
            "attributes had dtype " +
            std::string(c10::toString(attributes.scalar_type())) +
            ", expected " +
            std::string(scalar_to_string(pipeline.attribute_type())));
    }
    if (attributes.device().type() != at::kCUDA) {
        throw std::runtime_error("attributes must be on CUDA device");
    }

    if (point_adjacency_offsets.scalar_type() != at::kUInt32) {
        throw std::runtime_error(
            "point_adjacency_offsets must have uint32 dtype");
    }
    if (point_adjacency_offsets.device().type() != at::kCUDA) {
        throw std::runtime_error(
            "point_adjacency_offsets must be on CUDA device");
    }
    if (point_adjacency_offsets.numel() != num_points + 1) {
        throw std::runtime_error("point_adjacency_offsets must have num_points "
                                 "+ 1 elements");
    }

    if (point_adjacency.scalar_type() != at::kUInt32) {
        throw std::runtime_error("point_adjacency must have uint32 dtype");
    }
    if (point_adjacency.device().type() != at::kCUDA) {
        throw std::runtime_error("point_adjacency must be on CUDA device");
    }
}

void update_scene(Viewer &self,
                  torch::Tensor points_in,
                  torch::Tensor attributes_in,
                  torch::Tensor point_adjacency_in,
                  torch::Tensor point_adjacency_offsets_in,
                  torch::Tensor aabb_tree_in,
                  torch::Tensor sharpness_buffer_in) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor aabb_tree = aabb_tree_in.contiguous();
    torch::Tensor sharpness_buffer = sharpness_buffer_in.contiguous();

    validate_scene_data(self.get_pipeline(),
                        points,
                        attributes,
                        point_adjacency,
                        point_adjacency_offsets);

    set_default_stream();

    uint32_t num_points = points.size(0);
    uint32_t num_attrs = attributes.size(0);
    uint32_t num_point_adjacency = point_adjacency.size(0);
    if (sharpness_buffer.scalar_type() != at::kFloat) {
        throw std::runtime_error("sharpness_buffer must have float32 dtype");
    }
    if (sharpness_buffer.device().type() != at::kCUDA) {
        throw std::runtime_error("sharpness_buffer must be on CUDA device");
    }
    if (sharpness_buffer.numel() < num_points) {
        throw std::runtime_error(
            "sharpness_buffer must have at least num_points elements");
    }
    self.update_scene(num_points,
                      num_attrs,
                      num_point_adjacency,
                      points.data_ptr(),
                      attributes.data_ptr(),
                      point_adjacency.data_ptr(),
                      point_adjacency_offsets.data_ptr(),
                      aabb_tree.data_ptr(),
                      sharpness_buffer.data_ptr());
}

torch::Tensor inv_s_to_tensor(py::object inv_s, const torch::Tensor &like) {
    if (inv_s.is_none()) {
        return torch::ones(
            {1}, torch::dtype(torch::kFloat).device(like.device()));
    }

    torch::Tensor sharpness = inv_s.cast<torch::Tensor>().contiguous();
    if (sharpness.device() != like.device() ||
        sharpness.scalar_type() != at::kFloat) {
        sharpness = sharpness.to(like.device(), at::kFloat).contiguous();
    }
    return sharpness;
}

py::object trace_forward(Pipeline &self,
                         torch::Tensor points_in,
                         torch::Tensor attributes_in,
                         torch::Tensor point_adjacency_in,
                         torch::Tensor point_adjacency_offsets_in,
                         torch::Tensor rays_in,
                         torch::Tensor start_point_in,
                         std::optional<torch::Tensor> depth_quantiles_in,
                         py::object weight_threshold,
                         py::object max_intersections,
                         bool return_contribution,
                          py::object alpha_mode = py::none(),
                          py::object inv_s = py::none(),
                         py::object eps = py::none(),
                         py::object sdf_channel = py::none()) {

    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();
    torch::Tensor sharpness = inv_s_to_tensor(inv_s, rays);

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    torch::Tensor depth_quantiles;
    if (return_depth) {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        if (depth_quantiles.scalar_type() != at::kFloat) {
            throw std::runtime_error("depth_quantiles must have float32 dtype");
        }
        if (depth_quantiles.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_quantiles must be on CUDA device");
        }
        if (depth_quantiles.numel() / num_depth_quantiles != num_rays) {
            throw std::runtime_error("depth_quantiles must have the same batch size as rays");
        }
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    // --- SDFOAM options (all optional, backward compatible) ---
    if (!alpha_mode.is_none()) {
        std::string mode = alpha_mode.cast<std::string>();
        for (auto &c : mode) c = char(std::tolower(c));
        if (mode == "sdfoam" || mode == "sdf" || mode == "alpha_neus") {
            settings.alpha_mode = AlphaNeuS;
        } else {
            settings.alpha_mode = AlphaStandard;
        }
    }
    if (!inv_s.is_none()) {
        settings.inv_s = sharpness.reshape({-1})[0].item<float>();
    }
    if (!eps.is_none()) {
        settings.eps = eps.cast<float>();
    }
    if (!sdf_channel.is_none()) {
        settings.sdf_channel = sdf_channel.cast<int>();
    } else if (settings.alpha_mode == AlphaNeuS) {
        settings.sdf_channel = self.attribute_dim() - 1;
        settings.attr_dim_debug   = self.attribute_dim();
        settings.num_points_debug = static_cast<int>(num_points);
    }
    // --------------------------------------------------------

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++) {
        output_shape.push_back(rays.size(i));
    }

    auto output_rgba_shape = output_shape;
    output_rgba_shape.push_back(4);
    torch::Tensor output_rgba =
        torch::empty(output_rgba_shape,
                     torch::dtype(scalar_to_type_meta(self.attribute_type()))
                         .device(rays.device()));

    auto output_num_intersections_shape = output_shape;
    output_num_intersections_shape.push_back(1);
    torch::Tensor num_intersections =
        torch::empty(output_num_intersections_shape,
                     torch::dtype(scalar_to_type_meta(ScalarType::UInt32))
                         .device(rays.device()));

    torch::Tensor output_contribution;
    if (return_contribution) {
        output_contribution = torch::zeros(
            {num_points, 1},
            torch::dtype(scalar_to_type_meta(self.attribute_type()))
                .device(rays.device()));
    }

    auto output_depth_shape = output_shape;
    output_depth_shape.push_back(num_depth_quantiles);
    torch::Tensor output_depth;
    torch::Tensor output_depth_indices;
    if (return_depth) {
        output_depth =
            torch::zeros(output_depth_shape,
                         torch::dtype(scalar_to_type_meta(ScalarType::Float32))
                             .device(rays.device()));
        output_depth_indices =
            torch::zeros(output_depth_shape,
                         torch::dtype(scalar_to_type_meta(ScalarType::UInt32))
                             .device(rays.device()));
    }

    torch::Tensor alpha_output = torch::zeros(
        {num_points, 1},
        torch::dtype(scalar_to_type_meta(ScalarType::Float32))
            .device(rays.device()));

    set_default_stream();

    self.trace_forward(
        settings,
        num_points,
        reinterpret_cast<const sdfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const sdfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        num_depth_quantiles,
        return_depth
            ? reinterpret_cast<const float *>(depth_quantiles.data_ptr())
            : nullptr,
        output_rgba.data_ptr(),
        return_depth ? reinterpret_cast<float *>(output_depth.data_ptr())
                     : nullptr,
        return_depth
            ? reinterpret_cast<uint32_t *>(output_depth_indices.data_ptr())
            : nullptr,
        reinterpret_cast<uint32_t *>(num_intersections.data_ptr()),
        return_contribution ? output_contribution.data_ptr() : nullptr,
        reinterpret_cast<float *>(sharpness.data_ptr()),
        reinterpret_cast<float *>(alpha_output.data_ptr())
    );

    py::dict output_dict;
    output_dict["rgba"] = output_rgba;
    if (return_depth) {
        output_dict["depth"] = output_depth;
        output_dict["depth_indices"] = output_depth_indices;
    }
    if (return_contribution) {
        output_dict["contribution"] = output_contribution;
    }
    output_dict["num_intersections"] = num_intersections;
    output_dict["alpha"] = alpha_output;

    return output_dict;
}

py::object trace_backward(Pipeline &self,
                          torch::Tensor points_in,
                          torch::Tensor attributes_in,
                          torch::Tensor point_adjacency_in,
                          torch::Tensor point_adjacency_offsets_in,
                          torch::Tensor rays_in,
                          torch::Tensor start_point_in,
                          torch::Tensor rgb_out,
                          torch::Tensor rgb_grad_in,
                          std::optional<torch::Tensor> depth_quantiles_in,
                          std::optional<torch::Tensor> depth_indices_in,
                          std::optional<torch::Tensor> depth_grad_in,
                          std::optional<torch::Tensor> ray_error_in,
                          py::object weight_threshold,
                          py::object max_intersections,
                          py::object alpha_mode = py::none(),
                          py::object inv_s = py::none(),
                          py::object eps = py::none(),
                          py::object sdf_channel = py::none()) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();
    torch::Tensor sharpness = inv_s_to_tensor(inv_s, rays);

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    bool return_depth = depth_quantiles_in.has_value();
    bool return_error = ray_error_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;
    uint32_t num_depth_quantiles = 0;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size "
                                 "as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    torch::Tensor rgb_grad_in_c = rgb_grad_in.contiguous();
    if (rgb_grad_in_c.size(-1) != 4) {
        throw std::runtime_error("rgb_grad_in must have 4 as "
                                 "the last dimension");
    }
    if (dtype_to_scalar_type(rgb_grad_in_c.scalar_type()) !=
        self.attribute_type()) {
        throw std::runtime_error(
            "rgb_grad_in had dtype " +
            std::string(c10::toString(rgb_grad_in_c.scalar_type())) +
            ", expected " +
            std::string(scalar_to_string(self.attribute_type())));
    }
    if (rgb_grad_in_c.device().type() != at::kCUDA) {
        throw std::runtime_error("rgb_grad_in must be on CUDA device");
    }
    if (rgb_grad_in_c.numel() / 4 != num_rays) {
        throw std::runtime_error("rgb_grad_in must have the same batch size "
                                 "as rays");
    }

    torch::Tensor depth_quantiles;
    torch::Tensor depth_indices;
    torch::Tensor depth_grad;
    if (return_depth) {
        depth_quantiles = depth_quantiles_in.value().contiguous();
        num_depth_quantiles = depth_quantiles.size(-1);

        if (depth_quantiles.scalar_type() != at::kFloat) {
            throw std::runtime_error("depth_quantiles must have float32 dtype");
        }
        if (depth_quantiles.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_quantiles must be on CUDA device");
        }
        if (depth_quantiles.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_quantiles must have the same batch "
                                     "size as rays");
        }

        if (!depth_grad_in.has_value()) {
            throw std::runtime_error("depth_grad must be provided if "
                                     "depth_quantiles is provided");
        }

        depth_indices = depth_indices_in.value().contiguous();

        if (depth_indices.scalar_type() != at::kUInt32) {
            throw std::runtime_error("depth_indices must have uint32 dtype");
        }
        if (depth_indices.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_indices must be on CUDA device");
        }
        if (depth_indices.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_indices must have the same batch "
                                     "size as rays");
        }

        depth_grad = depth_grad_in.value().contiguous();

        if (depth_grad.size(-1) != num_depth_quantiles) {
            throw std::runtime_error("depth_grad must have the same number of "
                                     "depth quantiles as depth_quantiles");
        }
        if (dtype_to_scalar_type(depth_grad.scalar_type()) !=
            ScalarType::Float32) {
            throw std::runtime_error(
                "depth_grad had dtype " +
                std::string(c10::toString(depth_grad.scalar_type())) +
                ", expected " +
                std::string(scalar_to_string(ScalarType::Float32)));
        }
        if (depth_grad.device().type() != at::kCUDA) {
            throw std::runtime_error("depth_grad must be on CUDA device");
        }
        if (depth_grad.numel() != num_rays * num_depth_quantiles) {
            throw std::runtime_error("depth_grad must have the same batch "
                                     "size as rays");
        }
    }

    torch::Tensor ray_error;
    torch::Tensor point_error;
    if (return_error) {
        ray_error = ray_error_in.value().contiguous();

        if (dtype_to_scalar_type(ray_error.scalar_type()) !=
            self.attribute_type()) {
            throw std::runtime_error(
                "ray_error had dtype " +
                std::string(c10::toString(ray_error.scalar_type())) +
                ", expected " +
                std::string(scalar_to_string(self.attribute_type())));
        }
        if (ray_error.device().type() != at::kCUDA) {
            throw std::runtime_error("ray_error must be on CUDA device");
        }
        if (ray_error.numel() != num_rays) {
            std::cout << ray_error.numel() << " " << num_rays << std::endl;
            throw std::runtime_error("ray_error must have the same batch size "
                                     "as rays");
        }

        point_error = torch::zeros(
            {num_points, 1},
            torch::dtype(scalar_to_type_meta(self.attribute_type()))
                .device(rays.device()));
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    // --- SDFOAM options (all optional, backward compatible) ---
    if (!alpha_mode.is_none()) {
        std::string mode = alpha_mode.cast<std::string>();
        // accept a few common spellings
        for (auto &c : mode) c = char(std::tolower(c));
        if (mode == "sdfoam" || mode == "sdf" || mode == "alpha_neus") {
            settings.alpha_mode = AlphaNeuS;
        } else {
            settings.alpha_mode = AlphaStandard;
        }
    }
    // if (!inv_s.is_none()) {
    if (!inv_s.is_none()) {
        settings.inv_s = sharpness.reshape({-1})[0].item<float>();
    }
    if (!eps.is_none()) {
        settings.eps = eps.cast<float>();
    }
    // If user passes a channel, use it; otherwise, if SDFOAM is on, default to last slot
    if (!sdf_channel.is_none()) {
        settings.sdf_channel = sdf_channel.cast<int>();
    } else if (settings.alpha_mode == AlphaNeuS) {
        settings.sdf_channel = self.attribute_dim() - 1; // AoS: last scalar
        settings.attr_dim_debug   = self.attribute_dim();
        settings.num_points_debug = static_cast<int>(num_points);
    }


    int64_t num_attr = attributes.size(0);

    std::vector<int64_t> attr_grad_shape = {num_attr, self.attribute_dim()};

    torch::Tensor attr_grad =
        torch::zeros(attr_grad_shape,
                     torch::dtype(scalar_to_type_meta(self.attribute_type()))
                         .device(rays.device()));

    std::vector<int64_t> points_grad_shape = {(int64_t)num_points, 3};

    torch::Tensor points_grad = torch::zeros(
        points_grad_shape, torch::dtype(rays.dtype()).device(rays.device()));

    torch::Tensor ray_grad = torch::empty_like(rays);

    // grad buffer must be float32 on CUDA too
    torch::Tensor sharpness_grad = torch::zeros_like(
        sharpness, sharpness.options().dtype(at::kFloat));

    set_default_stream();

    self.trace_backward(
        settings,
        num_points,
        reinterpret_cast<const sdfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const sdfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        num_depth_quantiles,
        return_depth
            ? reinterpret_cast<const float *>(depth_quantiles.data_ptr())
            : nullptr,
        return_depth
            ? reinterpret_cast<const uint32_t *>(depth_indices.data_ptr())
            : nullptr,
        rgb_out.data_ptr(),
        rgb_grad_in_c.data_ptr(),
        return_depth ? reinterpret_cast<const float *>(depth_grad.data_ptr())
                     : nullptr,
        return_error ? ray_error.data_ptr() : nullptr,
        reinterpret_cast<sdfoam::Ray *>(ray_grad.data_ptr()),
        reinterpret_cast<sdfoam::Vec3f *>(points_grad.data_ptr()),
        attr_grad.data_ptr(),
        return_error ? point_error.data_ptr() : nullptr,
        // sharpness.data_ptr(),
        reinterpret_cast<float *>(sharpness.data_ptr()),
        reinterpret_cast<float *>(sharpness_grad.data_ptr()));

    py::dict output_dict;

    output_dict["points_grad"] = points_grad;
    output_dict["attr_grad"] = attr_grad;
    output_dict["ray_grad"] = ray_grad;
    output_dict["sharpness_grad"] = sharpness_grad;
    if (return_error) {
        output_dict["point_error"] = point_error;
    }

    return output_dict;
}

void trace_benchmark(Pipeline &self,
                     torch::Tensor points_in,
                     torch::Tensor attributes_in,
                     torch::Tensor point_adjacency_in,
                     torch::Tensor point_adjacency_offsets_in,
                     torch::Tensor adjacent_diff_in,
                     py::dict camera_in,
                     torch::Tensor start_point,
                     torch::Tensor output_rgba_in,
                     py::object weight_threshold,
                     py::object max_intersections,
                      py::object alpha_mode = py::none(),
                      py::object inv_s = py::none(),
                     py::object eps = py::none(),
                     py::object sdf_channel = py::none()) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor adjacent_diff = adjacent_diff_in.contiguous();
    torch::Tensor sharpness = inv_s_to_tensor(inv_s, output_rgba_in);

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    uint32_t num_points = points.size(0);

    sdfoam::Camera camera;
    camera.position = sdfoam::Vec3f(
        camera_in["position"].cast<torch::Tensor>().data_ptr<float>());
    camera.forward = sdfoam::Vec3f(
        camera_in["forward"].cast<torch::Tensor>().data_ptr<float>());
    camera.up =
        sdfoam::Vec3f(camera_in["up"].cast<torch::Tensor>().data_ptr<float>());
    camera.right = sdfoam::Vec3f(
        camera_in["right"].cast<torch::Tensor>().data_ptr<float>());
    camera.fov = camera_in["fov"].cast<float>();
    camera.width = camera_in["width"].cast<int>();
    camera.height = camera_in["height"].cast<int>();
    if (camera_in["model"].cast<std::string>() == "pinhole") {
        camera.model = sdfoam::CameraModel::Pinhole;
    } else if (camera_in["model"].cast<std::string>() == "fisheye") {
        camera.model = sdfoam::CameraModel::Fisheye;
    } else {
        throw std::runtime_error("Invalid camera model");
    }

    if (start_point.numel() != 1) {
        throw std::runtime_error("start_point must have a single element");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    if (output_rgba_in.numel() != camera.width * camera.height) {
        throw std::runtime_error("output_rgba must have width * height "
                                 "elements");
    }
    if (output_rgba_in.scalar_type() != at::kUInt32) {
        throw std::runtime_error("output_rgba must have uint32 dtype");
    }
    if (output_rgba_in.device().type() != at::kCUDA) {
        throw std::runtime_error("output_rgba must be on CUDA device");
    }

    TraceSettings settings = default_trace_settings();
    if (!weight_threshold.is_none()) {
        settings.weight_threshold = weight_threshold.cast<float>();
    }
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    // --- SDFOAM options (all optional, backward compatible) ---
    if (!alpha_mode.is_none()) {
        std::string mode = alpha_mode.cast<std::string>();
    
        for (auto &c : mode) c = char(std::tolower(c));
        if (mode == "sdfoam" || mode == "sdf" || mode == "alpha_neus") {
            settings.alpha_mode = AlphaNeuS;
        } else {
            settings.alpha_mode = AlphaStandard;
        }
    }
    if (!inv_s.is_none()) {
        settings.inv_s = sharpness.reshape({-1})[0].item<float>();
    }
    if (!eps.is_none()) {
        settings.eps = eps.cast<float>();
    }

    if (!sdf_channel.is_none()) {
        settings.sdf_channel = sdf_channel.cast<int>();
    } else if (settings.alpha_mode == AlphaNeuS) {
        settings.sdf_channel = self.attribute_dim() - 1; // AoS: last scalar
        settings.attr_dim_debug   = self.attribute_dim();
        settings.num_points_debug = static_cast<int>(num_points);

    }
    // --------------------------------------------------------

    self.trace_benchmark(
        settings,
        num_points,
        reinterpret_cast<const sdfoam::Vec3f *>(points.data_ptr()),
        attributes.data_ptr(),
        static_cast<uint32_t>(point_adjacency.numel()),
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        reinterpret_cast<const sdfoam::Vec4h *>(adjacent_diff.data_ptr()),
        default_visualization_settings(),
        camera,
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        reinterpret_cast<uint32_t *>(output_rgba_in.data_ptr()),
        // sharpness.data_ptr(),
        reinterpret_cast<float *>(sharpness.data_ptr()));
}

ScalarType dtype_string_to_scalar_type(const std::string &dtype) {
    if (dtype == "float32" || dtype == "torch.float32") {
        return ScalarType::Float32;
    }
    if (dtype == "float64" || dtype == "torch.float64") {
        return ScalarType::Float64;
    }
    if (dtype == "float16" || dtype == "torch.float16") {
        return ScalarType::Float16;
    }
    throw std::runtime_error("unsupported dtype '" + dtype + "'");
}

Pipeline *create_pipeline_binding(int sh_degree, const std::string &attr_dtype) {
    ScalarType scalar_type = dtype_string_to_scalar_type(attr_dtype);
    pipeline_registry.push_back(sdfoam::create_pipeline(sh_degree, scalar_type));
    return pipeline_registry.back().get();
}

void run_with_viewer_binding(Pipeline &pipeline,
                     std::function<void(std::shared_ptr<Viewer>)> callback,
                     std::optional<int> total_iterations,
                     std::optional<torch::Tensor> camera_pos,
                     std::optional<torch::Tensor> camera_forward,
                     std::optional<torch::Tensor> camera_up,
                     std::optional<bool> use_sdfoam,
                     std::optional<float> inv_s,
                     std::optional<float> eps,
                     std::optional<int> sdf_channel) {
    py::gil_scoped_release release;

    ViewerOptions options = default_viewer_options();
    if (total_iterations.has_value()) {
        options.total_iterations = total_iterations.value();
    }
    if (camera_pos.has_value()) {
        torch::Tensor camera_pos_cpu =
            camera_pos->contiguous().cpu().to(torch::kFloat);
        options.camera_pos = sdfoam::Vec3f(camera_pos_cpu.data_ptr<float>());
    }
    if (camera_forward.has_value()) {
        torch::Tensor camera_forward_cpu =
            camera_forward->contiguous().cpu().to(torch::kFloat);
        options.camera_forward =
            sdfoam::Vec3f(camera_forward_cpu.data_ptr<float>());
    }
    if (camera_up.has_value()) {
        torch::Tensor camera_up_cpu =
            camera_up->contiguous().cpu().to(torch::kFloat);
        options.camera_up = sdfoam::Vec3f(camera_up_cpu.data_ptr<float>());
    }

    // NEW:
    if (use_sdfoam)   options.use_sdfoam   = *use_sdfoam;
    if (inv_s)      options.inv_s      = *inv_s;
    if (eps)        options.eps        = *eps;
    if (sdf_channel)options.sdf_channel= *sdf_channel;

    set_default_stream();

    std::shared_ptr<Pipeline> pipeline_ref(
        &pipeline, [](Pipeline *) {});
    run_with_viewer(std::move(pipeline_ref), std::move(callback), options);
}

void init_pipeline_bindings(py::module &module) {
    py::class_<Pipeline>(module, "Pipeline")
        .def("trace_forward",
             trace_forward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("depth_quantiles") = py::none(),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none(),
             py::arg("return_contribution") = false,
             py::arg("alpha_mode") = py::none(),   // NEW
             py::arg("inv_s") = py::none(),        // NEW
             py::arg("eps") = py::none(),          // NEW
             py::arg("sdf_channel") = py::none())  // NEW

        .def("trace_backward",
             trace_backward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("rgb_out"),
             py::arg("grad_in"),
             py::arg("depth_quantiles") = py::none(),
             py::arg("depth_indices") = py::none(),
             py::arg("depth_grad_in") = py::none(),
             py::arg("ray_error") = py::none(),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none(),
             py::arg("alpha_mode") = py::none(),   // NEW
             py::arg("inv_s") = py::none(),        // NEW
             py::arg("eps") = py::none(),          // NEW
             py::arg("sdf_channel") = py::none())  // NEW

        .def("trace_benchmark",
             trace_benchmark,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("adjacent_diff"),
             py::arg("camera"),
             py::arg("start_point"),
             py::arg("output_rgba"),
             py::arg("weight_threshold") = py::none(),
             py::arg("max_intersections") = py::none(),
             py::arg("alpha_mode") = py::none(),   // NEW
             py::arg("inv_s") = py::none(),        // NEW
             py::arg("eps") = py::none(),          // NEW
             py::arg("sdf_channel") = py::none()); // NEW

    module.def("create_pipeline",
               create_pipeline_binding,
               py::arg("sh_degree"),
               py::arg("attr_dtype") = "float32",
               py::return_value_policy::reference);

    py::class_<Viewer, std::shared_ptr<Viewer>>(module, "Viewer")
        .def("update_scene",
             update_scene,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("aabb_tree"),
             py::arg("sharpness_buffer"))
        .def("step", &Viewer::step)
        .def("is_closed", &Viewer::is_closed);

    module.def("run_with_viewer",
               run_with_viewer_binding,
               py::arg("pipeline"),
               py::arg("callback"),
               py::arg("total_iterations") = py::none(),
               py::arg("camera_pos") = py::none(),
               py::arg("camera_forward") = py::none(),
               py::arg("camera_up") = py::none(),
               py::arg("use_sdfoam") = py::none(),
               py::arg("inv_s") = py::none(),
               py::arg("eps") = py::none(),
               py::arg("sdf_channel") = py::none());
}

} // namespace sdfoam_bindings
