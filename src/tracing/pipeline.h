#pragma once

#include <memory>

#include "../utils/typing.h"
#include "camera.h"

namespace sdfoam {

enum AlphaMode : uint32_t { AlphaStandard = 0, AlphaNeuS = 1 };

struct TraceSettings {
    float    weight_threshold;
    uint32_t max_intersections;

    // --- NEW: NeuS controls ---
    AlphaMode alpha_mode = AlphaStandard;
    float     inv_s      = 50.0f;   // 1/beta; raise during training
    float     eps        = 1e-6f;   // numerical guard
    int       sdf_channel = -1;     // if >= 0, use attributes[..., sdf_channel] as SDF
    int attr_dim_debug = 0;     // number of scalars per-point in `attributes`
    int num_points_debug = 0;   // total number of points
    float max_sdf = 2.0f;         // scale/threshold for |SDF| band mode
    float sdf_value = 0.0f;       // target value for "exact" mode
    float sdf_tolerance = 0.10f;   // >= 0, acceptance margin for "exact" mode
    bool  sdf_exact_mode = false;  // false: Max |SDF|, true: Exact SDF ± tol
};

inline TraceSettings default_trace_settings() {
    TraceSettings settings;
    settings.weight_threshold = 1e-7f;
    settings.max_intersections = 1024;
    settings.attr_dim_debug = 0;
    settings.num_points_debug = 0;
    settings.max_sdf = 2.0f;
    settings.sdf_value = 0.0f;
    settings.sdf_tolerance = 0.10f;
    settings.sdf_exact_mode = false;
    return settings;
}

enum VisualizationMode {
    RGB = 0,
    Depth = 1,
    Alpha = 2,
    Intersections = 3,
    SDF = 4
};

struct VisualizationSettings {
    VisualizationMode mode;
    ColorMap color_map;
    CVec3f bg_color;
    bool checker_bg;
    float max_depth;
    float depth_quantile;
};

inline VisualizationSettings default_visualization_settings() {
    VisualizationSettings settings;
    settings.mode = RGB;
    settings.color_map = Turbo;
    settings.bg_color = Vec3f(1.0f, 1.0f, 1.0f);
    settings.checker_bg = false;
    settings.max_depth = 10.0f;
    settings.depth_quantile = 0.5f;
    return settings;
}

/// @brief Prefetch offset for each edge in the adjacency matrix
void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream);

class Pipeline {
  public:
    virtual ~Pipeline() = default;

    virtual void trace_forward(const TraceSettings &settings,
                               uint32_t num_points,
                               const Vec3f *points,
                               const void *attributes,
                               uint32_t point_adjacency_size,
                               const uint32_t *point_adjacency,
                               const uint32_t *point_adjacency_offsets,
                               uint32_t num_rays,
                               const Ray *rays,
                               const uint32_t *start_point_index,
                               uint32_t num_depth_quantiles,
                               const float *depth_quantiles,
                               void *ray_rgba,
                               float *quantile_depths,
                               uint32_t *quantile_point_indices,
                               uint32_t *num_intersections,
                               void *point_contribution,
                               void *sharpness,
                               void *alpha_output) = 0;


    virtual void trace_backward(const TraceSettings &settings,
                                uint32_t num_points,
                                const Vec3f *points,
                                const void *attributes,
                                uint32_t point_adjacency_size,
                                const uint32_t *point_adjacency,
                                const uint32_t *point_adjacency_offsets,
                                uint32_t num_rays,
                                const Ray *rays,
                                const uint32_t *start_point_index,
                                uint32_t num_depth_quantiles,
                                const float *depth_quantiles,
                                const uint32_t *quantile_point_indices,
                                const void *ray_rgba,
                                const void *ray_rgba_grad,
                                const float *depth_grad,
                                const void *ray_error,
                                Ray *ray_grad,
                                Vec3f *points_grad,
                                void *attribute_grad,
                                void *point_error,
                                void *sharpness,
                                float *sharpness_grad) = 0;

    virtual void trace_visualization(const TraceSettings &settings,
                                     const VisualizationSettings &vis_settings,
                                     const Camera &camera,
                                     CMapTable cmap_table,
                                     uint32_t num_points,
                                     uint32_t num_tets,
                                     const void *points,
                                     const void *attributes,
                                     const void *point_adjacency,
                                     const void *point_adjacency_offsets,
                                     const void *adjacent_points,
                                     uint32_t start_index,
                                     uint64_t output_surface,
                                     const void *stream = nullptr,
                                     void *sharpness = nullptr) = 0;

    virtual void trace_benchmark(const TraceSettings &settings,
                                 uint32_t num_points,
                                 const Vec3f *points,
                                 const void *attributes,
                                 uint32_t point_adjacency_size,
                                 const uint32_t *point_adjacency,
                                 const uint32_t *point_adjacency_offsets,
                                 const Vec4h *adjacent_diff,
                                 const VisualizationSettings &vis_settings,
                                 Camera camera,
                                 const uint32_t *start_point_index,
                                 uint32_t *ray_rgba,
                                 void *sharpness) = 0;

    virtual uint32_t attribute_dim() const = 0;

    virtual ScalarType attribute_type() const = 0;
};

std::shared_ptr<Pipeline> create_pipeline(int sh_degree, ScalarType attr_type);

} // namespace sdfoam
