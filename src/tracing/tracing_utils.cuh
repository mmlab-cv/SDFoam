#pragma once

#include <cuda_fp16.h>

#include "../utils/geometry.h"
#include "camera.h"

// Add CUDART_INF_F definition if not already defined
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

namespace sdfoam {

template <int block_size, int chunk_size, typename CellFunctor>
__forceinline__ __device__ uint32_t
trace(const Ray &ray,
      const Vec3f *__restrict__ points,
      const uint32_t *__restrict__ point_adjacency,
      const uint32_t *__restrict__ point_adjacency_offsets,
      const Vec4h *__restrict__ adjacent_points, // encoded neighbor diffs
      uint32_t start_point,
      uint32_t max_steps,
      CellFunctor cell_functor,
      bool use_safe_mode,
      uint32_t num_points = UINT32_MAX,
      uint32_t point_adjacency_size = UINT32_MAX)
{
    float t_0 = 0.0f;
    uint32_t n = 0;

    uint32_t current_point_idx = start_point;
    if (current_point_idx >= num_points) {
        return n;
    }
    Vec3f primal_point = points[current_point_idx];

    for (;;) {
        ++n;
        if (n > max_steps) break;
        if (current_point_idx >= num_points) break;

        const uint32_t beg = point_adjacency_offsets[current_point_idx];
        const uint32_t end = point_adjacency_offsets[current_point_idx + 1];
        if (beg > end || end > point_adjacency_size) break;
        const uint32_t num_faces = end - beg;

        float    t_1 = CUDART_INF_F;
        uint32_t next_face = UINT32_MAX;

        //chunk
        for (uint32_t i = 0; i < num_faces; i += chunk_size) {
            const uint32_t todo = use_safe_mode ? min(chunk_size, num_faces - i) : chunk_size;
            half2 chunk[chunk_size * 2];

#pragma unroll
            for (uint32_t j = 0; j < todo; ++j) {
                const half2* h2 = reinterpret_cast<const half2*>(
                    adjacent_points + (beg + i + j));
                chunk[2*j + 0] = h2[0];
                chunk[2*j + 1] = h2[1];
            }

#pragma unroll
            for (uint32_t j = 0; j < todo; ++j) {
                const half2 a = chunk[2*j + 0];
                const half2 b = chunk[2*j + 1];
                Vec3f offset(__half2float(a.x),
                             __half2float(a.y),
                             __half2float(b.x));

                Vec3f face_origin = primal_point + offset * 0.5f;
                Vec3f face_normal = offset;
                float dp = face_normal.dot(ray.direction);
                float t = (face_origin - ray.origin).dot(face_normal) / dp;

                if (use_safe_mode) {
                    if (dp > 0.0f && t > t_0 && t < t_1) {
                        t_1 = t;
                        next_face = i + j;
                    }
                } else {
                    if (dp > 0.0f && t < t_1 && (i + j) < num_faces) {
                        t_1 = t;
                        next_face = i + j;
                    }
                }
            }
        }

        if (next_face == UINT32_MAX) break;

        const uint32_t next_point_idx = point_adjacency[beg + next_face];
        if (next_point_idx >= num_points) break;
        const Vec3f    next_point     = points[next_point_idx];

        if (t_1 > t_0) {
            const bool keep_going = cell_functor(
                current_point_idx, next_point_idx, t_0, t_1, primal_point, next_point);
            if (!keep_going) break;
        }

        t_0 = fmaxf(t_0, t_1);
        current_point_idx = next_point_idx;
        primal_point      = next_point;
    }

    return n;
}


__forceinline__ __device__ Vec3f cell_intersection_grad(
    const Vec3f &primal_point, const Vec3f &opposite_point, const Ray &ray) {
    Vec3f face_origin = (primal_point + opposite_point) / 2.0f;
    Vec3f face_normal = (opposite_point - primal_point);

    float num = (face_origin - ray.origin).dot(face_normal);
    float dp = face_normal.dot(ray.direction);

    Vec3f grad = num * ray.direction + dp * (ray.origin - primal_point);
    grad /= dp * dp;

    return grad;
}

inline SDFOAM_HD uint32_t make_rgba8(float r, float g, float b, float a) {
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
    a = std::max(0.0f, std::min(1.0f, a));
    int ri = static_cast<int>(r * 255.0f);
    int gi = static_cast<int>(g * 255.0f);
    int bi = static_cast<int>(b * 255.0f);
    int ai = static_cast<int>(a * 255.0f);
    return (ai << 24) | (bi << 16) | (gi << 8) | ri;
}

inline __device__ Vec3f colormap(float v,
                                 ColorMap map,
                                 const CMapTable &cmap_table) {
    int map_len = cmap_table.sizes[map];
    const Vec3f *map_vals =
        reinterpret_cast<const Vec3f *>(cmap_table.data[map]);

    int i0 = static_cast<int>(v * (map_len - 1));
    int i1 = i0 + 1;
    float t = v * (map_len - 1) - i0;
    i0 = max(0, min(i0, map_len - 1));
    i1 = max(0, min(i1, map_len - 1));
    return map_vals[i0] * (1.0f - t) + map_vals[i1] * t;
}

} // namespace sdfoam

