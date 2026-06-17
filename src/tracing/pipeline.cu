#include "../aabb_tree/aabb_tree.h"
#include "../delaunay/triangulation_ops.h"
#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "pipeline.h"

#include "../utils/common_kernels.cuh"
#include "sh_utils.cuh"
#include "tracing_utils.cuh"

#include <cuda_fp16.h>
#include <type_traits>

namespace sdfoam {

// === helpers ===
__device__ inline float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

template <typename T>
__device__ inline float read_attr_channel(const T* attrs, int dim, int idx, int ch);

// AoS layout: [p0: D scalars][p1: D scalars]...
template <>
__device__ inline float read_attr_channel<float>(const float* attrs, int dim, int idx, int ch) {
    return attrs[idx * dim + ch];
}

template <>
__device__ inline float read_attr_channel<__half>(const __half* attrs, int dim, int idx, int ch) {
    return __half2float(attrs[idx * dim + ch]);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void forward(TraceSettings settings,
                        const Vec3f *__restrict__ points,
                        const attr_scalar *__restrict__ attributes,
                        const uint32_t *__restrict__ point_adjacency,
                        const uint32_t *__restrict__ point_adjacency_offsets,
                        const Vec4h *__restrict__ adjacent_diff,
                        const Ray *__restrict__ rays,
                        uint32_t num_rays,
                        const uint32_t *__restrict__ start_point_index,
                        uint32_t num_depth_quantiles,
                        const float *__restrict__ depth_quantiles,
                        attr_scalar *__restrict__ ray_rgba,
                        float *__restrict__ quantile_depths,
                        uint32_t *__restrict__ quantile_point_indices,
                        uint32_t *__restrict__ num_intersections,
                        attr_scalar *__restrict__ point_contribution,
                        const attr_scalar *__restrict__ sharpness,
                        float *__restrict__ alpha_output) {
    // ---- tiny helpers (device-local) ----
    auto sigmoidf = [] __device__ (float x) {
        return 1.f / (1.f + __expf(-x));
    };

    auto read_attr_channel = [&] __device__ (const attr_scalar* base, int dim, int idx, int ch) -> float {
        if constexpr (std::is_same<attr_scalar, __half>::value) {
            const __half* h = reinterpret_cast<const __half*>(base);
            return __half2float(h[idx * dim + ch]);
        } else {
            return static_cast<float>(base[idx * dim + ch]);
        }
    };

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim; // [ SH ... | last_scalar ]

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_quantiles =
        depth_quantiles ? (depth_quantiles + thread_idx * num_depth_quantiles) : nullptr;

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    // NOTE: In legacy (density) mode, last_scalar = sigma (density).
    //       In sdfoam mode, last_scalar should carry SDF if you set settings.sdf_channel accordingly.
    auto load_attributes = [&] __device__ (uint32_t v_idx, Vec3f &rgb, float &last_scalar) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        last_scalar = static_cast<float>(attr_ptr[attr_memory_size - 1]);

        // For sdfoam we DO NOT gate color on last_scalar (it is SDF there).
        // For legacy density mode, keep your original gating.
        if (settings.alpha_mode == AlphaNeuS) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            if (last_scalar > 1e-6f) {
                rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
            } else {
                rgb = Vec3f::Zero();
            }
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    // --- SDF first-hit debug state ---
    float sdf_at_first_hit = 0.0f;
    float sdf_s0 = 0.0f, sdf_s1 = 0.0f;
    bool  has_first_hit = false;

    uint32_t current_quantile_idx = 0;
    float current_quantile = 0.f;
    if (ray_depth_quantiles && num_depth_quantiles > 0) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }

    // IMPORTANT: this functor now takes next_point_idx (2nd arg).
    auto functor = [&] __device__ (uint32_t point_idx,
                                   uint32_t next_point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal; // sigma (legacy) OR SDF (if you store SDF in last channel)
        load_attributes(point_idx, rgb_primal, s_primal);

        float alpha = 0.f;

        if (settings.alpha_mode == AlphaNeuS && settings.sdf_channel >= 0) {
            const int D = attr_memory_size;
            const int k = (settings.sdf_channel >= 0) ? settings.sdf_channel : (D - 1);
            // sdfoam alpha needs SDF at both ends of the segment
            float d_i = read_attr_channel(attributes, D, point_idx, k);
            // float d_ip1 = read_attr_channel(attributes, D, next_point_idx, k);

            float sha;
            if constexpr (std::is_same<attr_scalar, __half>::value) {
                sha = __half2float(*reinterpret_cast<const __half*>(sharpness));
            } else {
                sha = static_cast<float>(*sharpness);
            }

            // float phi_i   = sigmoidf(-sha * d_i);
            // float phi_ip1 = sigmoidf(-sha * d_ip1);
            // float numer = fmaxf(0.f, phi_i - phi_ip1);
            // float denom = fmaxf(settings.eps, phi_i);
            // alpha = fminf(1.f, numer / denom);

            s_primal = sha * sigmoidf(-sha * d_i) * (1.f - sigmoidf(-sha * d_i));
            float delta_t = fmaxf(t_1 - t_0, 0.0f);
            alpha = 1.f - __expf(-s_primal * delta_t);

        } else {
            float delta_t = fmaxf(t_1 - t_0, 0.0f);
            alpha = 1.f - __expf(-s_primal * delta_t);
        }

        float weight = transmittance * alpha;
        alpha_output[point_idx] = alpha;

        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, (attr_scalar)weight);
        }
        accumulated_rgb += weight * rgb_primal;

        float next_transmittance = transmittance * (1.f - alpha);

        if (!has_first_hit) {
            const int i0 = point_idx;
            const int i1 = next_point_idx;

            // attributes layout: [ SH coeffs (sh_dim) | last scalar ]
            constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
            constexpr int attr_memory_size = 1 + sh_dim;

            const attr_scalar* a0 = attributes + i0 * attr_memory_size;
            const attr_scalar* a1 = attributes + i1 * attr_memory_size;

            // last scalar is SDF in sdfoam mode
            sdf_s0 = (float)a0[attr_memory_size - 1];
            sdf_s1 = (float)a1[attr_memory_size - 1];

            // simple midpoint (use your face weight if you already compute one)
            sdf_at_first_hit = 0.5f * (sdf_s0 + sdf_s1);

            has_first_hit = true;
        }

        if (settings.alpha_mode == AlphaNeuS) {
            // Robust sdfoam crossing: linearize T on [t0,t1]
            const float drop = transmittance - next_transmittance;  // = T0 * alpha
            // If alpha ~ 0 or T didn't change, no crossing can occur inside this segment.
            if (drop > 1e-12f) {
                // Multiple quantiles can fall in the same segment; emit all of them
                while (current_quantile_idx < num_depth_quantiles &&
                       next_transmittance < current_quantile) {

                    // frac = (T0 - q) / (T0*alpha) in [0,1]
                    float frac = (transmittance - current_quantile) / drop;
                    // Clamp for numerical safety
                    frac = fminf(1.f, fmaxf(0.f, frac));

                    // tq = t0 + frac * (t1 - t0)
                    float tq = fmaf(frac, (t_1 - t_0), t_0);

                    const uint32_t out_ofs = thread_idx * num_depth_quantiles + current_quantile_idx;
                    quantile_depths[out_ofs] = tq;
                    quantile_point_indices[out_ofs] = point_idx;

                    // Next requested quantile for this ray
                    ++current_quantile_idx;
                    if (current_quantile_idx < num_depth_quantiles) {
                        current_quantile = ray_depth_quantiles[current_quantile_idx];
                    }
                }
            }
        } else {
            while (current_quantile_idx < num_depth_quantiles &&
                   next_transmittance < current_quantile) {
                quantile_depths[thread_idx * num_depth_quantiles +
                                current_quantile_idx] =
                    t_0 + logf(transmittance / current_quantile) / s_primal;
                quantile_point_indices[thread_idx * num_depth_quantiles +
                                       current_quantile_idx] = point_idx;
                current_quantile_idx++;
                if (current_quantile_idx < num_depth_quantiles) {
                    current_quantile = ray_depth_quantiles[current_quantile_idx];
                }
            }
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    uint32_t start_point = start_point_index[thread_idx];

    bool use_safe_mode = true;

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point,
                                      settings.max_intersections,
                                      functor,
                                      use_safe_mode);

    while (current_quantile_idx < num_depth_quantiles) {
        quantile_depths[thread_idx * num_depth_quantiles + current_quantile_idx] = -1.0f;
        quantile_point_indices[thread_idx * num_depth_quantiles + current_quantile_idx] = UINT32_MAX;
        current_quantile_idx++;
    }

    for (uint32_t i = 0; i < 3; ++i) {
        ray_rgba[thread_idx * 4 + i] = attr_scalar(accumulated_rgb[i]);
    }
    ray_rgba[thread_idx * 4 + 3] = attr_scalar(1.f - transmittance);

    if (num_intersections)
        num_intersections[thread_idx] = n;
}


// sdfoam backward
template <typename attr_scalar, int sh_degree, int block_size>
__global__ void backward_sdf(TraceSettings settings,
                         const Vec3f *__restrict__ points,
                         const attr_scalar *__restrict__ attributes,
                         const uint32_t *__restrict__ point_adjacency,
                         const uint32_t *__restrict__ point_adjacency_offsets,
                         const Vec4h *__restrict__ adjacent_diff,
                         const Ray *__restrict__ rays,
                         uint32_t num_rays,
                         const uint32_t *__restrict__ start_point_index,
                         uint32_t num_depth_quantiles,
                         const float *__restrict__ depth_quantiles,
                         const uint32_t *__restrict__ quantile_point_indices, // (unused here; kept for API parity)
                         const attr_scalar *__restrict__ ray_rgba,
                         const attr_scalar *__restrict__ ray_rgba_grad,
                         const float *__restrict__ depth_grad,
                         const attr_scalar *__restrict__ ray_error,
                         Ray *__restrict__ ray_grad,          // (unused; kept for API parity)
                         Vec3f *__restrict__ points_grad,
                         attr_scalar *__restrict__ attribute_grad,
                         attr_scalar *__restrict__ point_error,
                         const attr_scalar *__restrict__ sharpness,
                         float *__restrict__ sharpness_grad) {

                            auto sigmoidf = [] __device__ (float x) {
        return 1.f / (1.f + __expf(-x));
    };
    auto read_attr_channel = [&] __device__ (const attr_scalar* base, int dim, int idx, int ch) -> float {
        if constexpr (std::is_same<attr_scalar, __half>::value) {
            const __half* h = reinterpret_cast<const __half*>(base);
            return __half2float(h[idx * dim + ch]);
        } else {
            return static_cast<float>(base[idx * dim + ch]);
        }
    };

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays) return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    // normalized ray
    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    // upstream grads (RGBA and depth quantile grads)
    Vec4f rgba_grad_v, rgba_v;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        rgba_grad_v[i] = (float)ray_rgba_grad[thread_idx * 4 + i];
        rgba_v[i]      = (float)ray_rgba[thread_idx * 4 + i];
    }
    const float *ray_depth_quantiles = depth_quantiles ? (depth_quantiles + thread_idx * num_depth_quantiles) : nullptr;
    const float *ray_depth_grad      = depth_grad      ? (depth_grad      + thread_idx * num_depth_quantiles) : nullptr;

    float err_scalar = 0.f;
    if (ray_error) err_scalar = (float)ray_error[thread_idx];

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    // loader: RGB from SH, last slot is SDF (not used for gating in sdfoam)
    auto load_attributes = [&] __device__ (uint32_t v_idx, Vec3f &rgb, float &sdf_val) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        sdf_val = (float)attr_ptr[attr_memory_size - 1];
        rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
    };

    // running transmittance and color accumulation
    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    // --- SDF first-hit debug state ---
    float sdf_at_first_hit = 0.0f;
    float sdf_s0 = 0.0f, sdf_s1 = 0.0f;
    bool  has_first_hit = false;


    // quantile iteration state
    uint32_t current_quantile_idx = 0;
    float current_quantile = 0.f;
    if (ray_depth_quantiles && num_depth_quantiles > 0) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }

    // Coupling accumulator for dependence on log(T0) of *subsequent* quantiles
    float g_logT = 0.0f;

    // geometry accumulation across segments
    uint32_t prev_point_idx = UINT32_MAX;
    Vec3f prev_point = Vec3f::Zero();
    Vec3f prev_point_grad = Vec3f::Zero();
    Vec3f current_point_grad = Vec3f::Zero();
    Vec3f next_point_grad = Vec3f::Zero();

    // main per-segment functor
    auto functor = [&] __device__ (uint32_t point_idx,
                                   uint32_t next_point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point) {
        // --- forward recompute for this segment (matches your forward) ---
        Vec3f rgb_primal;
        float sdf_site;
        load_attributes(point_idx, rgb_primal, sdf_site);

        const float delta_t = fmaxf(t_1 - t_0, 0.0f);

        const int D = attr_memory_size;
        const int sdf_ch = (settings.sdf_channel >= 0) ? settings.sdf_channel : (D - 1);

        float d_i = read_attr_channel(attributes, D, point_idx, sdf_ch);

        float sha = std::is_same<attr_scalar,__half>::value
                  ? __half2float(*reinterpret_cast<const __half*>(sharpness))
                  : static_cast<float>(*sharpness);

        // NeuS-as-density
        float phi   = sigmoidf(-sha * d_i);
        float s_pri = sha * phi * (1.f - phi);
        float alpha = 1.f - __expf(-s_pri * delta_t);

        // composite color
        const float weight = transmittance * alpha;
        accumulated_rgb += weight * rgb_primal;
        if (point_error) atomicAdd(point_error + point_idx, (attr_scalar)(weight * err_scalar));

        // --- dL/dalpha from color & alpha channel ---
        Vec3f dL_drgb_primal = rgba_grad_v.template head<3>() * weight;

        Vec3f rgb_rest = rgba_v.template head<3>() - accumulated_rgb;
        rgb_rest /= (transmittance * (1.f - alpha) + 1e-6f);

        float dL_dalpha = 0.f;
        dL_dalpha += transmittance * (rgb_primal - rgb_rest).dot(rgba_grad_v.template head<3>());
        dL_dalpha += (1.f - rgba_v[3]) * rgba_grad_v[3] / (1.f - alpha + 1e-6f);

        // quantile coupling from T0 of *later* segments:
        // log T1 = log T0 + log(1-α)  =>  ∂/∂α of future-terms = -g_logT / (1-α)
        dL_dalpha += - g_logT / fmaxf(1.f - alpha, 1e-12f);

        // --- chain to s and Δt ---
        const float one_minus_alpha = (1.f - alpha);
        const float dalpha_ds       = delta_t * one_minus_alpha;
        const float dalpha_ddelta   = s_pri   * one_minus_alpha;

        float dL_ds     = dL_dalpha * dalpha_ds;
        float dL_ddelta = dL_dalpha * dalpha_ddelta;

        // --- quantiles (sdfoam linearized T; match your forward) ---
        const float next_transmittance = transmittance * (1.f - alpha);

        if (!has_first_hit) {
            const int i0 = point_idx;
            const int i1 = next_point_idx;

            // attributes layout: [ SH coeffs (sh_dim) | last scalar ]
            constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
            constexpr int attr_memory_size = 1 + sh_dim;

            const attr_scalar* a0 = attributes + i0 * attr_memory_size;
            const attr_scalar* a1 = attributes + i1 * attr_memory_size;

            // last scalar is SDF in sdfoam mode
            sdf_s0 = (float)a0[attr_memory_size - 1];
            sdf_s1 = (float)a1[attr_memory_size - 1];

            sdf_at_first_hit = 0.5f * (sdf_s0 + sdf_s1);

            has_first_hit = true;
        }

        float dL_dt0_units = 0.f;  // sums depth_grad_i * ∂tq/∂t0  (= 1 per crossing)
        // We'll aggregate ∂tq/∂Δ as dL_ddelta (+= depth_grad_i * frac)
        // and ∂tq/∂α as dL_dalpha (+= depth_grad_i * (-Δ * frac / α))
        // and ∂tq/∂logT0 into g_logT (+= depth_grad_i * Δ * q / (α*T0))

        if (ray_depth_quantiles && num_depth_quantiles > 0) {
            const float T0   = transmittance;
            const float drop = T0 - next_transmittance; // = T0 * alpha

            if (drop > 1e-12f) {
                while (current_quantile_idx < num_depth_quantiles &&
                       next_transmittance < current_quantile) {

                    const float q = current_quantile;
                    const float grad_q = ray_depth_grad[current_quantile_idx];

                    float frac = (T0 - q) / drop;
                    frac = fminf(1.f, fmaxf(0.f, frac));

                    dL_dt0_units += grad_q;
                    dL_ddelta += grad_q * frac;

                    if (alpha > 1e-12f) {
                        dL_dalpha += grad_q * ( -delta_t * frac / alpha );
                    }

                    // ∂tq/∂logT0 = Δ * q / (α * T0)
                    if (alpha > 1e-12f && T0 > 1e-20f) {
                        g_logT += grad_q * ( delta_t * q / (alpha * T0) );
                    }

                    // next quantile
                    ++current_quantile_idx;
                    if (current_quantile_idx < num_depth_quantiles) {
                        current_quantile = ray_depth_quantiles[current_quantile_idx];
                    }
                }
            }
        }

        // Convert Δ contributions to t0/t1, and add the "unit" t0 terms
        float dL_dt0 = dL_dt0_units - dL_ddelta; // ∂Δ/∂t0 = -1
        float dL_dt1 = dL_ddelta;                // ∂Δ/∂t1 = +1

        // --- geometry grads via intersection derivatives ---
        Vec3f dt0_dprev_point;
        if (prev_point_idx != UINT32_MAX) {
            dt0_dprev_point = cell_intersection_grad(prev_point, current_point, ray);
        } else {
            dt0_dprev_point = Vec3f::Zero();
        }
        Vec3f dt1_dcurrent_point = cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcurrent_point = cell_intersection_grad(current_point, prev_point, ray);
        Vec3f dt1_dnext_point    = cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad     += dL_dt0 * dt0_dprev_point;
        current_point_grad  += dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad     += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx != UINT32_MAX) {
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        }
        prev_point       = current_point;
        prev_point_idx   = point_idx;
        prev_point_grad  = current_point_grad;
        current_point_grad = next_point_grad;
        next_point_grad    = Vec3f::Zero();

        // --- backprop to SDF and sharpness through s ---
        // dphi/dd = -sha * phi * (1-phi)
        float dphi_dd  = -sha * phi * (1.f - phi);
        // ds/dd   = sha * (1 - 2phi) * dphi/dd = -sha^2 * phi(1-phi)(1-2phi)
        float ds_dd    = sha * (1.f - 2.f*phi) * dphi_dd;
        // ds/dsha = phi(1-phi) + sha*(1-2phi)*(-d_i*phi(1-phi))
        float ds_dsha  = phi*(1.f - phi) * (1.f - sha * d_i * (1.f - 2.f*phi));

        // Accumulate
        float dL_dd_i   = dL_ds * ds_dd;
        if (settings.sdf_channel >= 0) {
            atomicAdd(attribute_grad + point_idx * attr_memory_size + sdf_ch,
                      (attr_scalar)dL_dd_i);
        }
        if (sharpness_grad) {
            float dL_dsha = dL_ds * ds_dsha;
            atomicAdd(sharpness_grad, dL_dsha);
        }

        // --- SH color grads ---
        for (uint32_t i = 0; i < 3; ++i) {
            if (rgb_primal[i] == 0.0f) dL_drgb_primal[i] = 0.0f;
        }
        write_rgb_grad_to_sh<attr_scalar, sh_degree>(sh_coeffs, dL_drgb_primal,
                                                     attribute_grad + point_idx * attr_memory_size);

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    // run the traversal
    uint32_t start_point = start_point_index[thread_idx];
    trace<block_size, 2>(ray,
                         points,
                         point_adjacency,
                         point_adjacency_offsets,
                         adjacent_diff,
                         start_point,
                         settings.max_intersections,
                         functor,
                         /*use_safe_mode*/ true);
}


// Radfoam backward
template <typename attr_scalar, int sh_degree, int block_size>
__global__ void backward(TraceSettings settings,
                         const Vec3f *__restrict__ points,
                         const attr_scalar *__restrict__ attributes,
                         const uint32_t *__restrict__ point_adjacency,
                         const uint32_t *__restrict__ point_adjacency_offsets,
                         const Vec4h *__restrict__ adjacent_diff,
                         const Ray *__restrict__ rays,
                         uint32_t num_rays,
                         const uint32_t *__restrict__ start_point_index,
                         uint32_t num_depth_quantiles,
                         const float *__restrict__ depth_quantiles,
                         const uint32_t *__restrict__ quantile_point_indices,
                         const attr_scalar *__restrict__ ray_rgba,
                         const attr_scalar *__restrict__ ray_rgba_grad,
                         const float *__restrict__ depth_grad,
                         const attr_scalar *__restrict__ ray_error,
                         Ray *__restrict__ ray_grad,
                         Vec3f *__restrict__ points_grad,
                         attr_scalar *__restrict__ attribute_grad,
                         attr_scalar *__restrict__ point_error) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    const float *ray_depth_grad = depth_grad + thread_idx * num_depth_quantiles;
    const float *ray_depth_quantiles =
        depth_quantiles + thread_idx * num_depth_quantiles;

    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&](uint32_t v_idx, Vec3f &rgb, float &s) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        s = (float)attr_ptr[attr_memory_size - 1];
        if (s > 1e-6f) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            rgb = Vec3f::Zero();
        }
    };

    Vec4f rgba_grad, rgba;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        rgba_grad[i] = (float)ray_rgba_grad[thread_idx * 4 + i];
        rgba[i] = (float)ray_rgba[thread_idx * 4 + i];
    }

    float error;
    if (ray_error) {
        error = (float)ray_error[thread_idx];
    }

    uint32_t current_quantile_idx = 0;
    float current_quantile;
    if (depth_quantiles) {
        current_quantile = ray_depth_quantiles[current_quantile_idx];
    }
    float current_depth_grad = 0.0f;
    for (uint32_t i = 0; i < num_depth_quantiles; ++i) {
        if (quantile_point_indices[thread_idx * num_depth_quantiles + i] !=
            UINT32_MAX) {
            uint32_t point_idx =
                quantile_point_indices[thread_idx * num_depth_quantiles + i];
            float s = (float)
                attributes[point_idx * attr_memory_size + attr_memory_size - 1];
            current_depth_grad += ray_depth_grad[i] / s;
        }
    }

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    // --- SDF first-hit debug state ---
    float sdf_at_first_hit = 0.0f;
    float sdf_s0 = 0.0f, sdf_s1 = 0.0f;
    bool  has_first_hit = false;


    uint32_t prev_point_idx = UINT32_MAX;
    Vec3f prev_point = Vec3f::Zero();
    Vec3f prev_point_grad = Vec3f::Zero();

    Vec3f current_point_grad = Vec3f::Zero();
    Vec3f next_point_grad = Vec3f::Zero();

    auto functor = [&](uint32_t point_idx,
                          uint32_t next_point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        Vec3f rgb_primal;
        float s_primal;

        load_attributes(point_idx, rgb_primal, s_primal);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 1 - expf(-s_primal * delta_t);
        float weight = transmittance * alpha;
        float dalpha_ds_primal = delta_t * (1 - alpha);
        float dalpha_ddelta_t = 0.0f;
        if (delta_t > 0.0f) {
            dalpha_ddelta_t = s_primal * (1 - alpha);
        }

        accumulated_rgb += weight * rgb_primal;
        if (point_error) {
            atomicAdd(point_error + point_idx, (attr_scalar)(weight * error));
        }

        Vec3f dL_drgb_primal = rgba_grad.template head<3>() * weight;

        Vec3f rgb_rest = rgba.template head<3>() - accumulated_rgb;
        rgb_rest /= (transmittance * (1 - alpha + 1e-6f));

        float dL_dalpha =
            transmittance *
            (rgb_primal - rgb_rest).dot(rgba_grad.template head<3>());
        dL_dalpha += (1 - rgba[3]) * rgba_grad[3] / (1 - alpha + 1e-6f);

        float dL_ds_primal = dL_dalpha * dalpha_ds_primal;
        float dL_ddelta_t = dL_dalpha * dalpha_ddelta_t;

        float dL_dt0 = 0.0f;

        float next_transmittance = transmittance * (1.f - alpha);

        if (!has_first_hit) {
            const int i0 = point_idx;
            const int i1 = next_point_idx;

            constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
            constexpr int attr_memory_size = 1 + sh_dim;

            const attr_scalar* a0 = attributes + i0 * attr_memory_size;
            const attr_scalar* a1 = attributes + i1 * attr_memory_size;

            sdf_s0 = (float)a0[attr_memory_size - 1];
            sdf_s1 = (float)a1[attr_memory_size - 1];

            sdf_at_first_hit = 0.5f * (sdf_s0 + sdf_s1);

            has_first_hit = true;
        }

        while (current_quantile_idx < num_depth_quantiles &&
               next_transmittance < current_quantile) {

            float depth_grad_i =
                ray_depth_grad[current_quantile_idx] / s_primal;
            dL_dt0 += depth_grad_i;
            dL_ds_primal += -depth_grad_i *
                            logf(transmittance / current_quantile) / s_primal;

            current_depth_grad -= depth_grad_i;

            current_quantile_idx++;
            if (current_quantile_idx < num_depth_quantiles) {
                current_quantile = ray_depth_quantiles[current_quantile_idx];
            }
        }

        if (current_quantile_idx < num_depth_quantiles) {
            dL_ds_primal += -delta_t * current_depth_grad;
            dL_ddelta_t += -s_primal * current_depth_grad;
        }

        dL_dt0 += -dL_ddelta_t;
        float dL_dt1 = dL_ddelta_t;

        Vec3f dt0_dprev_point;
        if (prev_point_idx != UINT32_MAX) {
            dt0_dprev_point =
                cell_intersection_grad(prev_point, current_point, ray);
        } else {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point =
            cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcurrent_point =
            cell_intersection_grad(current_point, prev_point, ray);

        Vec3f dt1_dnext_point =
            cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad += dL_dt0 * dt0_dprev_point;
        current_point_grad +=
            dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx != UINT32_MAX) {
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        }
        prev_point = current_point;
        prev_point_idx = point_idx;
        prev_point_grad = current_point_grad;

        current_point_grad = next_point_grad;
        next_point_grad = Vec3f::Zero();

        transmittance = next_transmittance;

        for (uint32_t i = 0; i < 3; ++i) {
            if (rgb_primal[i] == 0.0f) {
                dL_drgb_primal[i] = 0.0f;
            }
        }
        write_rgb_grad_to_sh<attr_scalar, sh_degree>(
            sh_coeffs,
            dL_drgb_primal,
            attribute_grad + point_idx * attr_memory_size);
        atomicAdd(attribute_grad + point_idx * attr_memory_size +
                      (attr_memory_size - 1),
                  (attr_scalar)dL_ds_primal);

        return transmittance > settings.weight_threshold;
    };

    uint32_t start_point = start_point_index[thread_idx];

    trace<block_size, 2>(ray,
                         points,
                         point_adjacency,
                         point_adjacency_offsets,
                         adjacent_diff,
                         start_point,
                         settings.max_intersections,
                         functor,
                         false);
}

template <typename attr_scalar, int sh_degree, int block_size>
__global__ void
visualization(TraceSettings settings,
              const Vec3f *__restrict__ points,
              const attr_scalar *__restrict__ attributes,
              const uint32_t *__restrict__ point_adjacency,
              const uint32_t *__restrict__ point_adjacency_offsets,
              const Vec4h *__restrict__ adjacent_diff,
              VisualizationSettings vis_settings,
              CMapTable cmap_table,
              Camera camera,
              uint32_t num_points,
              uint32_t point_adjacency_size,
              cudaSurfaceObject_t output_rgba,
              uint32_t start_point_index,
              const attr_scalar *__restrict__ sharpness) {

    auto sigmoidf = [] __device__ (float x) {
        return 1.f / (1.f + __expf(-x));
    };
    auto read_attr_channel = [&] __device__ (const attr_scalar* base, int dim, int idx, int ch) -> float {
        if constexpr (std::is_same<attr_scalar, __half>::value) {
            const __half* h = reinterpret_cast<const __half*>(base);
            return __half2float(h[idx * dim + ch]);
        } else {
            return static_cast<float>(base[idx * dim + ch]);
        }
    };

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    if (ray.direction.norm() < 0.1f) {
        surf2Dwrite(0, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
        return;
    }

    // ray.direction /= ray.direction.norm();
    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&] __device__ (uint32_t v_idx, Vec3f &rgb, float &last_scalar) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        last_scalar = static_cast<float>(attr_ptr[attr_memory_size - 1]);
        if (settings.alpha_mode == AlphaNeuS) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            if (last_scalar > 1e-6f) {
                rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
            } else {
                rgb = Vec3f::Zero();
            }
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();
    float depth = 0.0f;
    bool depth_quantile_passed = false;
    float  sdf_at_first_hit = 0.0f;
    bool   has_first_hit    = false;
    float  sdf_s0 = 0.0f, sdf_s1 = 0.0f;
    bool any_cell_below = false;

    auto functor = [&] __device__ (uint32_t point_idx,
                                   uint32_t next_point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point) {
        Vec3f rgb_primal;
        float last_scalar; 
        load_attributes(point_idx, rgb_primal, last_scalar);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // per-segment alpha
        float alpha = 0.f;

        if (settings.alpha_mode == AlphaNeuS && settings.sdf_channel >= 0) {

            const int D = attr_memory_size;
            const int sdf_ch = (settings.sdf_channel >= 0) ? settings.sdf_channel : (D - 1);

            // float sha = (float)(*sharpness);
            float sha;
            if constexpr (std::is_same<attr_scalar, __half>::value) {
                sha = __half2float(*reinterpret_cast<const __half*>(sharpness));
            } else {
                sha = static_cast<float>(*sharpness);
            }
            // sha = __expf(sha * 10.f); // decode

            float d_i   = read_attr_channel(attributes, D, point_idx,      sdf_ch);
            // float d_ip1 = read_attr_channel(attributes, D, next_point_idx, sdf_ch);

            // float phi_i   = sigmoidf(-sha * d_i);
            // float phi_ip1 = sigmoidf(-sha * d_ip1);

            // float numer = fmaxf(0.f, phi_i - phi_ip1);
            // float denom = fmaxf(settings.eps, phi_i);
            // alpha = fminf(1.f, numer / denom);


            last_scalar = sha * sigmoidf(-sha * d_i) * (1.f - sigmoidf(-sha * d_i));
            float delta_t = fmaxf(t_1 - t_0, 0.0f);
            alpha = 1.f - __expf(-last_scalar * delta_t);
        } else {
            alpha = 1.f - __expf(-last_scalar * delta_t);
        }

        // composite
        accumulated_rgb += transmittance * alpha * rgb_primal;

        float next_transmittance = transmittance * (1.f - alpha);

        constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
        constexpr int attr_memory_size = 1 + sh_dim;

        const attr_scalar* a_cur = attributes + point_idx * attr_memory_size;
        float sdf_cur = (float)a_cur[attr_memory_size - 1];

        if (!settings.sdf_exact_mode) {
            // Mode 1: Max |SDF|
            const float tau = fmaxf(settings.max_sdf, 1e-6f);
            if (fabsf(sdf_cur) <= tau) any_cell_below = true;
        } else {
            // Mode 2: Exact SDF ± tolerance
            const float tol = fmaxf(settings.sdf_tolerance, 0.0f);
            if (fabsf(sdf_cur - settings.sdf_value) <= tol) any_cell_below = true;
        }


        // Depth quantile: find t where T falls below vis_settings.depth_quantile
        if (!depth_quantile_passed && next_transmittance < vis_settings.depth_quantile) {
            float q = vis_settings.depth_quantile;

            if (settings.alpha_mode == AlphaNeuS && settings.sdf_channel >= 0) {
                // Linearize T across segment: T1 = T0*(1-α),
                // drop = T0 - T1 = T0*α, frac = (T0 - q)/drop
                if (next_transmittance < q && alpha > 1e-12f) {
                    float drop = transmittance - next_transmittance; // T0*alpha
                    float frac = (transmittance - q) / drop;
                    frac = fminf(1.f, fmaxf(0.f, frac));
                    depth = t_0 + frac * (t_1 - t_0);
                    depth_quantile_passed = true;

                    
                    if (!has_first_hit) {
                        has_first_hit = true;

                        // per-point attribute layout: [ SH...(sh_dim) | last_scalar ]
                        // last_scalar is SDF when settings.alpha_mode == AlphaNeuS
                        constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
                        constexpr int attr_memory_size = 1 + sh_dim;

                        const attr_scalar* a0 = attributes + point_idx      * attr_memory_size;
                        const attr_scalar* a1 = attributes + next_point_idx * attr_memory_size;

                        sdf_s0 = (float)a0[attr_memory_size - 1];
                        sdf_s1 = (float)a1[attr_memory_size - 1];

                        sdf_at_first_hit = 0.5f * (sdf_s0 + sdf_s1);
                    }
                }
            } else {
                // Analytic solve for density-based: t_q = t0 + log(T0/q)/sigma
                float sigma = fmaxf(last_scalar, 1e-12f);
                if (next_transmittance < q) {
                    depth = t_0 + logf(transmittance / q) / sigma;
                    depth_quantile_passed = true;
                }
            }
        }

        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    bool use_safe_mode = true;

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point_index,
                                      settings.max_intersections,
                                      functor,
                                      use_safe_mode,
                                      num_points,
                                      point_adjacency_size);

    uint32_t out;

    if (vis_settings.mode == VisualizationMode::RGB) {
        Vec3f color = accumulated_rgb;

        Vec3f bg_color;
        if (vis_settings.checker_bg) {
            int is = 2 * ((pix_i / 20) % 2) - 1;
            int js = 2 * ((pix_j / 20) % 2) - 1;
            if (is * js > 0) {
                bg_color = Vec3f(0.3f, 0.3f, 0.3f);
            } else {
                bg_color = Vec3f(0.5f, 0.5f, 0.5f);
            }
        } else {
            bg_color = *vis_settings.bg_color;
        }

        color += transmittance * bg_color;

        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Depth) {
        float val = depth / vis_settings.max_depth;
        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);
        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::Alpha) {
        out = make_rgba8(1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f - transmittance,
                         1.0f);
    } else if (vis_settings.mode == VisualizationMode::Intersections) {
        float val = float(n - 1) / float(settings.max_intersections);
        Vec3f color = colormap(val, vis_settings.color_map, cmap_table);
        out = make_rgba8(color[0], color[1], color[2], 1.0f);
    } else if (vis_settings.mode == VisualizationMode::SDF) {
        const float accum_alpha = 1.0f - transmittance;
        const float gate = fmaxf(settings.weight_threshold, 1e-4f);

        // background (same as RGB)
        Vec3f bg_color;
        if (vis_settings.checker_bg) {
            int is = 2 * ((pix_i / 20) % 2) - 1;
            int js = 2 * ((pix_j / 20) % 2) - 1;
            bg_color = (is * js > 0) ? Vec3f(0.3f, 0.3f, 0.3f) : Vec3f(0.5f, 0.5f, 0.5f);
        } else {
            bg_color = *vis_settings.bg_color;
        }

        if (accum_alpha > gate && any_cell_below && has_first_hit) {
            // Choose normalization around either 0 (Max |SDF|) or sdf_value (Exact)
            float center = settings.sdf_exact_mode ? settings.sdf_value : 0.0f;
            float denom  = settings.sdf_exact_mode
                            ? fmaxf(settings.sdf_tolerance, 1e-6f)
                            : fmaxf(settings.max_sdf,      1e-6f);

            float u = 0.5f + 0.5f * tanhf((sdf_at_first_hit - center) / denom);
            u = fminf(fmaxf(u, 0.0f), 1.0f);

            Vec3f sdf_color = colormap(u, vis_settings.color_map, cmap_table);

            Vec3f final = sdf_color * accum_alpha + bg_color * (1.0f - accum_alpha);
            out = make_rgba8(final[0], final[1], final[2], 1.0f);
        } else {
            out = make_rgba8(bg_color[0], bg_color[1], bg_color[2], 1.0f);
        }


    }

    surf2Dwrite(out, output_rgba, 4 * pix_i, camera.height - 1 - pix_j);
}


template <typename attr_scalar, int sh_degree, int block_size>
__global__ void benchmark(TraceSettings settings,
                          const Vec3f *__restrict__ points,
                          const attr_scalar *__restrict__ attributes,
                          const uint32_t *__restrict__ point_adjacency,
                          const uint32_t *__restrict__ point_adjacency_offsets,
                          const Vec4h *__restrict__ adjacent_diff,
                          VisualizationSettings vis_settings,
                          Camera camera,
                          uint32_t num_points,
                          uint32_t point_adjacency_size,
                          const uint32_t *__restrict__ start_point_index,
                          uint32_t *__restrict__ output_rgba,
                          const attr_scalar *__restrict__ sharpness) {

    auto sigmoidf = [] __device__ (float x) {
        return 1.f / (1.f + __expf(-x));
    };
    auto read_attr_channel = [&] __device__ (const attr_scalar* base, int dim, int idx, int ch) -> float {
        if constexpr (std::is_same<attr_scalar, __half>::value) {
            const __half* h = reinterpret_cast<const __half*>(base);
            return __half2float(h[idx * dim + ch]);
        } else {
            return static_cast<float>(base[idx * dim + ch]);
        }
    };

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pix_i = thread_idx % camera.width;
    uint32_t pix_j = thread_idx / camera.width;

    if (pix_i >= camera.width || pix_j >= camera.height)
        return;

    constexpr int sh_dim = 3 * (1 + sh_degree) * (1 + sh_degree);
    constexpr int attr_memory_size = 1 + sh_dim;

    Ray ray = cast_ray(camera, pix_i, pix_j);
    uint32_t out_idx = (camera.height - 1 - pix_j) * camera.width + pix_i;
    if (ray.direction.norm() < 0.1f) {
        Vec3f bg_color = *vis_settings.bg_color;
        output_rgba[out_idx] =
            make_rgba8(bg_color[0], bg_color[1], bg_color[2], 1.0f);
        return;
    }

    ray.direction /= ray.direction.norm();
    auto sh_coeffs = sh_coefficients<sh_degree>(ray.direction);

    auto load_attributes = [&] __device__ (uint32_t v_idx, Vec3f &rgb, float &last_scalar) {
        const attr_scalar *attr_ptr = attributes + v_idx * attr_memory_size;
        last_scalar = static_cast<float>(attr_ptr[attr_memory_size - 1]);
        if (settings.alpha_mode == AlphaNeuS) {
            rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
        } else {
            if (last_scalar > 1e-6f) {
                rgb = load_sh_as_rgb<attr_scalar, sh_degree>(sh_coeffs, attr_ptr);
            } else {
                rgb = Vec3f::Zero();
            }
        }
    };

    float transmittance = 1.0f;
    Vec3f accumulated_rgb = Vec3f::Zero();

    auto functor = [&] __device__ (uint32_t point_idx,
                                   uint32_t next_point_idx,
                                   float t_0,
                                   float t_1,
                                   const Vec3f &current_point,
                                   const Vec3f &next_point) {
        Vec3f rgb_primal;
        float last_scalar;
        load_attributes(point_idx, rgb_primal, last_scalar);

        const int D = attr_memory_size;
        const int sdf_ch = (settings.sdf_channel >= 0) ? settings.sdf_channel : (D - 1);

        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float alpha = 0.f;

        if (settings.alpha_mode == AlphaNeuS && settings.sdf_channel >= 0) {
            float d_i = read_attr_channel(attributes, D, point_idx, sdf_ch);

            float sha;
            if constexpr (std::is_same<attr_scalar, __half>::value) {
                sha = __half2float(*reinterpret_cast<const __half*>(sharpness));
            } else {
                sha = static_cast<float>(*sharpness);
            }

            float sigma = sha * sigmoidf(-sha * d_i) * (1.f - sigmoidf(-sha * d_i));
            alpha = 1.f - __expf(-sigma * delta_t);
        } else {
            float sigma = last_scalar;
            alpha = 1.f - __expf(-sigma * delta_t);
        }

        // composite
        accumulated_rgb += transmittance * alpha * rgb_primal;
        transmittance = transmittance * (1.f - alpha);

        return transmittance > settings.weight_threshold;
    };

    bool use_safe_mode = true;

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      *start_point_index,
                                      settings.max_intersections,
                                      functor,
                                      use_safe_mode,
                                      num_points,
                                      point_adjacency_size);

    Vec3f bg_color;
    if (vis_settings.checker_bg) {
        int is = 2 * ((pix_i / 20) % 2) - 1;
        int js = 2 * ((pix_j / 20) % 2) - 1;
        bg_color = (is * js > 0) ? Vec3f(0.3f, 0.3f, 0.3f)
                                 : Vec3f(0.5f, 0.5f, 0.5f);
    } else {
        bg_color = *vis_settings.bg_color;
    }

    Vec3f color = accumulated_rgb + transmittance * bg_color;
    output_rgba[out_idx] = make_rgba8(color[0], color[1], color[2], 1.0f);
}


__global__ void prefetch_adjacent_diff_kernel(
    const Vec3f *__restrict__ points,
    uint32_t num_points,
    uint32_t point_adjacency_size,
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    Vec4h *__restrict__ adjacent_diff) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    Vec3f p = points[i];
    uint32_t offset_start = point_adjacency_offsets[i];
    uint32_t offset_end = point_adjacency_offsets[i + 1];
    uint32_t num_adjacent = offset_end - offset_start;

    for (uint32_t j = 0; j < num_adjacent; ++j) {
        uint32_t adjacent_idx = point_adjacency[offset_start + j];
        Vec3f q = points[adjacent_idx];
        Vec3f diff = q - p;
        adjacent_diff[offset_start + j] = Vec4h(diff[0], diff[1], diff[2], 0);
    }
}

void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream) {
    launch_kernel_1d<256>(prefetch_adjacent_diff_kernel,
                          num_points,
                          stream,
                          points,
                          num_points,
                          point_adjacency_size,
                          point_adjacency,
                          point_adjacency_offsets,
                          adjacent_diff);
}

template <typename attr_scalar, int sh_degree>
class CUDATracingPipeline : public Pipeline {
  public:
    CUDATracingPipeline() = default;

    virtual ~CUDATracingPipeline() {}

    void trace_forward(const TraceSettings &settings,
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
                       void *alpha_output) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            forward<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            num_depth_quantiles,
            depth_quantiles,
            static_cast<attr_scalar *>(ray_rgba),
            quantile_depths,
            quantile_point_indices,
            num_intersections,
            static_cast<attr_scalar *>(point_contribution),
            reinterpret_cast<attr_scalar *>(sharpness),
            reinterpret_cast<float *>(alpha_output));
    }

    void trace_backward(const TraceSettings &settings,
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
                        float *sharpness_grad) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        if (settings.alpha_mode == AlphaNeuS){
            launch_kernel_1d<block_size>(
                backward_sdf<attr_scalar, sh_degree, block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                reinterpret_cast<const attr_scalar *>(attributes),
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                rays,
                num_rays,
                start_point_index,
                num_depth_quantiles,
                depth_quantiles,
                quantile_point_indices,
                static_cast<const attr_scalar *>(ray_rgba),
                static_cast<const attr_scalar *>(ray_rgba_grad),
                depth_grad,
                static_cast<const attr_scalar *>(ray_error),
                ray_grad,
                points_grad,
                static_cast<attr_scalar *>(attribute_grad),
                static_cast<attr_scalar *>(point_error),
                reinterpret_cast<attr_scalar *>(sharpness),
                sharpness_grad);
        } else {
            launch_kernel_1d<block_size>(
                backward<attr_scalar, sh_degree, block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                reinterpret_cast<const attr_scalar *>(attributes),
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                rays,
                num_rays,
                start_point_index,
                num_depth_quantiles,
                depth_quantiles,
                quantile_point_indices,
                static_cast<const attr_scalar *>(ray_rgba),
                static_cast<const attr_scalar *>(ray_rgba_grad),
                depth_grad,
                static_cast<const attr_scalar *>(ray_error),
                ray_grad,
                points_grad,
                static_cast<attr_scalar *>(attribute_grad),
                static_cast<attr_scalar *>(point_error));
        }
    }
    void trace_visualization(const TraceSettings &settings,
                             const VisualizationSettings &vis_settings,
                             const Camera &camera,
                             CMapTable cmap_table,
                             uint32_t num_points,
                             uint32_t num_tets,
                             const void *points,
                             const void *attributes,
                             const void *point_adjacency,
                             const void *point_adjacency_offsets,
                             const void *adjacent_diff,
                             uint32_t start_index,
                             uint64_t output_surface,
                             const void *stream,
                             void *sharpness) override {

        uint32_t num_rays = camera.width * camera.height;
        constexpr uint32_t block_size = 128;

        launch_kernel_1d<block_size>(
            visualization<attr_scalar, sh_degree, block_size>,
            num_rays,
            stream,
            settings,
            reinterpret_cast<const Vec3f *>(points),
            reinterpret_cast<const attr_scalar *>(attributes),
            reinterpret_cast<const uint32_t *>(point_adjacency),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
            reinterpret_cast<const Vec4h *>(adjacent_diff),
            vis_settings,
            cmap_table,
            camera,
            num_points,
            num_tets,
            output_surface,
            start_index,
            reinterpret_cast<attr_scalar *>(sharpness));
    }

    void trace_benchmark(const TraceSettings &settings,
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
                         void *sharpness) override {

        uint32_t num_rays = camera.width * camera.height;

        constexpr uint32_t block_size = 512;
        launch_kernel_1d<block_size>(
            benchmark<attr_scalar, sh_degree, block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            reinterpret_cast<const attr_scalar *>(attributes),
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff,
            vis_settings,
            camera,
            num_points,
            point_adjacency_size,
            start_point_index,
            ray_rgba,
            reinterpret_cast<attr_scalar *>(sharpness));
    }

    uint32_t attribute_dim() const override {
        return 1 + 3 * (1 + sh_degree) * (1 + sh_degree);
    }

    ScalarType attribute_type() const override {
        return scalar_code<attr_scalar>();
    }
};

std::shared_ptr<Pipeline> create_pipeline(int sh_degree, ScalarType attr_type) {

    if (attr_type == ScalarType::Float32) {
        if (sh_degree == 0) {
            return std::make_shared<CUDATracingPipeline<float, 0>>();
        } else if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<float, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<float, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<float, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else if (attr_type == ScalarType::Float16) {
        if (sh_degree == 0) {
            return std::make_shared<CUDATracingPipeline<__half, 0>>();
        } else if (sh_degree == 1) {
            return std::make_shared<CUDATracingPipeline<__half, 1>>();
        } else if (sh_degree == 2) {
            return std::make_shared<CUDATracingPipeline<__half, 2>>();
        } else if (sh_degree == 3) {
            return std::make_shared<CUDATracingPipeline<__half, 3>>();
        } else {
            throw std::runtime_error("Unsupported SH degree");
        }
    } else {
        throw std::runtime_error("Unsupported attribute type");
    }
}

} // namespace sdfoam
