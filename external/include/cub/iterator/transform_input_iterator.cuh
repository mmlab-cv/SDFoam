#pragma once

#include <cstddef>

#include <thrust/iterator/transform_iterator.h>

namespace cub {

template <typename ValueType,
          typename ConversionOp,
          typename InputIterator,
          typename Difference = ptrdiff_t>
using TransformInputIterator =
    thrust::transform_iterator<ConversionOp,
                               InputIterator,
                               ValueType,
                               ValueType>;

} // namespace cub
