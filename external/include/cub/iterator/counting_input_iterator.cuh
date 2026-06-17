#pragma once

#include <cstddef>

#include <thrust/iterator/counting_iterator.h>

namespace cub {

template <typename T, typename Difference = ptrdiff_t>
using CountingInputIterator = thrust::counting_iterator<T, Difference>;

} // namespace cub
