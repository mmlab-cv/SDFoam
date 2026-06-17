#pragma once

#include <cstddef>

#include <thrust/iterator/discard_iterator.h>

namespace cub {

template <typename Difference = ptrdiff_t>
class DiscardOutputIterator : public thrust::discard_iterator<Difference> {
  public:
    using thrust::discard_iterator<Difference>::discard_iterator;
    DiscardOutputIterator() : thrust::discard_iterator<Difference>(0) {}
};

} // namespace cub
