#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_SCALAR_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_SCALAR_HPP

#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
// Sub-native: Delegate to the native version
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize < min_native_size<T>)
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  return VectorFor<T, tSize>{expand(x, index_tag<min_native_size<T>>, zero)};
}

// Larger than the smallest native size: Merge with zero/undefined
template<Vectorizable T, std::size_t tSize, bool tZero>
requires(tSize > min_native_size<T>)
inline VectorFor<T, tSize> expand(Scalar<T> x, IndexTag<tSize> /*tag*/, BoolTag<tZero> zero) {
  constexpr std::size_t half = tSize / 2;
  return expand(expand(x, index_tag<half>, zero), index_tag<tSize>, zero);
}

template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_any(Scalar<T> x, IndexTag<tSize> size) {
  return expand(x, size, false_tag);
}
template<Vectorizable T, std::size_t tSize>
inline VectorFor<T, tSize> expand_zero(Scalar<T> x, IndexTag<tSize> size) {
  return expand(x, size, true_tag);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_SCALAR_HPP
