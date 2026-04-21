#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP

#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
////////////
// Scalar //
////////////

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

////////////
// Vector //
////////////

// unchanged size: no-op
template<AnyVector TVec, bool tZero>
inline TVec expand(TVec v, IndexTag<TVec::size> /*size*/, BoolTag<tZero> /*zero*/) {
  return v;
}

// sub-native → sub-native/native
template<typename T, std::size_t tPart, std::size_t tSize, std::size_t tDstSize, bool tZero>
inline VectorFor<T, tDstSize> expand(SubVector<T, tPart, tSize> v, IndexTag<tDstSize> size_tag,
                                     BoolTag<tZero> zero_tag) {
  using Work = VectorFor<T, std::min(tDstSize, min_native_size<T>)>;
  Work work = [&] {
    if constexpr (tZero) {
      return Work{full_cutoff(v).r};
    } else {
      return Work{v.registr()};
    }
  }();
  if constexpr (tDstSize <= min_native_size<T>) {
    return work;
  } else {
    return expand(work, size_tag, zero_tag);
  }
}

template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_any(TVec v, IndexTag<tSize> size) {
  return expand(v, size, false_tag);
}
template<std::size_t tSize, AnyVector TVec>
inline VectorFor<typename TVec::Value, tSize> expand_any(TVec v) {
  return expand(v, index_tag<tSize>, false_tag);
}

template<AnyVector TVec, std::size_t tSize>
inline VectorFor<typename TVec::Value, tSize> expand_zero(TVec v, IndexTag<tSize> size) {
  return expand(v, size, true_tag);
}
template<std::size_t tSize, AnyVector TVec>
inline VectorFor<typename TVec::Value, tSize> expand_zero(TVec v) {
  return expand(v, index_tag<tSize>, true_tag);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_HPP
