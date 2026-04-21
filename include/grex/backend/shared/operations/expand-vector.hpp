#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_VECTOR_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_VECTOR_HPP

#include <cstddef>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
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

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_EXPAND_VECTOR_HPP
