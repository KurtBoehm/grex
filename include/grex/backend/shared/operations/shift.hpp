#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_SUBSUPER(NAME) \
  template<typename THalf> \
  inline SuperVector<THalf> NAME(SuperVector<THalf> v, AnyIndexTag auto offset) { \
    return {.lower = NAME(v.lower, offset), .upper = NAME(v.upper, offset)}; \
  } \
  template<IntVectorizable T, std::size_t tPart, std::size_t tSize> \
  inline SubVector<T, tPart, tSize> NAME(SubVector<T, tPart, tSize> v, AnyIndexTag auto offset) { \
    return SubVector<T, tPart, tSize>{NAME(v.full, offset)}; \
  }

GREX_SUBSUPER(shift_left)
GREX_SUBSUPER(shift_right)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHIFT_HPP
