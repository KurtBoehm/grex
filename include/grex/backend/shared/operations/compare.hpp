#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_NN_CMP(NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline SubMask<T, tPart, tSize> NAME(SubVector<T, tPart, tSize> a, \
                                       SubVector<T, tPart, tSize> b) { \
    return SubMask<T, tPart, tSize>{NAME(a.full, b.full)}; \
  } \
  template<typename THalf> \
  inline auto NAME(SuperVector<THalf> a, SuperVector<THalf> b) { \
    return SuperMask{.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }

GREX_NN_CMP(compare_eq)
GREX_NN_CMP(compare_neq)
GREX_NN_CMP(compare_lt)
GREX_NN_CMP(compare_ge)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_COMPARE_HPP
