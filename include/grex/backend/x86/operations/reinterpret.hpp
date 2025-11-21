// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_REINTERPRET_BASE(DSTKIND, DSTBITS, DSTSIZE, SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS) \
  inline Vector<DSTKIND##DSTBITS, DSTSIZE> reinterpret(Vector<SRCKIND##SRCBITS, SRCSIZE> v, \
                                                       TypeTag<DSTKIND##DSTBITS>) { \
    return Vector<DSTKIND##DSTBITS, DSTSIZE>{ \
      .r = GREX_KINDCAST_EXT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, REGISTERBITS, v.r)}; \
  }
#define GREX_REINTERPRET(SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS) \
  GREX_FOREACH_TYPE_R(GREX_REINTERPRET_BASE, REGISTERBITS, SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS)
#define GREX_REINTERPRET_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_REINTERPRET, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_REINTERPRET_ALL)

template<Vectorizable TDst, Vectorizable TSrc, std::size_t tPart, std::size_t tSize>
inline SubVector<TDst, tPart, tSize> reinterpret(SubVector<TSrc, tPart, tSize> v,
                                                 TypeTag<TDst> tag) {
  return SubVector<TDst, tPart, tSize>{reinterpret(v.full, tag)};
}
template<Vectorizable TDst, typename THalf>
inline VectorFor<TDst, THalf::size> reinterpret(SuperVector<THalf> v, TypeTag<TDst> tag) {
  return {.lower = reinterpret(v.lower, tag), .upper = reinterpret(v.upper, tag)};
}

template<Vectorizable TDst, typename TSrc>
inline auto reinterpret(TSrc src) {
  return reinterpret(src, type_tag<TDst>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP
