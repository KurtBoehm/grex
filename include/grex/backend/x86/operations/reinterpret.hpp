// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP

#include <cstddef>

#include <immintrin.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/x86/macros/for-each.hpp"
#include "grex/backend/x86/macros/intrinsics.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

namespace grex::backend {
#define GREX_REINTERPRET_BASE(DSTKIND, DSTBITS, DSTSIZE, SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS) \
  inline NativeVector<DSTKIND##DSTBITS, DSTSIZE> reinterpret( \
    NativeVector<SRCKIND##SRCBITS, SRCSIZE> v, TypeTag<DSTKIND##DSTBITS>) { \
    return NativeVector<DSTKIND##DSTBITS, DSTSIZE>{ \
      .r = GREX_KINDCAST_EXT(DSTKIND, DSTBITS, SRCKIND, SRCBITS, REGISTERBITS, v.r)}; \
  }
#define GREX_REINTERPRET(SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS) \
  GREX_FOREACH_TYPE_R(GREX_REINTERPRET_BASE, REGISTERBITS, SRCKIND, SRCBITS, SRCSIZE, REGISTERBITS)
#define GREX_REINTERPRET_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_REINTERPRET, REGISTERBITS, REGISTERBITS)
GREX_FOREACH_X86_64_LEVEL(GREX_REINTERPRET_ALL)

// f16 shares its register with u16, so reinterpreting from/to f16 is reinterpreting from/to u16.
template<Vectorizable TDst, std::size_t tSize>
inline auto reinterpret(NativeVector<f16, tSize> v, TypeTag<TDst> tag) {
  return reinterpret(NativeVector<u16, tSize>{.r = v.r}, tag);
}
template<Vectorizable TSrc, std::size_t tSize>
inline NativeVector<f16, tSize * sizeof(TSrc) / 2> reinterpret(NativeVector<TSrc, tSize> v,
                                                               TypeTag<f16> /*tag*/) {
  return NativeVector<f16, tSize * sizeof(TSrc) / 2>{.r = reinterpret(v, type_tag<u16>).r};
}
template<std::size_t tSize>
inline NativeVector<f16, tSize> reinterpret(NativeVector<f16, tSize> v, TypeTag<f16> /*tag*/) {
  return v;
}

template<Vectorizable TDst, Vectorizable TSrc, std::size_t tSize>
inline SubVector<TDst, tSize * sizeof(TSrc) / sizeof(TDst)> reinterpret(SubVector<TSrc, tSize> v,
                                                                        TypeTag<TDst> tag) {
  using Dst = SubVector<TDst, tSize * sizeof(TSrc) / sizeof(TDst)>;
  return Dst{reinterpret(v.full, tag)};
}
template<Vectorizable TDst, typename THalf>
inline VectorFor<TDst, 2 * THalf::size * sizeof(typename THalf::Value) / sizeof(TDst)>
reinterpret(SuperVector<THalf> v, TypeTag<TDst> tag) {
  return {.lower = reinterpret(v.lower, tag), .upper = reinterpret(v.upper, tag)};
}

template<Vectorizable TDst, typename TSrc>
inline auto as(TSrc src) {
  return reinterpret(src, type_tag<TDst>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_REINTERPRET_HPP
