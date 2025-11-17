// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_512_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_512_HPP

#include "grex/backend/x86/instruction-sets.hpp"

// TODO This only includes the basic operations and could be extended using the vinserti family,
//      the vshufi family, and others
#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/backend/x86/operations/blend-zero-static/shared.hpp"

namespace grex::backend {
template<AnyBlendZeros auto tBzs>
requires((tBzs.value_size * tBzs.size == 64))
struct ZeroBlenderTrait<tBzs> {
  using Type = CheapestType<tBzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderAnd>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_512_HPP
