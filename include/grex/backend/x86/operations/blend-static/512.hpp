// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_512_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_512_HPP

#include "grex/backend/x86/instruction-sets.hpp"

// TODO This only includes the basic operations and could be extended using the vinserti family,
//      the vshufi family, and others
#if GREX_X86_64_LEVEL >= 4
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/operations/blend-static/base.hpp"
#include "grex/backend/x86/operations/blend-static/shared.hpp"

namespace grex::backend {
template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size == 64))
struct BlenderTrait<tBls> {
  using Type = CheapestType<tBls, BlenderConstant, BlenderVariable>;
};
} // namespace grex::backend
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_STATIC_512_HPP
