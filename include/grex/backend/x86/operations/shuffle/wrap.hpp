// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_WRAP_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_WRAP_HPP

#include <array>

#include "thesauros/types/value-tag.hpp"

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/operations/shuffle/128.hpp" // IWYU pragma: keep
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
#define GREX_SHUFFLE(KIND, BITS, SIZE, REGISTERBITS) \
  inline Vector<KIND##BITS, SIZE> shuffle( \
    Vector<KIND##BITS, SIZE> v, thes::TypedValueTag<std::array<ShuffleIndex, SIZE>> auto idxs) { \
    const auto shuffled = \
      backend::shuffle##SIZE##x##BITS(GREX_KINDCAST(KIND, i, BITS, REGISTERBITS, v.r), idxs); \
    return {.r = GREX_KINDCAST(i, KIND, BITS, REGISTERBITS, shuffled)}; \
  }

GREX_SHUFFLE(f, 64, 2, 128)
GREX_SHUFFLE(i, 64, 2, 128)
GREX_SHUFFLE(u, 64, 2, 128)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_WRAP_HPP
