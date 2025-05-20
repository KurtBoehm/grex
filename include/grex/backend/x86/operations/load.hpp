// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP

#include "thesauros/types/value-tag.hpp"

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// sadly, a cast to “const __m128i*” (or a larger register type) is required for integer vectors
#define GREX_LOAD_CAST_f(REGISTERBITS, X) X
#define GREX_LOAD_CAST_i(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)
#define GREX_LOAD_CAST_u(REGISTERBITS, X) reinterpret_cast<const __m##REGISTERBITS##i*>(X)

#define GREX_LOAD(KIND, BITS, SIZE, BITPREFIX, REGISTERBITS, NAME, OP) \
  inline Vector<KIND##BITS, SIZE> NAME(const KIND##BITS* data, thes::IndexTag<SIZE>) { \
    return {.r = GREX_CAT(BITPREFIX##_##OP##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
              GREX_LOAD_CAST_##KIND(REGISTERBITS, data))}; \
  }

#define GREX_LOAD_ALL(REGISTERBITS, BITPREFIX) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load, loadu) \
  GREX_FOREACH_TYPE(GREX_LOAD, REGISTERBITS, BITPREFIX, REGISTERBITS, load_aligned, load)
GREX_FOREACH_X86_64_LEVEL(GREX_LOAD_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOAD_HPP
