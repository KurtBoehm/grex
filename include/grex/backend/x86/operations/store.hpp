// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP

#include <immintrin.h>

#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/types.hpp"

namespace grex::backend {
// Define the casts
#define GREX_STORE_CAST_f(REGISTERBITS)
#define GREX_STORE_CAST_u(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>
#define GREX_STORE_CAST_i(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>

#define GREX_STORE_BASE(NAME, ELEMENT, SIZE, MACRO, CAST) \
  inline void NAME(ELEMENT* dst, Vector<ELEMENT, SIZE> src) { \
    MACRO(CAST(dst), src.r); \
  }
#define GREX_STORE_IMPL(ELEMENT, SIZE, KINDPREFIX, SUFFIX, CAST) \
  GREX_STORE_BASE(store, ELEMENT, SIZE, _##KINDPREFIX##_storeu_##SUFFIX, CAST) \
  GREX_STORE_BASE(store_aligned, ELEMENT, SIZE, _##KINDPREFIX##_store_##SUFFIX, CAST)
#define GREX_STORE(KIND, BITS, SIZE, KINDPREFIX, REGISTERBITS) \
  GREX_APPLY(GREX_STORE_IMPL, KIND##BITS, SIZE, KINDPREFIX, \
             GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS), GREX_STORE_CAST_##KIND(REGISTERBITS))
#define GREX_STORE_ALL(REGISTERBITS, KINDPREFIX) \
  GREX_FOREACH_TYPE(GREX_STORE, REGISTERBITS, KINDPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_STORE_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
