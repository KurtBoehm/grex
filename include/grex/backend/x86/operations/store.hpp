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
#include "grex/base/defs.hpp" // IWYU pragma: keep

namespace grex::backend {
// Define the casts
#define GREX_STORE_CAST_f(REGISTERBITS) dst
#define GREX_STORE_CAST_u(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)
#define GREX_STORE_CAST_i(REGISTERBITS) reinterpret_cast<__m##REGISTERBITS##i*>(dst)

#define GREX_STORE_BASE(NAME, INFIX, KIND, BITS, SIZE, KINDPREFIX, REGISTERBITS) \
  inline void NAME(KIND##BITS* dst, Vector<KIND##BITS, SIZE> src) { \
    BOOST_PP_CAT(KINDPREFIX##_##INFIX##_, GREX_SI_SUFFIX(KIND, BITS, REGISTERBITS))( \
      GREX_STORE_CAST_##KIND(REGISTERBITS), src.r); \
  }
#define GREX_STORE(...) \
  GREX_STORE_BASE(store, storeu, __VA_ARGS__) \
  GREX_STORE_BASE(store_aligned, store, __VA_ARGS__)
#define GREX_STORE_ALL(REGISTERBITS, KINDPREFIX) \
  GREX_FOREACH_TYPE(GREX_STORE, REGISTERBITS, KINDPREFIX, REGISTERBITS)

GREX_FOREACH_X86_64_LEVEL(GREX_STORE_ALL)
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_STORE_HPP
