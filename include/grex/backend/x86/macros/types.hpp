// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_TYPES_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_TYPES_HPP

#include "grex/backend/macros/base.hpp"
#include "grex/backend/macros/math.hpp"

#define GREX_VTYPE_16(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_DIVIDE(128, BITS)>
#define GREX_VTYPE_32(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_DIVIDE(128, BITS)>
#define GREX_VTYPE_64(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_DIVIDE(128, BITS)>
#define GREX_VTYPE_128(BASE, KIND, BITS, SIZE) BASE<KIND##BITS, SIZE>
#if GREX_X86_64_LEVEL >= 3
#define GREX_VTYPE_256(BASE, KIND, BITS, SIZE) BASE<KIND##BITS, SIZE>
#else
#define GREX_VTYPE_256(BASE, KIND, BITS, SIZE) BASE##For<KIND##BITS, SIZE>
#endif
#if GREX_X86_64_LEVEL >= 4
#define GREX_VTYPE_512(BASE, KIND, BITS, SIZE) BASE<KIND##BITS, SIZE>
#else
#define GREX_VTYPE_512(BASE, KIND, BITS, SIZE) BASE##For<KIND##BITS, SIZE>
#endif
#define GREX_VTYPE_1024(BASE, KIND, BITS, SIZE) BASE##For<KIND##BITS, SIZE>
#define GREX_VECTOR_TYPE(KIND, BITS, SIZE) \
  GREX_CAT(GREX_VTYPE_, GREX_MULTIPLY(BITS, SIZE))(Vector, KIND, BITS, SIZE)
#define GREX_MASK_TYPE(KIND, BITS, SIZE) \
  GREX_CAT(GREX_VTYPE_, GREX_MULTIPLY(BITS, SIZE))(Mask, KIND, BITS, SIZE)

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_TYPES_HPP
