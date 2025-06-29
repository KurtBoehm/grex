// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP

#include <cstddef> // IWYU pragma: keep

#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/macros/base.hpp"
#include "grex/backend/x86/macros/math.hpp"

#define GREX_FOREACH_FP_TYPE(MACRO, REGISTERBITS, ...) \
  MACRO(f, 32, GREX_ELEMENTS(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(f, 64, GREX_ELEMENTS(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__)

#define GREX_FOREACH_INT_TYPE_BASE(MACRO, KIND, REGISTERBITS, ...) \
  MACRO(KIND, 64, GREX_ELEMENTS(REGISTERBITS, 64) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 32, GREX_ELEMENTS(REGISTERBITS, 32) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 16, GREX_ELEMENTS(REGISTERBITS, 16) __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(KIND, 8, GREX_ELEMENTS(REGISTERBITS, 8) __VA_OPT__(, ) __VA_ARGS__)
#define GREX_FOREACH_UINT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_INT_TYPE_BASE(MACRO, u, REGISTERBITS, __VA_ARGS__)
#define GREX_FOREACH_SINT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_INT_TYPE_BASE(MACRO, i, REGISTERBITS, __VA_ARGS__)

#define GREX_FOREACH_INT_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_UINT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__) \
  GREX_FOREACH_SINT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__)

#define GREX_FOREACH_TYPE(MACRO, REGISTERBITS, ...) \
  GREX_FOREACH_FP_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__) \
  GREX_FOREACH_INT_TYPE(MACRO, REGISTERBITS __VA_OPT__(, ) __VA_ARGS__)

#define GREX_SUPER_UNARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> v) { \
    return {.lower = NAME(v.lower), .upper = NAME(v.upper)}; \
  }
#define GREX_SUPER_BINARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> a, TYPE<THalf> b) { \
    return {.lower = NAME(a.lower, b.lower), .upper = NAME(a.upper, b.upper)}; \
  }
#define GREX_SUPER_TERNARY(TYPE, NAME) \
  template<typename THalf> \
  inline TYPE<THalf> NAME(TYPE<THalf> a, TYPE<THalf> b, TYPE<THalf> c) { \
    return {.lower = NAME(a.lower, b.lower, c.lower), .upper = NAME(a.upper, b.upper, c.upper)}; \
  }
#define GREX_SUPERVECTOR_UNARY(NAME) GREX_SUPER_UNARY(SuperVector, NAME)
#define GREX_SUPERVECTOR_BINARY(NAME) GREX_SUPER_BINARY(SuperVector, NAME)
#define GREX_SUPERVECTOR_TERNARY(NAME) GREX_SUPER_TERNARY(SuperVector, NAME)
#define GREX_SUPERMASK_UNARY(NAME) GREX_SUPER_UNARY(SuperMask, NAME)
#define GREX_SUPERMASK_BINARY(NAME) GREX_SUPER_BINARY(SuperMask, NAME)
#define GREX_SUPERMASK_TERNARY(NAME) GREX_SUPER_TERNARY(SuperMask, NAME)

#define GREX_SUB_UNARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> v) { \
    return TYPE<T, tPart, tSize>{NAME(v.full)}; \
  }
#define GREX_SUB_BINARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> a, TYPE<T, tPart, tSize> b) { \
    return TYPE<T, tPart, tSize>{NAME(a.full, b.full)}; \
  }
#define GREX_SUB_TERNARY(TYPE, NAME) \
  template<Vectorizable T, std::size_t tPart, std::size_t tSize> \
  inline TYPE<T, tPart, tSize> NAME(TYPE<T, tPart, tSize> a, TYPE<T, tPart, tSize> b, \
                                    TYPE<T, tPart, tSize> c) { \
    return TYPE<T, tPart, tSize>{NAME(a.full, b.full, c.full)}; \
  }
#define GREX_SUBVECTOR_UNARY(NAME) GREX_SUB_UNARY(SubVector, NAME)
#define GREX_SUBVECTOR_BINARY(NAME) GREX_SUB_BINARY(SubVector, NAME)
#define GREX_SUBVECTOR_TERNARY(NAME) GREX_SUB_TERNARY(SubVector, NAME)
#define GREX_SUBMASK_UNARY(NAME) GREX_SUB_UNARY(SubMask, NAME)
#define GREX_SUBMASK_BINARY(NAME) GREX_SUB_BINARY(SubMask, NAME)
#define GREX_SUBMASK_TERNARY(NAME) GREX_SUB_TERNARY(SubMask, NAME)

#define GREX_VTYPE_16(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_MINSIZE(BITS)>
#define GREX_VTYPE_32(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_MINSIZE(BITS)>
#define GREX_VTYPE_64(BASE, KIND, BITS, SIZE) Sub##BASE<KIND##BITS, SIZE, GREX_MINSIZE(BITS)>
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
  GREX_CAT(GREX_VTYPE_, GREX_PARTBITS(BITS, SIZE))(Vector, KIND, BITS, SIZE)
#define GREX_MASK_TYPE(KIND, BITS, SIZE) \
  GREX_CAT(GREX_VTYPE_, GREX_PARTBITS(BITS, SIZE))(Mask, KIND, BITS, SIZE)

#define GREX_FOREACH_SUB(MACRO, ...) \
  MACRO(f, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 32, 2, 4 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, 4, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, 4, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, 2, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, 2, 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 8, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 8, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 4, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 4, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, 2, 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 8, 2, 16 __VA_OPT__(, ) __VA_ARGS__)

#if GREX_X86_64_LEVEL >= 4
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) \
  MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(256, _mm256 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(512, _mm512 __VA_OPT__(, ) __VA_ARGS__)
#elif GREX_X86_64_LEVEL >= 3
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) \
  MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(256, _mm256 __VA_OPT__(, ) __VA_ARGS__)
#else
#define GREX_FOREACH_X86_64_LEVEL(MACRO, ...) MACRO(128, _mm __VA_OPT__(, ) __VA_ARGS__)
#endif

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_FOR_EACH_HPP
