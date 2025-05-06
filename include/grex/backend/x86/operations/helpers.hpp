// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HELPERS_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HELPERS_HPP

#define GREX_IDENTITY(X) X
#define GREX_APPLY(MACRO, ...) MACRO(__VA_ARGS__)

#define GREX_FP_SUFFIX_f32 ps
#define GREX_FP_SUFFIX_f64 pd
#define GREX_FP_SUFFIX(TYPE) GREX_FP_SUFFIX_##TYPE

#define GREX_REGISTERBITS_PREFIX_128 mm
#define GREX_REGISTERBITS_PREFIX_256 mm256
#define GREX_REGISTERBITS_PREFIX_512 mm512
#define GREX_REGISTERBITS_PREFIX(REGISTERBITS) GREX_REGISTERBITS_PREFIX_##REGISTERBITS

#define GREX_DEFINE_SI_OPERATION(MACRO, REGISTERBITS, ...) \
  MACRO(u8, (REGISTERBITS) / 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i8, (REGISTERBITS) / 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u16, (REGISTERBITS) / 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i16, (REGISTERBITS) / 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u32, (REGISTERBITS) / 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i32, (REGISTERBITS) / 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u64, (REGISTERBITS) / 64 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i64, (REGISTERBITS) / 64 __VA_OPT__(, ) __VA_ARGS__)

#define GREX_DEFINE_SI_OPERATION_EXT(MACRO, REGISTERBITS, ...) \
  MACRO(u, 8, (REGISTERBITS) / 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 8, (REGISTERBITS) / 8 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 16, (REGISTERBITS) / 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 16, (REGISTERBITS) / 16 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 32, (REGISTERBITS) / 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 32, (REGISTERBITS) / 32 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(u, 64, (REGISTERBITS) / 64 __VA_OPT__(, ) __VA_ARGS__) \
  MACRO(i, 64, (REGISTERBITS) / 64 __VA_OPT__(, ) __VA_ARGS__)

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HELPERS_HPP
