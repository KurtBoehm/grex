// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP

#include <immintrin.h>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
inline Scalar<f32> extract_single(f32x4 v) {
  return {.value = _mm_cvtss_f32(v.r)};
}
inline Scalar<f64> extract_single(f64x2 v) {
  return {.value = _mm_cvtsd_f64(v.r)};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_EXTRACT_SINGLE_HPP
