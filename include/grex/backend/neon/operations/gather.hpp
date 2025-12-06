// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_GATHER_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_GATHER_HPP

#include <cstddef>

#include "grex/backend/neon/operations/extract.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/merge.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/set.hpp" // IWYU pragma: keep
#include "grex/backend/neon/operations/split.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

// shared definitions
#include "grex/backend/shared/operations/gather.hpp" // IWYU pragma: export

namespace grex::backend {
#define GREX_GTHR_MOD_64 "x"
#define GREX_GTHR_MOD_32 "w"
#define GREX_GTHR_MOD_16 "w"
#define GREX_GTHR_MOD_8 "w"

#define GREX_GTHR_LSL_64 "lsl"
#define GREX_GTHR_LSL_32 "uxtw"
#define GREX_GTHR_LSL_16 "uxtw"
#define GREX_GTHR_LSL_8 "uxtw"

#define GREX_GTHR_LDR_f64(IDX, IDXBITS) \
  const auto i##IDX = extract(idxs, index_tag<IDX>); \
  float64x2_t v##IDX; \
  asm volatile("ldr %d0, [%x1, %" GREX_GTHR_MOD_##IDXBITS "2, " GREX_GTHR_LSL_##IDXBITS " #3]" \
               : "=w"(v##IDX) \
               : "r"(data.data()), "r"(i##IDX) \
               : "memory")

#define GREX_GTHR_f64(IDXBITS) \
  template<std::size_t tExtent> \
  inline VectorFor<f64, 4> gather(std::span<const f64, tExtent> data, \
                                  VectorFor<u##IDXBITS, 4> idxs) { \
    GREX_GTHR_LDR_f64(0, IDXBITS); \
    GREX_GTHR_LDR_f64(1, IDXBITS); \
    GREX_GTHR_LDR_f64(2, IDXBITS); \
    GREX_GTHR_LDR_f64(3, IDXBITS); \
\
    return VectorFor<f64, 4>{.lower = {.r = vzip1q_f64(v0, v1)}, \
                             .upper = {.r = vzip1q_f64(v2, v3)}}; \
  }

GREX_GTHR_f64(64) // NOLINT
  GREX_GTHR_f64(32) // NOLINT
  GREX_GTHR_f64(16) // NOLINT
  GREX_GTHR_f64(8) // NOLINT
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_GATHER_HPP
