// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SLIDE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SLIDE_HPP

#include <cstddef>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/math.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand64.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/neon/sizes.hpp"
#include "grex/backend/neon/types.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
//==================================================================================================
// Costs
//==================================================================================================

namespace ext {
/**
 * One `EXT`, which has a latency of 2 and is present on each of the four SIMD execution units on
 * each performance core on Apple Silicon, according to the Apple Silicon CPU Optimization Guide.
 * A `Shift` of `0` or `N` just returns one input vector and costs nothing.
 */
template<std::size_t N, std::size_t Shift>
[[nodiscard]] consteval Cost cost() noexcept {
  return (Shift == 0 || Shift == N) ? Cost{.inv_throughput = 0, .latency = 0}
                                    : Cost{.inv_throughput = 0.25, .latency = 2};
}
} // namespace ext

/** See `ext::cost`. */
template<std::size_t ValueBytes, std::size_t N, std::size_t Shift>
requires(Shift <= N && ValueBytes * N <= register_bytes.back())
[[nodiscard]] consteval Cost concat_slide_down_cost() noexcept {
  return ext::cost<N, Shift>();
}
/** See `ext::cost`. */
template<std::size_t ValueBytes, std::size_t N, std::size_t Shift>
requires(Shift <= N)
[[nodiscard]] consteval Cost rotate_down_cost() noexcept {
  return ext::cost<N, Shift>();
}
/**
 * See `ext::cost`. The zero register, if needed, is usually hoisted out of loops and ignored here.
 */
template<std::size_t ValueBytes, std::size_t N, ShiftDirection Dir, std::size_t Shift>
requires(Shift <= N)
[[nodiscard]] consteval Cost slide_cost() noexcept {
  return ext::cost<N, Shift>();
}

//==================================================================================================
// Native
//==================================================================================================

namespace ext {
#define GREX_EXT_I(KIND, BITS, SIZE, REGKIND) \
  template<std::size_t Shift> \
  requires(Shift <= SIZE) \
  [[nodiscard, GREX_ALWAYS_INLINE]] inline KIND##BITS##x##SIZE ext(KIND##BITS##x##SIZE lo, \
                                                                   KIND##BITS##x##SIZE hi) { \
    if constexpr (Shift == 0) { \
      return lo; \
    } else if constexpr (Shift == SIZE) { \
      return hi; \
    } else { \
      return {.r = GREX_ISUFFIXED(vextq, REGKIND, BITS)(lo.r, hi.r, Shift)}; \
    } \
  }
#define GREX_EXT(KIND, BITS, SIZE) GREX_EXT_I(KIND, BITS, SIZE, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_TYPE_EXT(GREX_EXT, 128)
} // namespace ext

template<std::size_t Shift, Vectorizable T, std::size_t N>
requires(Shift <= N)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, N>
concat_slide_down(NativeVector<T, N> lo, NativeVector<T, N> hi) noexcept {
  return ext::ext<Shift>(lo, hi);
}
template<std::size_t Shift, Vectorizable T, std::size_t N>
requires(Shift <= N)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, N>
rotate_down(NativeVector<T, N> v) noexcept {
  return ext::ext<Shift>(v, v);
}

template<ShiftDirection Dir, std::size_t Shift, Vectorizable T, std::size_t N>
requires(Shift <= N)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<T, N> slide(NativeVector<T, N> v) noexcept {
  if constexpr (Dir == shift_down) {
    return ext::ext<Shift>(v, zeros<NativeVector<T, N>>());
  } else {
    return ext::ext<N - Shift>(zeros<NativeVector<T, N>>(), v);
  }
}

//==================================================================================================
// Sub-native
//==================================================================================================

#define GREX_EXT_SUB_II(KIND, BITS, PART, SIZE, PARTBITS, REGKIND) \
  template<ShiftDirection Dir, std::size_t Shift> \
  requires(Shift <= PART) \
  [[nodiscard, GREX_ALWAYS_INLINE]] inline SubVector<KIND##BITS, PART> slide( \
    SubVector<KIND##BITS, PART> v) noexcept { \
    if constexpr (Shift == 0) { \
      return v; \
    } else if constexpr (Shift == PART) { \
      return SubVector<KIND##BITS, PART>{GREX_ISUFFIXED(vdupq_n, REGKIND, BITS)(0)}; \
    } else { \
      const auto low = vget_low_u##PARTBITS(as<u##PARTBITS>(v.full.r)); \
      const auto sh = [&] { \
        if constexpr (Dir == shift_down) { \
          return vshr_n_u##PARTBITS(low, BITS * Shift); \
        } else { \
          return vshl_n_u##PARTBITS(low, BITS * Shift); \
        } \
      }(); \
      return SubVector<KIND##BITS, PART>{as<KIND##BITS>(expand64(sh))}; \
    } \
  }

#define GREX_EXT_SUB_I(KIND, BITS, PART, SIZE, PARTBITS, REGKIND) \
  GREX_EXT_SUB_II(KIND, BITS, PART, SIZE, PARTBITS, REGKIND)
#define GREX_EXT_SUB(KIND, BITS, PART, SIZE) \
  GREX_EXT_SUB_I(KIND, BITS, PART, SIZE, GREX_MULTIPLY(BITS, PART), GREX_REGKIND(KIND, BITS))
GREX_FOREACH_SUB_EXT(GREX_EXT_SUB)

#define GREX_EXT_SUB_CSD64(KIND, BITS, PART, REGKIND) \
  template<std::size_t Shift> \
  requires(Shift <= PART) \
  [[nodiscard, GREX_ALWAYS_INLINE]] inline SubVector<KIND##BITS, PART> concat_slide_down( \
    SubVector<KIND##BITS, PART> lo, SubVector<KIND##BITS, PART> hi) noexcept { \
    if constexpr (Shift == 0) { \
      return lo; \
    } else if constexpr (Shift == PART) { \
      return hi; \
    } else { \
      const auto lo64 = GREX_ISUFFIXED(vget_low, REGKIND, BITS)(lo.full.r); \
      const auto hi64 = GREX_ISUFFIXED(vget_low, REGKIND, BITS)(hi.full.r); \
      const auto slid = GREX_ISUFFIXED(vext, REGKIND, BITS)(lo64, hi64, Shift); \
      return SubVector<KIND##BITS, PART>{expand64(slid)}; \
    } \
  }
GREX_EXT_SUB_CSD64(f, 32, 2, f)
GREX_EXT_SUB_CSD64(i, 32, 2, i)
GREX_EXT_SUB_CSD64(u, 32, 2, u)
GREX_EXT_SUB_CSD64(f, 16, 4, u)
GREX_EXT_SUB_CSD64(i, 16, 4, i)
GREX_EXT_SUB_CSD64(u, 16, 4, u)
GREX_EXT_SUB_CSD64(i, 8, 8, i)
GREX_EXT_SUB_CSD64(u, 8, 8, u)
} // namespace grex::backend

#include "grex/backend/shared/operations/slide.hpp" // IWYU pragma: keep

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SLIDE_HPP
