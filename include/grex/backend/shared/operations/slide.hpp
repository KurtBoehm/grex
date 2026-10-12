// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SLIDE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SLIDE_HPP

#include <cstddef>

#include "grex/backend/active/operations/merge.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/neon/sizes.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<std::size_t Shift, AnySubNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard]] GREX_ALWAYS_INLINE inline V concat_slide_down(V lo, V hi) noexcept {
  // The merged vector may itself be sub-native, hence `native()`.
  return V{slide<shift_down, Shift>(merge(lo, hi)).native()};
}
template<std::size_t Shift, AnySuperNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard]] GREX_ALWAYS_INLINE inline V concat_slide_down(V lo, V hi) noexcept {
  static constexpr std::size_t half_size = size_of<V> / 2;
  if constexpr (Shift <= half_size) {
    return V{
      .lower = concat_slide_down<Shift>(lo.lower, lo.upper),
      .upper = concat_slide_down<Shift>(lo.upper, hi.lower),
    };
  } else {
    return V{
      .lower = concat_slide_down<Shift - half_size>(lo.upper, hi.lower),
      .upper = concat_slide_down<Shift - half_size>(hi.lower, hi.upper),
    };
  }
}

/**
 * The half the elements move into is a `concat_slide_down` of both halves if `Shift` is at most
 * half the size, and a slide of the other half otherwise; the half they leave becomes a slide of
 * itself or zero, respectively.
 */
template<ShiftDirection Dir, std::size_t Shift, AnySuperNativeVector V>
requires(Shift <= size_of<V>)
[[nodiscard]] GREX_ALWAYS_INLINE inline V slide(V v) noexcept {
  static constexpr bool down = Dir == shift_down;
  static constexpr std::size_t half_size = size_of<V> / 2;
  if constexpr (down && Shift <= half_size) {
    return V{
      .lower = concat_slide_down<Shift>(v.lower, v.upper),
      .upper = slide<Dir, Shift>(v.upper),
    };
  } else if constexpr (down) {
    return V{
      .lower = slide<Dir, Shift - half_size>(v.upper),
      .upper = {},
    };
  } else if constexpr (Shift <= half_size) {
    return V{
      .lower = slide<Dir, Shift>(v.lower),
      .upper = concat_slide_down<half_size - Shift>(v.lower, v.upper),
    };
  } else {
    return V{
      .lower = {},
      .upper = slide<Dir, Shift - half_size>(v.lower),
    };
  }
}

/** Rotation of sub- and super-native vectors as `concat_slide_down` of the vector with itself. */
template<std::size_t Shift, typename V>
requires((AnySubNativeVector<V> || AnySuperNativeVector<V>) && Shift <= size_of<V>)
[[nodiscard]] GREX_ALWAYS_INLINE inline V rotate_down(V v) noexcept {
  return concat_slide_down<Shift>(v, v);
}

/**
 * Super-native vectors: like `concat_slide_down`, both halves of the result are independent
 * `concat_slide_down`s of half-sized vectors with the same shift.
 */
template<std::size_t ValueBytes, std::size_t N, std::size_t Shift>
requires(Shift <= N && ValueBytes * N > register_bytes.back())
consteval Cost concat_slide_down_cost() noexcept {
  static constexpr std::size_t half = N / 2;
  const Cost c = concat_slide_down_cost<ValueBytes, half, (Shift <= half) ? Shift : Shift - half>();
  return {.inv_throughput = 2 * c.inv_throughput, .latency = c.latency};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SLIDE_HPP
