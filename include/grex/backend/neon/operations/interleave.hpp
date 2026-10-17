// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INTERLEAVE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INTERLEAVE_HPP

#include <array>
#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/expand64.hpp"
#include "grex/backend/neon/operations/merge.hpp"
#include "grex/backend/neon/operations/reinterpret.hpp"
#include "grex/base.hpp"

namespace grex::backend {
//==================================================================================================
// Separate functions → Overloads
//==================================================================================================

#define GREX_ZIP(KIND, BITS, SIZE, NAME, ISUFFIX) \
  [[nodiscard, GREX_ALWAYS_INLINE]] inline GREX_REGISTER(KIND, BITS, SIZE) \
    NAME(GREX_REGISTER(KIND, BITS, SIZE) r0, GREX_REGISTER(KIND, BITS, SIZE) r1) noexcept { \
    return GREX_ISUFFIXED(v##NAME##ISUFFIX, KIND, BITS)(r0, r1); \
  }
#define GREX_ZIP1(KIND, BITS, SIZE, SUFFIX) \
  GREX_ZIP(KIND, BITS, SIZE, zip1, SUFFIX) \
  GREX_ZIP(KIND, BITS, SIZE, zip2, SUFFIX) \
  GREX_ZIP(KIND, BITS, SIZE, uzp1, SUFFIX) \
  GREX_ZIP(KIND, BITS, SIZE, uzp2, SUFFIX)
GREX_ZIP1(u, 8, 8, )
GREX_ZIP1(u, 16, 4, )
GREX_ZIP1(u, 32, 2, )
GREX_FOREACH_UINT_TYPE(GREX_ZIP1, 128, q)
#undef GREX_ZIP1

//==================================================================================================
// Interleave
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// Two-argument interleave: `v0[i]` become the lower, `v1[i]` the upper half of `dst[i]`
//--------------------------------------------------------------------------------------------------

template<UnsignedIntVectorizable T, std::size_t N>
requires(is_subnative<UnsignedInt<sizeof(T) * 2>, N>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline SubVector<UnsignedInt<sizeof(T) * 2>, N>
interleave(SubVector<T, N> v0, SubVector<T, N> v1) noexcept {
  using Dst = UnsignedInt<sizeof(T) * 2>;
  return VectorFor<Dst, N>{as<Dst>(expand64(zip1(low64(v0.registr()), low64(v1.registr()))))};
}

template<UnsignedIntVectorizable T, std::size_t N>
requires(is_native<UnsignedInt<sizeof(T) * 2>, N>)
[[nodiscard, GREX_ALWAYS_INLINE]] inline NativeVector<UnsignedInt<sizeof(T) * 2>, N>
interleave(SubVector<T, N> v0, SubVector<T, N> v1) noexcept {
  return {.r = as<UnsignedInt<sizeof(T) * 2>>(zip1(v0.registr(), v1.registr()))};
}

template<UnsignedIntVectorizable T, std::size_t N>
[[nodiscard, GREX_ALWAYS_INLINE]] inline VectorFor<UnsignedInt<sizeof(T) * 2>, N>
interleave(NativeVector<T, N> v0, NativeVector<T, N> v1) noexcept {
  using Dst = UnsignedInt<sizeof(T) * 2>;
  const NativeVector<Dst, N / 2> lo{.r = as<Dst>(zip1(v0.r, v1.r))};
  const NativeVector<Dst, N / 2> hi{.r = as<Dst>(zip2(v0.r, v1.r))};
  return VectorFor<Dst, N>{.lower = lo, .upper = hi};
}

template<AnySuperNativeVector V>
[[nodiscard, GREX_ALWAYS_INLINE]] inline VectorFor<UnsignedInt<2 * sizeof(ValueOf<V>)>, size_of<V>>
interleave(V v0, V v1) noexcept {
  using Dst = VectorFor<UnsignedInt<2 * sizeof(ValueOf<V>)>, size_of<V>>;
  return Dst{.lower = interleave(v0.lower, v1.lower), .upper = interleave(v0.upper, v1.upper)};
}

//--------------------------------------------------------------------------------------------------
// Four-argument interleave: `v0[i]` through `v3[i]` are merged into `dst[i]`
//--------------------------------------------------------------------------------------------------

template<AnyVector V>
[[nodiscard, GREX_ALWAYS_INLINE]] inline VectorFor<UnsignedInt<4 * sizeof(ValueOf<V>)>, size_of<V>>
interleave(V v0, V v1, V v2, V v3) noexcept {
  return interleave(interleave(v0, v1), interleave(v2, v3));
}

//--------------------------------------------------------------------------------------------------
// Eight-argument interleave: `v0[i]` through `v7[i]` are merged into `dst[i]`
//--------------------------------------------------------------------------------------------------

template<AnyVector V>
[[nodiscard, GREX_ALWAYS_INLINE]] inline VectorFor<UnsignedInt<8 * sizeof(ValueOf<V>)>, size_of<V>>
interleave(V v0, V v1, V v2, V v3, V v4, V v5, V v6, V v7) noexcept {
  return interleave(interleave(v0, v1, v2, v3), interleave(v4, v5, v6, v7));
}

//==================================================================================================
// De-Interleave
//==================================================================================================

//--------------------------------------------------------------------------------------------------
// De-interleave splitting in two
//--------------------------------------------------------------------------------------------------

template<UnsignedIntVectorizable Dst, UnsignedIntVectorizable Src, std::size_t N>
requires(sizeof(Dst) * 2 == sizeof(Src))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<SubVector<Dst, N>, 2>
deinterleave(SubVector<Src, N> v) noexcept {
  const auto v0 = low64(as<Dst>(v.registr()));
  return {SubVector<Dst, N>{expand64(uzp1(v0, v0))}, SubVector<Dst, N>{expand64(uzp2(v0, v0))}};
}

template<UnsignedIntVectorizable Dst, UnsignedIntVectorizable Src, std::size_t N>
requires(sizeof(Dst) * 2 == sizeof(Src))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<SubVector<Dst, N>, 2>
deinterleave(NativeVector<Src, N> v) noexcept {
  const auto v0 = as<Dst>(v.registr());
  return {SubVector<Dst, N>{uzp1(v0, v0)}, SubVector<Dst, N>{uzp2(v0, v0)}};
}

template<UnsignedIntVectorizable Dst, UnsignedIntVectorizable Src, std::size_t N>
requires(sizeof(Dst) * 2 == sizeof(Src))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<NativeVector<Dst, 2 * N>, 2>
deinterleave(SuperVector<NativeVector<Src, N>> v) noexcept {
  const auto v0 = as<Dst>(v.lower.r);
  const auto v1 = as<Dst>(v.upper.r);
  return {{{.r = uzp1(v0, v1)}, {.r = uzp2(v0, v1)}}};
}

template<UnsignedIntVectorizable Dst, AnySuperNativeVector V>
requires(sizeof(Dst) * 2 == sizeof(ValueOf<V>))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<VectorFor<Dst, size_of<V>>, 2>
deinterleave(V v) noexcept {
  const auto [v0a, v0b] = deinterleave<Dst>(v.lower);
  const auto [v1a, v1b] = deinterleave<Dst>(v.upper);
  return {merge(v0a, v1a), merge(v0b, v1b)};
}

//--------------------------------------------------------------------------------------------------
// De-interleave splitting in four
//--------------------------------------------------------------------------------------------------

template<UnsignedIntVectorizable Dst, AnyVector V>
requires(sizeof(Dst) * 4 == sizeof(ValueOf<V>))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<VectorFor<Dst, size_of<V>>, 4>
deinterleave(V v) noexcept {
  const auto [two0, two1] = deinterleave<UnsignedInt<sizeof(Dst) * 2>>(v);
  const auto [v0, v1] = deinterleave<Dst>(two0);
  const auto [v2, v3] = deinterleave<Dst>(two1);
  return {v0, v1, v2, v3};
}

//--------------------------------------------------------------------------------------------------
// De-interleave splitting in eight
//--------------------------------------------------------------------------------------------------

template<UnsignedIntVectorizable Dst, AnyVector V>
requires(sizeof(Dst) * 8 == sizeof(ValueOf<V>))
[[nodiscard, GREX_ALWAYS_INLINE]] inline std::array<VectorFor<Dst, size_of<V>>, 8>
deinterleave(V v) noexcept {
  const auto [four0, four1, four2, four3] = deinterleave<UnsignedInt<sizeof(Dst) * 2>>(v);
  const auto [v0, v1] = deinterleave<Dst>(four0);
  const auto [v2, v3] = deinterleave<Dst>(four1);
  const auto [v4, v5] = deinterleave<Dst>(four2);
  const auto [v6, v7] = deinterleave<Dst>(four3);
  return {v0, v1, v2, v3, v4, v5, v6, v7};
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_INTERLEAVE_HPP
