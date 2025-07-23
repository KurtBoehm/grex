// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP

#include <cstddef>
#include <optional>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/bitwise.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

// TODO There are no optimized implementations at level 1, as those would require a lot of work
//      for very old architectures.

namespace grex::backend {
struct ZeroBlenderNoop : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable(BlendZeros<tValueBytes, tSize> bzs) {
    return static_apply<tSize>([&]<std::size_t... tIdxs>() {
      return (... && (bzs[tIdxs] == keep_bz || bzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    static_assert(is_applicable(TTag::value));
    return vec;
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};
struct ZeroBlenderZero : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable(BlendZeros<tValueBytes, tSize> bzs) {
    return static_apply<tSize>([&]<std::size_t... tIdxs>() {
      return (... && (bzs[tIdxs] == zero_bz || bzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec /*vec*/, TTag /*tag*/) {
    static_assert(is_applicable(TTag::value));
    return zeros(type_tag<TVec>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 1};
  }
};

struct ZeroBlenderMovq : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable(BlendZeros<tValueBytes, tSize> bzs) {
    const std::optional<BlendZeros<8, 2>> obz64 = convert<8>(bzs);
    if (!obz64.has_value()) {
      return false;
    }
    const BlendZeros<8, 2> bz64 = obz64.value();
    return (bz64[0] == keep_bz || bz64[0] == any_bz) && (bz64[1] == zero_bz || bz64[1] == any_bz);
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    static_assert(is_applicable(TTag::value));
    return reinterpret(i64x2{_mm_move_epi64(reinterpret(vec, type_tag<i64>).r)},
                       type_tag<typename TVec::Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

// TODO insertps is slightly more efficient than blendps on AMD,
//      but slightly less efficient on Intel
struct ZeroBlenderInsert32 : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable([[maybe_unused]] BlendZeros<tValueBytes, tSize> bzs) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(bzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<4>(TTag::value).value();

    const f32x4 fvec = reinterpret(vec, type_tag<f32>);
    static constexpr int imm8 = static_apply<4>([]<std::size_t... tIdxs>() {
      return (0 + ... + (int(bzs[tIdxs] != BlendZero::keep) << tIdxs));
    });
    return reinterpret(f32x4{_mm_insert_ps(fvec.r, fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

struct ZeroBlenderBlend32 : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable([[maybe_unused]] BlendZeros<tValueBytes, tSize> bzs) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(bzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<4>(TTag::value).value();

    const f32x4 fvec = reinterpret(vec, type_tag<f32>);
    static constexpr int imm8 = static_apply<4>([]<std::size_t... tIdxs>() {
      return (0 + ... + (int(bzs[tIdxs] == BlendZero::keep) << tIdxs));
    });
    return reinterpret(f32x4{_mm_blend_ps(_mm_setzero_ps(), fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

struct ZeroBlenderBlend16 : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable([[maybe_unused]] BlendZeros<tValueBytes, tSize> bzs) {
#if GREX_X86_64_LEVEL >= 2
    return convert<2>(bzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<2>(TTag::value).value();

    const i16x8 fvec = reinterpret(vec, type_tag<i16>);
    static constexpr int imm8 = static_apply<8>([]<std::size_t... tIdxs>() {
      return (0 + ... + (int(bzs[tIdxs] == BlendZero::keep) << tIdxs));
    });
    return reinterpret(i16x8{_mm_blend_epi16(_mm_setzero_si128(), fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

struct ZeroBlenderAnd : public BaseExpensiveOp {
  template<std::size_t tValueBytes, std::size_t tSize>
  static constexpr bool is_applicable(BlendZeros<tValueBytes, tSize> /*bzs*/) {
    return true;
  }
  template<AnyVector TVec, TypedValueTag<BlendZerosFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr BlendZerosFor<TVec> bzs = TTag::value;
    static constexpr std::size_t size = TVec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = Vector<Int, size>;

    const IVec ivec = reinterpret(vec, type_tag<Int>);
    const IVec mask = static_apply<size>([]<std::size_t... tIdxs>() {
      return set(type_tag<IVec>, ((bzs[tIdxs] == BlendZero::keep) ? Int(-1) : Int(0))...);
    });
    return reinterpret(bitwise_and(ivec, mask), type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 4};
  }
};

template<BlendZero... tBzs, AnyVector TVec>
requires(TVec::size == sizeof...(tBzs))
inline TVec blend_zero(TVec vec) {
  static constexpr auto bzs = BlendZeros<sizeof(typename TVec::Value), TVec::size>{tBzs...};
  using ZeroBlender =
    CheapestType<bzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderMovq, ZeroBlenderInsert32,
                 ZeroBlenderAnd, ZeroBlenderBlend32, ZeroBlenderBlend16>;
  return ZeroBlender::apply(vec, auto_tag<bzs>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP
