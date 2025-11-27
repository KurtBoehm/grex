// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP

#include <optional>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend-zero-static/shared.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base.hpp"

// TODO There are no optimized implementations at level 1, as those would require a lot of work
//      for very old architectures.

namespace grex::backend {
struct ZeroBlenderMovq : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    constexpr std::optional<BlendZeroSelectors<8, 2>> obz64 = convert<8>(tBzs);
    if (!obz64.has_value()) {
      return false;
    }
    const BlendZeroSelectors<8, 2> bz64 = obz64.value();
    return (bz64[0] == keep_bz || bz64[0] == any_bz) && (bz64[1] == zero_bz || bz64[1] == any_bz);
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
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
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<4>(tBzs).value().imm8();
    const f32x4 fvec = reinterpret(vec, type_tag<f32>);
    return reinterpret(f32x4{_mm_insert_ps(fvec.r, fvec.r, ~imm8 & 0xF)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 1};
  }
};

struct ZeroBlenderBlend32x4 : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<4>(tBzs).value().imm8();
    const f32x4 fvec = reinterpret(vec, type_tag<f32>);
    return reinterpret(f32x4{_mm_blend_ps(_mm_setzero_ps(), fvec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

struct ZeroBlenderBlend16x8 : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<2>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr int imm8 = convert<2>(tBzs).value().imm8();
    const i16x8 ivec = reinterpret(vec, type_tag<i16>);
    return reinterpret(i16x8{_mm_blend_epi16(_mm_setzero_si128(), ivec.r, imm8)}, type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 2};
  }
};

template<AnyBlendZeroSelectors auto tBzs>
requires((tBzs.value_size * tBzs.size == 16))
struct ZeroBlenderTrait<tBzs> {
  using Type =
    CheapestType<tBzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderMovq, ZeroBlenderInsert32,
                 ZeroBlenderBlend32x4, ZeroBlenderBlend16x8, ZeroBlenderAnd>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP
