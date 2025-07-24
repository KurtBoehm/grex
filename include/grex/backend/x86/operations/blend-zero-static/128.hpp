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
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == keep_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return vec;
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};
struct ZeroBlenderZero : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == zero_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec /*vec*/, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return zeros(type_tag<TVec>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 1};
  }
};

struct ZeroBlenderMovq : public BaseExpensiveOp {
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    constexpr std::optional<BlendZeros<8, 2>> obz64 = convert<8>(tBzs);
    if (!obz64.has_value()) {
      return false;
    }
    const BlendZeros<8, 2> bz64 = obz64.value();
    return (bz64[0] == keep_bz || bz64[0] == any_bz) && (bz64[1] == zero_bz || bz64[1] == any_bz);
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
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
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<4>(tBzs).value();

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
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<4>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<4>(tBzs).value();

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
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
#if GREX_X86_64_LEVEL >= 2
    return convert<2>(tBzs).has_value();
#else
    return false;
#endif
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto bzs = convert<2>(tBzs).value();

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
  template<AnyBlendZeros auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, BlendZerosFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    using Value = TVec::Value;
    static constexpr std::size_t size = TVec::size;
    using Int = SignedInt<sizeof(Value)>;
    using IVec = Vector<Int, size>;

    const IVec ivec = reinterpret(vec, type_tag<Int>);
    const IVec mask = static_apply<size>([]<std::size_t... tIdxs>() {
      return set(type_tag<IVec>, ((tBzs[tIdxs] == BlendZero::keep) ? Int(-1) : Int(0))...);
    });
    return reinterpret(bitwise_and(ivec, mask), type_tag<Value>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0.5, 4};
  }
};

template<AnyBlendZeros auto tBzs>
requires((tBzs.value_size * tBzs.size == 16))
struct ZeroBlenderTrait<tBzs> {
  using Type =
    CheapestType<tBzs, ZeroBlenderNoop, ZeroBlenderZero, ZeroBlenderMovq, ZeroBlenderInsert32,
                 ZeroBlenderAnd, ZeroBlenderBlend32, ZeroBlenderBlend16>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_128_HPP
