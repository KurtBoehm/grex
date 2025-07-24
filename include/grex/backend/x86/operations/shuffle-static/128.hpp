// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP

#include <cstddef>
#include <utility>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/defs.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/blend-zero-static/base.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/reinterpret.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/operations/shuffle-static/base.hpp"
#include "grex/backend/x86/types.hpp"
#include "grex/base/defs.hpp"

// TODO Very inefficient for 8- and 16-bit values on level 1, and while a more efficient 16-bit
//      implementation is not that complicated, even that requires more work than I am willing
//      to invest in such old archtectures (level 2 is ubiquitous, even on the low end, since 2013).

namespace grex::backend {
struct ShufflerShuffle8 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return GREX_X86_64_LEVEL >= 2;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndices<1, 16> idxs = convert<1>(tSh).value();

    const i8x16 ivec = reinterpret(vec, type_tag<i8>);
    const i8x16 shuf = static_apply<16>([]<std::size_t... tIdxs>() {
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? u8(sh) : u8(0xFF); };
      return set(type_tag<i8x16>, f(idxs[tIdxs])...);
    });
    return reinterpret(i8x16{_mm_shuffle_epi8(ivec.r, shuf.r)}, type_tag<Value>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    return {0.5, 4};
  }
};

struct ShufflerShuffle32 : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return convert<4>(tSh).has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec blend_zeros(TVec vec, AutoTag<tSh> /*tag*/) {
    return ZeroBlender<tSh.blend_zeros()>::apply(vec, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    using Value = TVec::Value;
    static constexpr auto idxs = convert<4>(tSh).value();

    const i32x4 ivec = reinterpret(vec, type_tag<i32>);
    static constexpr int imm8 = static_apply<4>([]<std::size_t... tIdxs>() {
      auto f = [](int i, ShuffleIndex sh) { return is_index(sh) ? int(sh) : i; };
      return (0 + ... + (f(tIdxs, idxs[tIdxs]) << (2 * tIdxs)));
    });
    const TVec shuffled = reinterpret(i32x4{_mm_shuffle_epi32(ivec.r, imm8)}, type_tag<Value>);
    return ZeroBlender<tSh.blend_zeros()>::apply(shuffled, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    const auto [c0, c1] = ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
    return {0.5 + c0, std::max<f64>(c1, 1)};
  }
};

struct ShufflerBlendZero : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return static_apply<tSh.size>([]<std::size_t... tIdxs>() {
      return (... && (!is_index(tSh[tIdxs]) || u8(tSh[tIdxs]) == tIdxs));
    });
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    static_assert(is_applicable(auto_tag<tSh>));
    return ZeroBlender<tSh.blend_zeros()>::apply(vec, auto_tag<tSh.blend_zeros()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*idxs*/) {
    static_assert(is_applicable(auto_tag<tSh>));
    return ZeroBlender<tSh.blend_zeros()>::cost(auto_tag<tSh>);
  }
};

struct ShufflerExtractSet : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tIdxs>
  static constexpr bool is_applicable(AutoTag<tIdxs> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, TypedValueTag<ShuffleIndicesFor<TVec>> TTag>
  static TVec apply(TVec vec, TTag /*tag*/) {
    using Value = TVec::Value;
    static constexpr ShuffleIndicesFor<TVec> bzs = TTag::value;
    static constexpr std::size_t size = TVec::size;

    auto f = [&](std::size_t i) { return is_index(bzs[i]) ? extract(vec, u8(bzs[i])) : Value{}; };
    return static_apply<size>(
      [&]<std::size_t... tIdxs>() { return set(type_tag<TVec>, f(tIdxs)...); });
  }
  template<AnyShuffleIndices auto tIdxs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tIdxs> /*idxs*/) {
    return {f64(tIdxs.size * 2), 1};
  }
};

template<ShuffleIndex... tIdxs, AnyVector TVec>
requires(TVec::size == sizeof...(tIdxs) && sizeof(typename TVec::Register) == 16)
inline TVec shuffle(TVec vec) {
  static constexpr auto idxs = ShuffleIndicesFor<TVec>{.indices = {tIdxs...}};
  using ZeroBlender =
    CheapestType<idxs, ShufflerShuffle8, ShufflerShuffle32, ShufflerBlendZero, ShufflerExtractSet>;
  return ZeroBlender::apply(vec, auto_tag<idxs>);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_128_HPP
