// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_ZERO_STATIC_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_ZERO_STATIC_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<std::size_t tValueBytes, std::size_t tSize>
struct BlendZeroSelectors {
  static constexpr std::size_t value_size = tValueBytes;
  static constexpr std::size_t size = tSize;
  static constexpr std::size_t lane_size = 16 / value_size;
  using Ctrl = std::array<BlendZeroSelector, size>;

  Ctrl ctrl;

  constexpr BlendZeroSelector operator[](std::size_t i) const {
    return ctrl[i];
  }

  constexpr bool operator==(const BlendZeroSelectors&) const = default;

  [[nodiscard]] constexpr int imm8() const
  requires(size <= 8)
  {
    return static_apply<size>(
      [&]<std::size_t... tIdxs>() { return (0 + ... + (int(ctrl[tIdxs] == keep_bz) << tIdxs)); });
  }

  [[nodiscard]] constexpr BlendZeroSelectors<value_size, lane_size> sub_extended() const
  requires(size < lane_size)
  {
    return static_apply<lane_size>([&]<std::size_t... tIdxs>() {
      return BlendZeroSelectors<value_size, lane_size>{((tIdxs < size) ? ctrl[tIdxs] : any_bz)...};
    });
  }

  [[nodiscard]] constexpr BlendZeroSelectors<value_size, size / 2> lower() const {
    return static_apply<size / 2>([&]<std::size_t... tIdxs>() {
      return BlendZeroSelectors<value_size, size / 2>{ctrl[tIdxs]...};
    });
  }
  [[nodiscard]] constexpr BlendZeroSelectors<value_size, size / 2> upper() const {
    return static_apply<size / 2>([&]<std::size_t... tIdxs>() {
      return BlendZeroSelectors<value_size, size / 2>{ctrl[tIdxs + size / 2]...};
    });
  }

  [[nodiscard]] constexpr std::optional<BlendZeroSelectors<value_size, lane_size>>
  single_lane() const {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");
    if constexpr (size == lane_size) {
      return *this;
    } else {
      std::array<BlendZeroSelector, lane_size> data =
        static_apply<lane_size>([&]<std::size_t... tIdxs>() {
          return std::array<BlendZeroSelector, lane_size>{ctrl[tIdxs]...};
        });
      for (std::size_t i = lane_size; i < size; ++i) {
        const BlendZeroSelector bz = ctrl[i];
        switch (data[i % lane_size]) {
          case zero_bz: {
            if (bz != any_bz && bz != zero_bz) {
              return std::nullopt;
            }
            break;
          }
          case keep_bz: {
            if (bz != any_bz && bz != keep_bz) {
              return std::nullopt;
            }
            break;
          }
          case any_bz: {
            data[i % lane_size] = bz;
            break;
          }
          default: {
            return std::nullopt;
          }
        }
      }
      return BlendZeroSelectors<value_size, lane_size>{data};
    }
  }

  template<std::size_t tDstValueBytes>
  friend constexpr std::optional<
    BlendZeroSelectors<tDstValueBytes, tSize * tValueBytes / tDstValueBytes>>
  convert(const BlendZeroSelectors& self) {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");

    constexpr auto dst_size = tSize * tValueBytes / tDstValueBytes;
    using Dst = BlendZeroSelectors<tDstValueBytes, dst_size>;

    if constexpr (tDstValueBytes == tValueBytes) {
      return self;
    } else if constexpr (tDstValueBytes < tValueBytes) {
      // simply repeat the entries
      constexpr auto factor = tValueBytes / tDstValueBytes;
      const auto entries = static_apply<dst_size>(
        [&]<std::size_t... tIdxs>() { return std::array{self.ctrl[tIdxs / factor]...}; });
      return Dst{.ctrl = entries};
    } else {
      // check whether the entries are the same (ignoring any)
      constexpr auto factor = tDstValueBytes / tValueBytes;
      std::array<BlendZeroSelector, dst_size> entries{};
      for (std::size_t i = 0; i < dst_size; ++i) {
        BlendZeroSelector entry = any_bz;
        for (std::size_t j = 0; j < factor; ++j) {
          const std::size_t k = i * factor + j;
          switch (self.ctrl[k]) {
            case zero_bz: {
              if (entry == keep_bz) {
                return std::nullopt;
              }
              entry = zero_bz;
              break;
            }
            case keep_bz: {
              if (entry == zero_bz) {
                return std::nullopt;
              }
              entry = keep_bz;
              break;
            }
            case any_bz: break;
            default: {
              throw std::invalid_argument{"Invalid BlendZeroSelector!"};
            }
          }
        }
        entries[i] = entry;
      }
      return Dst{.ctrl = entries};
    }
  }
};
template<AnyVector TVec>
using BlendZeroSelectorsFor = BlendZeroSelectors<sizeof(typename TVec::Value), TVec::size>;

template<typename T>
struct AnyBlendZeroSelectorsTrait : public std::false_type {};
template<std::size_t tValueBytes, std::size_t tSize>
struct AnyBlendZeroSelectorsTrait<BlendZeroSelectors<tValueBytes, tSize>> : public std::true_type {
};
template<typename T>
concept AnyBlendZeroSelectors = AnyBlendZeroSelectorsTrait<T>::value;

template<AnyBlendZeroSelectors auto tBzs>
struct ZeroBlenderTrait;
template<AnyBlendZeroSelectors auto tBzs>
using ZeroBlender = ZeroBlenderTrait<tBzs>::Type;

template<BlendZeroSelector... tBzs, AnyVector TVec>
requires(TVec::size == sizeof...(tBzs))
inline TVec blend_zero(TVec vec) {
  static constexpr auto bzs = BlendZeroSelectors<sizeof(typename TVec::Value), TVec::size>{tBzs...};
  return ZeroBlender<bzs>::apply(vec, auto_tag<bzs>);
}

inline void blend_zero_static_test() {
  static constexpr BlendZeroSelectors<4, 4> bzs0{.ctrl = {zero_bz, zero_bz, keep_bz, any_bz}};
  static constexpr BlendZeroSelectors<4, 4> bzs1{.ctrl = {zero_bz, keep_bz, keep_bz, any_bz}};
  static constexpr BlendZeroSelectors<8, 4> bzs2{.ctrl = {zero_bz, zero_bz, keep_bz, any_bz}};
  static constexpr BlendZeroSelectors<8, 4> bzs3{.ctrl = {zero_bz, any_bz, any_bz, keep_bz}};

  static constexpr auto ext0 = convert<2>(bzs0);
  static_assert(ext0->ctrl ==
                std::array{zero_bz, zero_bz, zero_bz, zero_bz, keep_bz, keep_bz, any_bz, any_bz});

  static constexpr auto sub0 = convert<8>(bzs0);
  static_assert(sub0->ctrl == std::array{zero_bz, keep_bz});
  static constexpr auto sub1 = convert<8>(bzs1);
  static_assert(!sub1.has_value());

  static_assert(bzs0.single_lane() == bzs0);
  static_assert(bzs1.single_lane() == bzs1);
  static_assert(!bzs2.single_lane().has_value());
  static_assert(bzs3.single_lane()->ctrl == std::array{zero_bz, keep_bz});
}

struct ZeroBlenderNoop : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == keep_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return vec;
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};
struct ZeroBlenderZero : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return static_apply<tBzs.size>([&]<std::size_t... tIdxs>() {
      return (... && (tBzs[tIdxs] == zero_bz || tBzs[tIdxs] == any_bz));
    });
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec /*vec*/, AutoTag<tBzs> /*tag*/) {
    static_assert(is_applicable(auto_tag<tBzs>));
    return zeros(type_tag<TVec>);
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 1};
  }
};

struct SubZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  using Base = ZeroBlender<tBzs.sub_extended()>;

  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return Base<tBzs>::is_applicable(auto_tag<tBzs.sub_extended()>);
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    return TVec{Base<tBzs>::apply(vec.full, auto_tag<tBzs.sub_extended()>)};
  }
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBzs> /*tag*/) {
    return Base<tBzs>::cost(auto_tag<tBzs.sub_extended()>);
  }
};
struct SuperZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr bool is_applicable(AutoTag<tBzs> /*tag*/) {
    return ZeroBlender<tBzs.lower()>::is_applicable(auto_tag<tBzs.lower()>) &&
           ZeroBlender<tBzs.upper()>::is_applicable(auto_tag<tBzs.upper()>);
  }
  template<AnyVector TVec, BlendZeroSelectorsFor<TVec> tBzs>
  static TVec apply(TVec vec, AutoTag<tBzs> /*tag*/) {
    return TVec{
      .lower = ZeroBlender<tBzs.lower()>::apply(vec.lower, auto_tag<tBzs.lower()>),
      .upper = ZeroBlender<tBzs.upper()>::apply(vec.upper, auto_tag<tBzs.upper()>),
    };
  }
  template<AnyBlendZeroSelectors auto tBzs>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBzs> /*tag*/) {
    const auto [c0a, c1a] = ZeroBlender<tBzs.lower()>::cost(auto_tag<tBzs.lower()>);
    const auto [c0b, c1b] = ZeroBlender<tBzs.upper()>::cost(auto_tag<tBzs.upper()>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyBlendZeroSelectors auto tBzs>
requires((tBzs.value_size * tBzs.size < register_bytes.front()))
struct ZeroBlenderTrait<tBzs> {
  using Type = SubZeroBlender;
};
template<AnyBlendZeroSelectors auto tBzs>
requires((tBzs.value_size * tBzs.size > register_bytes.back()))
struct ZeroBlenderTrait<tBzs> {
  using Type = SuperZeroBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_ZERO_STATIC_HPP
