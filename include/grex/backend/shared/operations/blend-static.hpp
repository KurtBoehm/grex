// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_STATIC_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_STATIC_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<std::size_t tValueBytes, std::size_t tSize>
struct BlendSelectors {
  static constexpr std::size_t value_size = tValueBytes;
  static constexpr std::size_t size = tSize;
  static constexpr std::size_t lane_size = 16 / value_size;
  using Ctrl = std::array<BlendSelector, size>;

  Ctrl ctrl;

  constexpr BlendSelector operator[](std::size_t i) const {
    return ctrl[i];
  }

  constexpr bool operator==(const BlendSelectors&) const = default;

  [[nodiscard]] constexpr int imm8() const
  requires(size <= 8)
  {
    return static_apply<size>(
      [&]<std::size_t... tIdxs>() { return (0 + ... + (int(ctrl[tIdxs] == rhs_bl) << tIdxs)); });
  }

  [[nodiscard]] constexpr std::optional<BlendSelector> constant() const {
    BlendSelector constant = any_bl;
    for (std::size_t i = 0; i < size; ++i) {
      BlendSelector bl = ctrl[i];
      switch (bl) {
        case lhs_bl: {
          if (constant == rhs_bl) {
            return std::nullopt;
          }
          constant = lhs_bl;
          break;
        }
        case rhs_bl: {
          if (constant == lhs_bl) {
            return std::nullopt;
          }
          constant = rhs_bl;
          break;
        }
        case any_bl: break;
        default: return std::nullopt;
      }
    }
    return constant;
  }

  [[nodiscard]] constexpr BlendSelectors<value_size, lane_size> sub_extended() const
  requires(size < lane_size)
  {
    return static_apply<lane_size>([&]<std::size_t... tIdxs>() {
      return BlendSelectors<value_size, lane_size>{((tIdxs < size) ? ctrl[tIdxs] : any_bl)...};
    });
  }

  [[nodiscard]] constexpr BlendSelectors<value_size, size / 2> lower() const {
    return static_apply<size / 2>(
      [&]<std::size_t... tIdxs>() { return BlendSelectors<value_size, size / 2>{ctrl[tIdxs]...}; });
  }
  [[nodiscard]] constexpr BlendSelectors<value_size, size / 2> upper() const {
    return static_apply<size / 2>([&]<std::size_t... tIdxs>() {
      return BlendSelectors<value_size, size / 2>{ctrl[tIdxs + size / 2]...};
    });
  }

  [[nodiscard]] constexpr std::optional<BlendSelectors<value_size, lane_size>> single_lane() const {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");
    if constexpr (size == lane_size) {
      return *this;
    } else {
      std::array<BlendSelector, lane_size> data =
        static_apply<lane_size>([&]<std::size_t... tIdxs>() {
          return std::array<BlendSelector, lane_size>{ctrl[tIdxs]...};
        });
      for (std::size_t i = lane_size; i < size; ++i) {
        const BlendSelector bz = ctrl[i];
        switch (data[i % lane_size]) {
          case lhs_bl: {
            if (bz != any_bl && bz != lhs_bl) {
              return std::nullopt;
            }
            break;
          }
          case rhs_bl: {
            if (bz != any_bl && bz != rhs_bl) {
              return std::nullopt;
            }
            break;
          }
          case any_bl: {
            data[i % lane_size] = bz;
            break;
          }
          default: {
            return std::nullopt;
          }
        }
      }
      return BlendSelectors<value_size, lane_size>{data};
    }
  }

  template<std::size_t tDstValueBytes>
  friend constexpr std::optional<
    BlendSelectors<tDstValueBytes, tSize * tValueBytes / tDstValueBytes>>
  convert(const BlendSelectors& self) {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");

    constexpr auto dst_size = tSize * tValueBytes / tDstValueBytes;
    using Dst = BlendSelectors<tDstValueBytes, dst_size>;

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
      std::array<BlendSelector, dst_size> entries{};
      for (std::size_t i = 0; i < dst_size; ++i) {
        BlendSelector entry = any_bl;
        for (std::size_t j = 0; j < factor; ++j) {
          const std::size_t k = i * factor + j;
          switch (self.ctrl[k]) {
            case lhs_bl: {
              if (entry == rhs_bl) {
                return std::nullopt;
              }
              entry = lhs_bl;
              break;
            }
            case rhs_bl: {
              if (entry == lhs_bl) {
                return std::nullopt;
              }
              entry = rhs_bl;
              break;
            }
            case any_bl: break;
            default: {
              throw std::invalid_argument{"Invalid BlendSelector!"};
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
using BlendSelectorsFor = BlendSelectors<sizeof(typename TVec::Value), TVec::size>;

template<typename T>
struct AnyBlendSelectorsTrait : public std::false_type {};
template<std::size_t tValueBytes, std::size_t tSize>
struct AnyBlendSelectorsTrait<BlendSelectors<tValueBytes, tSize>> : public std::true_type {};
template<typename T>
concept AnyBlendSelectors = AnyBlendSelectorsTrait<T>::value;

template<AnyBlendSelectors auto tBzs>
struct BlenderTrait;
template<AnyBlendSelectors auto tBzs>
using Blender = BlenderTrait<tBzs>::Type;

template<BlendSelector... tBzs, AnyVector TVec>
requires(TVec::size == sizeof...(tBzs))
inline TVec blend(TVec a, TVec b) {
  static constexpr auto bzs = BlendSelectors<sizeof(typename TVec::Value), TVec::size>{tBzs...};
  return Blender<bzs>::apply(a, b, auto_tag<bzs>);
}

inline void blend_static_test() {
  static constexpr BlendSelectors<4, 4> bzs0{.ctrl = {lhs_bl, lhs_bl, rhs_bl, any_bl}};
  static constexpr BlendSelectors<4, 4> bzs1{.ctrl = {lhs_bl, rhs_bl, rhs_bl, any_bl}};
  static constexpr BlendSelectors<8, 4> bzs2{.ctrl = {lhs_bl, lhs_bl, rhs_bl, any_bl}};
  static constexpr BlendSelectors<8, 4> bzs3{.ctrl = {lhs_bl, any_bl, any_bl, rhs_bl}};

  static constexpr auto ext0 = convert<2>(bzs0);
  static_assert(ext0->ctrl ==
                std::array{lhs_bl, lhs_bl, lhs_bl, lhs_bl, rhs_bl, rhs_bl, any_bl, any_bl});

  static constexpr auto sub0 = convert<8>(bzs0);
  static_assert(sub0->ctrl == std::array{lhs_bl, rhs_bl});
  static constexpr auto sub1 = convert<8>(bzs1);
  static_assert(!sub1.has_value());

  static_assert(bzs0.single_lane() == bzs0);
  static_assert(bzs1.single_lane() == bzs1);
  static_assert(!bzs2.single_lane().has_value());
  static_assert(bzs3.single_lane()->ctrl == std::array{lhs_bl, rhs_bl});
}

struct BlenderConstant : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return tBls.constant().has_value();
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    constexpr BlendSelector bl = tBls.constant().value();
    if constexpr (bl == rhs_bl) {
      return b;
    } else {
      return a;
    }
  }
  static constexpr std::pair<f64, f64> cost(auto /*bzs*/) {
    return {0, 0};
  }
};

struct SubBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  using Base = Blender<tBls.sub_extended()>;

  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return Base<tBls>::is_applicable(auto_tag<tBls.sub_extended()>);
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    return TVec{Base<tBls>::apply(a.full, b.full, auto_tag<tBls.sub_extended()>)};
  }
  template<AnyBlendSelectors auto tBls>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBls> /*tag*/) {
    return Base<tBls>::cost(auto_tag<tBls.sub_extended()>);
  }
};
struct SuperBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto tBls>
  static constexpr bool is_applicable(AutoTag<tBls> /*tag*/) {
    return Blender<tBls.lower()>::is_applicable(auto_tag<tBls.lower()>) &&
           Blender<tBls.upper()>::is_applicable(auto_tag<tBls.upper()>);
  }
  template<AnyVector TVec, BlendSelectorsFor<TVec> tBls>
  static TVec apply(TVec a, TVec b, AutoTag<tBls> /*tag*/) {
    return TVec{
      .lower = Blender<tBls.lower()>::apply(a.lower, b.lower, auto_tag<tBls.lower()>),
      .upper = Blender<tBls.upper()>::apply(a.upper, b.upper, auto_tag<tBls.upper()>),
    };
  }
  template<AnyBlendSelectors auto tBls>
  static constexpr std::pair<f64, f64> cost(AutoTag<tBls> /*tag*/) {
    const auto [c0a, c1a] = Blender<tBls.lower()>::cost(auto_tag<tBls.lower()>);
    const auto [c0b, c1b] = Blender<tBls.upper()>::cost(auto_tag<tBls.upper()>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size < register_bytes.front()))
struct BlenderTrait<tBls> {
  using Type = SubBlender;
};
template<AnyBlendSelectors auto tBls>
requires((tBls.value_size * tBls.size > register_bytes.back()))
struct BlenderTrait<tBls> {
  using Type = SuperBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_STATIC_HPP
