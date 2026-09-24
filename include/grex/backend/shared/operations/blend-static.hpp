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
#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<std::size_t ValueBytes, std::size_t N>
struct BlendSelectors {
  static constexpr std::size_t value_size = ValueBytes;
  static constexpr std::size_t size = N;
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
    return static_apply<size>([&]<std::size_t... I> {
      return (0 + ... + (int(ctrl[I] == rhs_bl) << I)); // NOLINT(bugprone-signed-bitwise)
    });
  }

  [[nodiscard]] constexpr std::optional<BlendSelector> constant() const {
    BlendSelector constant = any_bl;
    for (std::size_t i = 0; i < size; ++i) {
      switch (ctrl[i]) {
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
    return static_apply<lane_size>([&]<std::size_t... I> {
      return BlendSelectors<value_size, lane_size>{((I < size) ? ctrl[I] : any_bl)...};
    });
  }

  [[nodiscard]] constexpr BlendSelectors<value_size, size / 2> lower() const {
    return static_apply<size / 2>(
      [&]<std::size_t... I> { return BlendSelectors<value_size, size / 2>{ctrl[I]...}; });
  }
  [[nodiscard]] constexpr BlendSelectors<value_size, size / 2> upper() const {
    return static_apply<size / 2>([&]<std::size_t... I> {
      return BlendSelectors<value_size, size / 2>{ctrl[I + size / 2]...};
    });
  }

  [[nodiscard]] constexpr std::optional<BlendSelectors<value_size, lane_size>> single_lane() const {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");
    if constexpr (size == lane_size) {
      return *this;
    } else {
      std::array<BlendSelector, lane_size> data = static_apply<lane_size>(
        [&]<std::size_t... I> { return std::array<BlendSelector, lane_size>{ctrl[I]...}; });
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

  template<std::size_t DstValueBytes>
  friend constexpr std::optional<BlendSelectors<DstValueBytes, N * ValueBytes / DstValueBytes>>
  convert(const BlendSelectors& self) {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");

    constexpr auto dst_size = N * ValueBytes / DstValueBytes;
    using Dst = BlendSelectors<DstValueBytes, dst_size>;

    if constexpr (DstValueBytes == ValueBytes) {
      return self;
    } else if constexpr (DstValueBytes < ValueBytes) {
      // simply repeat the entries
      constexpr auto factor = ValueBytes / DstValueBytes;
      const auto entries = static_apply<dst_size>(
        [&]<std::size_t... I> { return std::array{self.ctrl[I / factor]...}; });
      return Dst{.ctrl = entries};
    } else {
      // check whether the entries are the same (ignoring any)
      constexpr auto factor = DstValueBytes / ValueBytes;
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
template<AnyVector Vec>
using BlendSelectorsFor = BlendSelectors<sizeof(typename Vec::Value), Vec::size>;

template<typename T>
struct AnyBlendSelectorsTrait : public std::false_type {};
template<std::size_t ValueBytes, std::size_t N>
struct AnyBlendSelectorsTrait<BlendSelectors<ValueBytes, N>> : public std::true_type {};
template<typename T>
concept AnyBlendSelectors = AnyBlendSelectorsTrait<T>::value;

template<AnyBlendSelectors auto BS>
struct BlenderTrait;
template<AnyBlendSelectors auto BS>
using Blender = BlenderTrait<BS>::Type;

template<BlendSelector... BS, AnyVector Vec>
requires(Vec::size == sizeof...(BS))
inline Vec blend(Vec a, Vec b) {
  static constexpr auto bzs = BlendSelectors<sizeof(typename Vec::Value), Vec::size>{BS...};
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
  template<AnyBlendSelectors auto BS>
  static constexpr bool is_applicable(AutoTag<BS> /*tag*/) {
    return BS.constant().has_value();
  }
  template<AnyVector Vec, BlendSelectorsFor<Vec> BS>
  static Vec apply(Vec a, Vec b, AutoTag<BS> /*tag*/) {
    constexpr BlendSelector bl = BS.constant().value();
    if constexpr (bl == rhs_bl) {
      return b;
    } else {
      return a;
    }
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 0, .latency = 0};
  }
};

struct SubBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto BS>
  using Base = Blender<BS.sub_extended()>;

  template<AnyBlendSelectors auto BS>
  static constexpr bool is_applicable(AutoTag<BS> /*tag*/) {
    return Base<BS>::is_applicable(auto_tag<BS.sub_extended()>);
  }
  template<AnyVector Vec, BlendSelectorsFor<Vec> BS>
  static Vec apply(Vec a, Vec b, AutoTag<BS> /*tag*/) {
    return Vec{Base<BS>::apply(a.full, b.full, auto_tag<BS.sub_extended()>)};
  }
  template<AnyBlendSelectors auto BS>
  static constexpr Cost cost(AutoTag<BS> /*tag*/) {
    return Base<BS>::cost(auto_tag<BS.sub_extended()>);
  }
};
struct SuperBlender : public BaseExpensiveOp {
  template<AnyBlendSelectors auto BS>
  static constexpr bool is_applicable(AutoTag<BS> /*tag*/) {
    return Blender<BS.lower()>::is_applicable(auto_tag<BS.lower()>) &&
           Blender<BS.upper()>::is_applicable(auto_tag<BS.upper()>);
  }
  template<AnyVector Vec, BlendSelectorsFor<Vec> BS>
  static Vec apply(Vec a, Vec b, AutoTag<BS> /*tag*/) {
    return Vec{
      .lower = Blender<BS.lower()>::apply(a.lower, b.lower, auto_tag<BS.lower()>),
      .upper = Blender<BS.upper()>::apply(a.upper, b.upper, auto_tag<BS.upper()>),
    };
  }
  template<AnyBlendSelectors auto BS>
  static constexpr Cost cost(AutoTag<BS> /*tag*/) {
    const auto [c0a, c1a] = Blender<BS.lower()>::cost(auto_tag<BS.lower()>);
    const auto [c0b, c1b] = Blender<BS.upper()>::cost(auto_tag<BS.upper()>);
    return {.inv_throughput = c0a + c0b, .latency = c1a + c1b};
  }
};

template<AnyBlendSelectors auto BS>
requires((BS.value_size * BS.size < register_bytes.front())) // NOLINT(*-redundant-parentheses)
struct BlenderTrait<BS> {
  using Type = SubBlender;
};
template<AnyBlendSelectors auto BS>
requires((BS.value_size * BS.size > register_bytes.back())) // NOLINT(*-redundant-parentheses)
struct BlenderTrait<BS> {
  using Type = SuperBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_STATIC_HPP
