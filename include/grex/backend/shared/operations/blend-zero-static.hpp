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

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<std::size_t ValueBytes, std::size_t N>
struct BlendZeroSelectors {
  static constexpr std::size_t value_size = ValueBytes;
  static constexpr std::size_t size = N;
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
    return static_apply<size>([&]<std::size_t... I> {
      return (0 + ... + (int(ctrl[I] == keep_bz) << I)); // NOLINT(bugprone-signed-bitwise)
    });
  }

  [[nodiscard]] constexpr BlendZeroSelectors<value_size, lane_size> sub_extended() const
  requires(size < lane_size)
  {
    return static_apply<lane_size>([&]<std::size_t... I> {
      return BlendZeroSelectors<value_size, lane_size>{((I < size) ? ctrl[I] : any_bz)...};
    });
  }

  [[nodiscard]] constexpr BlendZeroSelectors<value_size, size / 2> lower() const {
    return static_apply<size / 2>(
      [&]<std::size_t... I> { return BlendZeroSelectors<value_size, size / 2>{ctrl[I]...}; });
  }
  [[nodiscard]] constexpr BlendZeroSelectors<value_size, size / 2> upper() const {
    return static_apply<size / 2>([&]<std::size_t... I> {
      return BlendZeroSelectors<value_size, size / 2>{ctrl[I + size / 2]...};
    });
  }

  [[nodiscard]] constexpr std::optional<BlendZeroSelectors<value_size, lane_size>>
  single_lane() const {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");
    if constexpr (size == lane_size) {
      return *this;
    } else {
      std::array<BlendZeroSelector, lane_size> data = static_apply<lane_size>(
        [&]<std::size_t... I> { return std::array<BlendZeroSelector, lane_size>{ctrl[I]...}; });
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

  template<std::size_t DstValueBytes>
  friend constexpr std::optional<BlendZeroSelectors<DstValueBytes, N * ValueBytes / DstValueBytes>>
  convert(const BlendZeroSelectors& self) {
    static_assert(size >= lane_size, "At least one lane needs to be populated!");

    constexpr auto dst_size = N * ValueBytes / DstValueBytes;
    using Dst = BlendZeroSelectors<DstValueBytes, dst_size>;

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
template<AnyVector Vec>
using BlendZeroSelectorsFor = BlendZeroSelectors<sizeof(typename Vec::Value), Vec::size>;

template<typename T>
struct AnyBlendZeroSelectorsTrait : public std::false_type {};
template<std::size_t ValueBytes, std::size_t N>
struct AnyBlendZeroSelectorsTrait<BlendZeroSelectors<ValueBytes, N>> : public std::true_type {};
template<typename T>
concept AnyBlendZeroSelectors = AnyBlendZeroSelectorsTrait<T>::value;

template<AnyBlendZeroSelectors auto BZS>
struct ZeroBlenderTrait;
template<AnyBlendZeroSelectors auto BZS>
using ZeroBlender = ZeroBlenderTrait<BZS>::Type;

template<BlendZeroSelector... BZS, AnyVector Vec>
requires(Vec::size == sizeof...(BZS))
inline Vec blend_zero(Vec vec) {
  static constexpr auto bzs = BlendZeroSelectors<sizeof(typename Vec::Value), Vec::size>{BZS...};
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
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return static_apply<BZS.size>(
      [&]<std::size_t... I> { return (... && (BZS[I] == keep_bz || BZS[I] == any_bz)); });
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    static_assert(is_applicable(auto_tag<BZS>));
    return vec;
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 0, .latency = 0};
  }
};
struct ZeroBlenderZero : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return static_apply<BZS.size>(
      [&]<std::size_t... I> { return (... && (BZS[I] == zero_bz || BZS[I] == any_bz)); });
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec /*vec*/, AutoTag<BZS> /*tag*/) {
    static_assert(is_applicable(auto_tag<BZS>));
    return zeros(type_tag<Vec>);
  }
  static constexpr Cost cost(auto /*bzs*/) {
    return {.inv_throughput = 0, .latency = 1};
  }
};

struct SubZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto BZS>
  using Base = ZeroBlender<BZS.sub_extended()>;

  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return Base<BZS>::is_applicable(auto_tag<BZS.sub_extended()>);
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    return Vec{Base<BZS>::apply(vec.full, auto_tag<BZS.sub_extended()>)};
  }
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr Cost cost(AutoTag<BZS> /*tag*/) {
    return Base<BZS>::cost(auto_tag<BZS.sub_extended()>);
  }
};
struct SuperZeroBlender : public BaseExpensiveOp {
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr bool is_applicable(AutoTag<BZS> /*tag*/) {
    return ZeroBlender<BZS.lower()>::is_applicable(auto_tag<BZS.lower()>) &&
           ZeroBlender<BZS.upper()>::is_applicable(auto_tag<BZS.upper()>);
  }
  template<AnyVector Vec, BlendZeroSelectorsFor<Vec> BZS>
  static Vec apply(Vec vec, AutoTag<BZS> /*tag*/) {
    return Vec{
      .lower = ZeroBlender<BZS.lower()>::apply(vec.lower, auto_tag<BZS.lower()>),
      .upper = ZeroBlender<BZS.upper()>::apply(vec.upper, auto_tag<BZS.upper()>),
    };
  }
  template<AnyBlendZeroSelectors auto BZS>
  static constexpr Cost cost(AutoTag<BZS> /*tag*/) {
    const auto [c0a, c1a] = ZeroBlender<BZS.lower()>::cost(auto_tag<BZS.lower()>);
    const auto [c0b, c1b] = ZeroBlender<BZS.upper()>::cost(auto_tag<BZS.upper()>);
    return {.inv_throughput = c0a + c0b, .latency = c1a + c1b};
  }
};

template<AnyBlendZeroSelectors auto BZS>
requires((BZS.value_size * BZS.size < register_bytes.front())) // NOLINT(*-redundant-parentheses)
struct ZeroBlenderTrait<BZS> {
  using Type = SubZeroBlender;
};
template<AnyBlendZeroSelectors auto BZS>
requires((BZS.value_size * BZS.size > register_bytes.back())) // NOLINT(*-redundant-parentheses)
struct ZeroBlenderTrait<BZS> {
  using Type = SuperZeroBlender;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_BLEND_ZERO_STATIC_HPP
