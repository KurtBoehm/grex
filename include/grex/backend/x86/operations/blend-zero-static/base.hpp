// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_BASE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_BASE_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "grex/backend/defs.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<std::size_t tValueBytes, std::size_t tSize>
struct BlendZeros {
  static constexpr std::size_t value_size = tValueBytes;
  static constexpr std::size_t size = tSize;
  using Ctrl = std::array<BlendZero, size>;

  Ctrl ctrl;

  constexpr BlendZero operator[](std::size_t i) const {
    return ctrl[i];
  }

  template<std::size_t tDstValueBytes>
  friend constexpr std::optional<BlendZeros<tDstValueBytes, tSize * tValueBytes / tDstValueBytes>>
  convert(const BlendZeros& self) {
    constexpr auto dst_size = tSize * tValueBytes / tDstValueBytes;
    using Dst = BlendZeros<tDstValueBytes, dst_size>;

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
      std::array<BlendZero, dst_size> entries{};
      for (std::size_t i = 0; i < dst_size; ++i) {
        BlendZero entry = BlendZero::any;
        for (std::size_t j = 0; j < factor; ++j) {
          const std::size_t k = i * factor + j;
          switch (self.ctrl[k]) {
          case BlendZero::zero: {
            if (entry == BlendZero::keep) {
              return std::nullopt;
            }
            entry = BlendZero::zero;
            break;
          }
          case BlendZero::keep: {
            if (entry == BlendZero::zero) {
              return std::nullopt;
            }
            entry = BlendZero::keep;
            break;
          }
          case BlendZero::any: break;
          default: {
            throw std::invalid_argument{"Invalid BlendZero!"};
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
using BlendZerosFor = BlendZeros<sizeof(typename TVec::Value), TVec::size>;

template<typename T>
struct AnyBlendZerosTrait : public std::false_type {};
template<std::size_t tValueBytes, std::size_t tSize>
struct AnyBlendZerosTrait<BlendZeros<tValueBytes, tSize>> : public std::true_type {};
template<typename T>
concept AnyBlendZeros = AnyBlendZerosTrait<T>::value;

template<AnyBlendZeros auto tBzs>
struct ZeroBlenderTrait;
template<AnyBlendZeros auto tBzs>
using ZeroBlender = ZeroBlenderTrait<tBzs>::Type;

template<BlendZero... tBzs, AnyVector TVec>
requires(TVec::size == sizeof...(tBzs))
inline TVec blend_zero(TVec vec) {
  static constexpr auto bzs = BlendZeros<sizeof(typename TVec::Value), TVec::size>{tBzs...};
  return ZeroBlender<bzs>::apply(vec, auto_tag<bzs>);
}

inline void blend_zero_static_test() {
  static constexpr BlendZeros<4, 4> bzs0{.ctrl = {zero_bz, zero_bz, keep_bz, any_bz}};
  static constexpr BlendZeros<4, 4> bzs1{.ctrl = {zero_bz, keep_bz, keep_bz, any_bz}};

  static constexpr auto ext0 = convert<2>(bzs0);
  static_assert(ext0->ctrl ==
                std::array{zero_bz, zero_bz, zero_bz, zero_bz, keep_bz, keep_bz, any_bz, any_bz});

  static constexpr auto sub0 = convert<8>(bzs0);
  static_assert(sub0->ctrl == std::array{zero_bz, keep_bz});
  static constexpr auto sub1 = convert<8>(bzs1);
  static_assert(!sub1.has_value());
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_BLEND_ZERO_STATIC_BASE_HPP
