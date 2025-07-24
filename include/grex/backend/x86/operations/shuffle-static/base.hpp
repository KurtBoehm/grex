// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_BASE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_BASE_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "grex/backend/defs.hpp"
#include "grex/backend/x86/operations/blend-zero-static.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<std::size_t tValueBytes, std::size_t tSize>
struct ShuffleIndices {
  static constexpr std::size_t value_size = tValueBytes;
  static constexpr std::size_t size = tSize;
  using Indices = std::array<ShuffleIndex, size>;

  Indices indices;
  // whether partial entries need to be zeroed
  bool subzero = false;
  // whether these indices are converted already
  bool converted = false;

  constexpr ShuffleIndex operator[](std::size_t i) const {
    return indices[i];
  }

  template<std::size_t tDstValueBytes>
  friend constexpr std::optional<
    ShuffleIndices<tDstValueBytes, tSize * tValueBytes / tDstValueBytes>>
  convert(const ShuffleIndices& self) {
    constexpr auto dst_size = tSize * tValueBytes / tDstValueBytes;
    using Dst = ShuffleIndices<tDstValueBytes, dst_size>;

    if constexpr (tDstValueBytes == tValueBytes) {
      return self;
    } else if constexpr (tDstValueBytes < tValueBytes) {
      // simply multiply the entries with factor and add their chunk index
      constexpr auto factor = tValueBytes / tDstValueBytes;
      auto f = [&](ShuffleIndex sh, std::size_t chunki) {
        if (is_index(sh)) {
          return ShuffleIndex(u8(sh) * factor + chunki);
        }
        return sh;
      };
      const auto idxs = static_apply<dst_size>([&]<std::size_t... tIdxs>() {
        return std::array{f(self.indices[tIdxs / factor], tIdxs % factor)...};
      });
      return Dst{.indices = idxs, .subzero = self.subzero, .converted = true};
    } else {
      // check whether the indices in each chunk that is converted to one index
      // start at a multiple of `factor` and are ascending from there (apart from any/zero)
      constexpr auto factor = tDstValueBytes / tValueBytes;
      std::array<ShuffleIndex, dst_size> idxs{};
      bool subz = false;
      for (std::size_t i = 0; i < dst_size; ++i) {
        std::optional<u8> dsti{};
        bool haszero = false;

        for (std::size_t j = 0; j < factor; ++j) {
          const std::size_t k = i * factor + j;
          const ShuffleIndex shi = self.indices[k];
          switch (shi) {
          case ShuffleIndex::any: break;
          case ShuffleIndex::zero: {
            haszero = true;
            break;
          }
          default: {
            const auto srci = u8(shi);
            if (srci % factor != j) {
              return std::nullopt;
            }
            const auto fi = srci / factor;
            if (dsti.has_value() && *dsti != fi) {
              return std::nullopt;
            }
            dsti = fi;
            break;
          }
          }
        }

        if (dsti.has_value()) {
          idxs[i] = ShuffleIndex{*dsti};
          if (haszero) {
            subz = true;
          }
        } else {
          idxs[i] = haszero ? ShuffleIndex::zero : ShuffleIndex::any;
        }
      }

      return Dst{.indices = idxs, .subzero = subz, .converted = true};
    }
  }

  [[nodiscard]] constexpr BlendZeros<tValueBytes, tSize> blend_zeros() const {
    auto f = [](ShuffleIndex sh) {
      switch (sh) {
      case ShuffleIndex::any: return any_bz;
      case ShuffleIndex::zero: return zero_bz;
      default: return keep_bz;
      }
    };
    return static_apply<tSize>(
      [&]<std::size_t... tIdxs>() { return BlendZeros<tValueBytes, tSize>{f(indices[tIdxs])...}; });
  }
};
template<typename T>
struct AnyShuffleIndicesTrait : public std::false_type {};
template<std::size_t tValueBytes, std::size_t tSize>
struct AnyShuffleIndicesTrait<ShuffleIndices<tValueBytes, tSize>> : public std::true_type {};
template<typename T>
concept AnyShuffleIndices = AnyShuffleIndicesTrait<T>::value;
template<AnyVector TVec>
using ShuffleIndicesFor = ShuffleIndices<sizeof(typename TVec::Value), TVec::size>;

inline void shuffle_test() {
  using namespace literals;
  static constexpr ShuffleIndices<4, 4> idxs0{.indices = {2_sh, 3_sh, 1_sh, zero_sh}};
  static constexpr ShuffleIndices<4, 4> idxs1{.indices = {2_sh, zero_sh, 0_sh, any_sh}};

  static constexpr auto ext0 = convert<2>(idxs0);
  static_assert(ext0->indices == std::array{4_sh, 5_sh, 6_sh, 7_sh, 2_sh, 3_sh, zero_sh, zero_sh});

  static constexpr auto sub0 = convert<8>(idxs0);
  static_assert(!sub0.has_value());
  static constexpr auto sub1 = convert<8>(idxs1);
  static_assert(sub1->indices == std::array{1_sh, 0_sh});
  static_assert(sub1->subzero);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_BASE_HPP
