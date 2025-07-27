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
  static constexpr std::size_t lane_size = 16 / value_size;
  using Indices = std::array<ShuffleIndex, size>;
  using SignedVal = SignedInt<value_size>;
  using SignedArr = std::array<SignedVal, size>;
  using SignedVec = Vector<SignedVal, size>;

  Indices indices;
  // whether partial entries need to be zeroed
  bool subzero = false;

  static constexpr bool is_in_lane(std::size_t i, u8 idx) {
    const std::size_t lane_off = i / lane_size * lane_size;
    return lane_off <= idx && idx < lane_off + lane_size;
  }
  static constexpr std::optional<u8> index_in_lane(std::size_t i, ShuffleIndex sh) {
    return (is_index(sh) && is_in_lane(i, u8(sh)) ? std::make_optional(u8(sh)) : std::nullopt);
  }

  constexpr ShuffleIndex operator[](std::size_t i) const {
    return indices[i];
  }

  [[nodiscard]] constexpr int imm8() const
  requires(size == 4)
  {
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](int i, ShuffleIndex sh) { return is_index(sh) ? int(sh) : i; };
      return (0 + ... + (f(tIdxs, indices[tIdxs]) << (2 * tIdxs)));
    });
  }

  [[nodiscard]] SignedVec vector() const {
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? SignedVal{u8(sh)} : SignedVal{-1}; };
      return set(type_tag<SignedVec>, f(indices[tIdxs])...);
    });
  }

  [[nodiscard]] constexpr std::optional<SignedArr> laned_indices() const {
    if (!is_lane_local()) {
      return std::nullopt;
    }
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? SignedVal(sh) : SignedVal{-1}; };
      return std::array{f(indices[tIdxs])...};
    });
  }

  [[nodiscard]] constexpr SignedArr intralane_indices() const {
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        return (is_index(sh) && is_in_lane(i, u8(sh)) ? SignedVal(u8(sh)) : SignedVal{-1});
      };
      return std::array{f(tIdxs, indices[tIdxs])...};
    });
  }
  [[nodiscard]] constexpr SignedArr extralane_indices() const {
    static_assert(size == 2 * lane_size,
                  "This function is designed for 256-bit, i.e. two-laned, vectors!");
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        return (is_index(sh) && !is_in_lane(i, u8(sh)) ? SignedVal(u8(sh)) : SignedVal{-1});
      };
      return std::array{f(tIdxs, indices[tIdxs])...};
    });
  }

  [[nodiscard]] constexpr bool is_lane_local() const {
    for (std::size_t i = 0; i < size; ++i) {
      const auto sh = indices[i];
      if (!is_index(sh)) {
        continue;
      }
      const auto idx = u8(sh);
      const auto lane = i / lane_size;
      if (idx < lane * lane_size || (lane + 1) * lane_size <= idx) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, lane_size>> single_lane() const {
    if constexpr (size == lane_size) {
      return *this;
    } else {
      if (!is_lane_local()) {
        return std::nullopt;
      }
      std::array<ShuffleIndex, lane_size> idxs =
        static_apply<lane_size>([&]<std::size_t... tIdxs>() {
          return std::array<ShuffleIndex, lane_size>{indices[tIdxs]...};
        });
      bool subz = subzero;
      for (std::size_t i = lane_size; i < size; ++i) {
        const ShuffleIndex sh = indices[i];
        ShuffleIndex& dst = idxs[i % lane_size];
        switch (sh) {
          case any_sh: break;
          case zero_sh: {
            switch (dst) {
              case any_sh: {
                dst = zero_sh;
                break;
              }
              case zero_sh: {
                break;
              }
              default: {
                subz = true;
                break;
              }
            }
            break;
          }
          default: {
            const auto idx = ShuffleIndex{u8(u8(sh) - (i / lane_size * lane_size))};
            switch (dst) {
              case any_sh: {
                dst = idx;
                break;
              }
              case zero_sh: {
                subz = true;
                dst = idx;
                break;
              }
              default: {
                if (dst != idx) {
                  return std::nullopt;
                }
                break;
              }
            }
          }
        }
      }
      return ShuffleIndices<value_size, lane_size>{.indices = idxs, .subzero = subz};
    }
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
      return Dst{.indices = idxs, .subzero = self.subzero};
    } else {
      // check whether the indices in each chunk that is converted to one index
      // start at a multiple of `factor` and are ascending from there (apart from any/zero)
      constexpr auto factor = tDstValueBytes / tValueBytes;
      std::array<ShuffleIndex, dst_size> idxs{};
      bool subz = self.subzero;
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

      return Dst{.indices = idxs, .subzero = subz};
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
template<AnyVector TVec>
using ShuffleIndicesFor = ShuffleIndices<sizeof(typename TVec::Value), TVec::size>;

template<typename T>
struct AnyShuffleIndicesTrait : public std::false_type {};
template<std::size_t tValueBytes, std::size_t tSize>
struct AnyShuffleIndicesTrait<ShuffleIndices<tValueBytes, tSize>> : public std::true_type {};
template<typename T>
concept AnyShuffleIndices = AnyShuffleIndicesTrait<T>::value;

template<AnyShuffleIndices auto tIdxs>
struct ShufflerTrait;
template<AnyShuffleIndices auto tIdxs>
using Shuffler = ShufflerTrait<tIdxs>::Shuffler;

template<ShuffleIndex... tIdxs, AnyVector TVec>
requires(TVec::size == sizeof...(tIdxs))
inline TVec shuffle(TVec vec) {
  static constexpr auto idxs = ShuffleIndicesFor<TVec>{.indices = {tIdxs...}};
  return Shuffler<idxs>::apply(vec, auto_tag<idxs>);
}

inline void shuffle_test() {
  using namespace literals;
  static constexpr ShuffleIndices<4, 4> idxs0{.indices = {2_sh, 3_sh, 1_sh, zero_sh}};
  static constexpr ShuffleIndices<4, 4> idxs1{.indices = {2_sh, zero_sh, 0_sh, any_sh}};
  static constexpr ShuffleIndices<8, 4> idxs2{.indices = {2_sh, 0_sh, 2_sh, zero_sh}};
  static constexpr ShuffleIndices<8, 4> idxs3{.indices = {1_sh, 0_sh, 3_sh, zero_sh}};

  static constexpr auto ext0 = convert<2>(idxs0);
  static_assert(ext0->indices == std::array{4_sh, 5_sh, 6_sh, 7_sh, 2_sh, 3_sh, zero_sh, zero_sh});

  static constexpr auto sub0 = convert<8>(idxs0);
  static_assert(!sub0.has_value());
  static constexpr auto sub1 = convert<8>(idxs1);
  static_assert(sub1->indices == std::array{1_sh, 0_sh});
  static_assert(sub1->subzero);

  static constexpr auto sin2 = idxs2.single_lane();
  static_assert(!sin2.has_value());

  static constexpr auto sin3 = idxs3.single_lane();
  static_assert(sin3.has_value());
  static_assert(sin3->indices == std::array{1_sh, 0_sh});
  static_assert(sin3->subzero);
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_STATIC_BASE_HPP
