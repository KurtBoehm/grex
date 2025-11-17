// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHUFFLE_STATIC_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHUFFLE_STATIC_HPP

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "grex/backend/defs.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/base/defs.hpp"

namespace grex::backend {
template<std::size_t tValueBytes, std::size_t tSize>
struct ShuffleIndices {
  static constexpr std::size_t value_size = tValueBytes;
  static constexpr std::size_t size = tSize;
  static constexpr std::size_t lane_size = 16 / value_size;
  using Half = ShuffleIndices<value_size, size / 2>;
  using Blend = BlendSelectors<value_size, size>;
  using BlendHalf = BlendSelectors<value_size, size / 2>;
  using Indices = std::array<ShuffleIndex, size>;

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

  [[nodiscard]] constexpr bool requires_zeroing() const {
    return subzero || static_apply<size>([&]<std::size_t... tIdxs>() {
             return (... || (indices[tIdxs] == zero_sh));
           });
  }

  [[nodiscard]] constexpr int imm8() const
  requires(size == 4)
  {
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](int i, ShuffleIndex sh) { return is_index(sh) ? int(sh) : i; };
      return (0 + ... + (f(tIdxs, indices[tIdxs]) << (2 * tIdxs)));
    });
  }

  [[nodiscard]] GREX_ALWAYS_INLINE auto vector(AnyBoolTag auto signed_idxs = true_tag) const {
    return static_apply<size>([&]<std::size_t... tIdxs>() GREX_ALWAYS_INLINE {
      using Val = std::conditional_t<signed_idxs, SignedInt<value_size>, UnsignedInt<value_size>>;
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? Val(sh) : Val(-1); };
      return set(type_tag<Vector<Val, size>>, f(indices[tIdxs])...);
    });
  }

  [[nodiscard]] GREX_ALWAYS_INLINE auto mask() const {
    return static_apply<size>([&]<std::size_t... tIdxs>() GREX_ALWAYS_INLINE {
      return set(type_tag<Mask<SignedInt<value_size>, size>>, is_index(indices[tIdxs])...);
    });
  }

  [[nodiscard]] constexpr auto laned_indices() const {
    using Val = SignedInt<value_size>;
    using Opt = std::optional<std::array<Val, size>>;

    if (!is_lane_local()) {
      return Opt{};
    }
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? Val(sh) : Val{-1}; };
      return Opt{std::array{f(indices[tIdxs])...}};
    });
  }

  [[nodiscard]] constexpr auto intralane_indices() const {
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        using Val = SignedInt<value_size>;
        return (is_index(sh) && is_in_lane(i, u8(sh)) ? Val(u8(sh)) : Val{-1});
      };
      return std::array{f(tIdxs, indices[tIdxs])...};
    });
  }
  [[nodiscard]] constexpr auto extralane_indices() const {
    static_assert(size == 2 * lane_size,
                  "This function is designed for 256-bit, i.e. two-laned, vectors!");
    using Val = SignedInt<value_size>;
    return static_apply<size>([&]<std::size_t... tIdxs>() {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        return (is_index(sh) && !is_in_lane(i, u8(sh)) ? Val(u8(sh)) : Val{-1});
      };
      return std::array{f(tIdxs, indices[tIdxs])...};
    });
  }

  template<std::size_t tSegment>
  [[nodiscard]] constexpr bool is_segment_local() const {
    for (std::size_t i = 0; i < size; ++i) {
      const auto sh = indices[i];
      if (!is_index(sh)) {
        continue;
      }
      const auto idx = u8(sh);
      const auto lane_off = i / tSegment * tSegment;
      if (idx < lane_off || lane_off + tSegment <= idx) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] constexpr bool is_lane_local() const {
    return is_segment_local<lane_size>();
  }
  [[nodiscard]] constexpr bool is_half_local() const {
    return is_segment_local<size / 2>();
  }

  [[nodiscard]] constexpr ShuffleIndices<value_size, lane_size> sub_extended() const
  requires(size < lane_size)
  {
    return static_apply<lane_size>([&]<std::size_t... tIdxs>() {
      return ShuffleIndices<value_size, lane_size>{
        .indices = std::array{((tIdxs < size) ? indices[tIdxs] : any_sh)...},
        .subzero = subzero,
      };
    });
  }

  [[nodiscard]] constexpr Half half_raw(std::size_t half) const {
    return Half{
      .indices = static_apply<size / 2>(
        [&]<std::size_t... tIdxs>() { return std::array{indices[tIdxs + half * size / 2]...}; }),
      .subzero = subzero,
    };
  }
  [[nodiscard]] constexpr Half half(std::size_t half) const {
    std::array<ShuffleIndex, size / 2> arr{};
    for (std::size_t i = 0; i < size / 2; ++i) {
      const ShuffleIndex sh = indices[i];
      if (is_index(sh) && (u8(sh) < half * size / 2 || (half + 1) * size / 2 <= u8(sh))) {
        return std::nullopt;
      }
      arr[i] = sh;
    }
    return Half{.indices = arr, .subzero = subzero};
  }

  // Returns the indices if all fall within [index * size, (index + 1) * size)
  [[nodiscard]] constexpr std::optional<ShuffleIndices> indices_in_vector(std::size_t index) const {
    Indices arr{};
    for (std::size_t i = 0; i < size; ++i) {
      ShuffleIndex sh = indices[i];
      if (is_index(sh)) {
        if (u8(sh) < index * size || (index + 1) * size <= u8(sh)) {
          return std::nullopt;
        }
        sh = ShuffleIndex(u8(sh) - index * size);
      }
      arr[i] = sh;
    }
    return ShuffleIndices{.indices = arr, .subzero = subzero};
  }
  // Returns indices that fall within [index * size, (index + 1) * size) and replaces others
  // with “any”
  [[nodiscard]] constexpr ShuffleIndices indices_in_vector_fallback(std::size_t index,
                                                                    ShuffleIndex fallback) const {
    Indices arr{};
    for (std::size_t i = 0; i < size; ++i) {
      ShuffleIndex sh = indices[i];
      if (is_index(sh)) {
        sh = (index * size <= u8(sh) && u8(sh) < (index + 1) * size)
               ? ShuffleIndex(u8(sh) - index * size)
               : fallback;
      }
      arr[i] = sh;
    }
    return ShuffleIndices{.indices = arr, .subzero = subzero};
  }
  [[nodiscard]] constexpr Blend blend_vectors() const {
    auto f = [&](ShuffleIndex sh) { return (is_index(sh) && u8(sh) >= size) ? rhs_bl : lhs_bl; };
    const auto arr =
      static_apply<size>([&]<std::size_t... tIdxs>() { return std::array{f(indices[tIdxs])...}; });
    return Blend{.ctrl = arr};
  }

  template<std::size_t tSegment>
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, tSegment>> repeated() const {
    static_assert(size >= tSegment);
    if constexpr (size == tSegment) {
      return *this;
    } else {
      if (!is_segment_local<tSegment>()) {
        return std::nullopt;
      }
      std::array<ShuffleIndex, tSegment> idxs = static_apply<tSegment>([&]<std::size_t... tIdxs>() {
        return std::array<ShuffleIndex, tSegment>{indices[tIdxs]...};
      });
      bool subz = subzero;
      for (std::size_t i = tSegment; i < size; ++i) {
        const ShuffleIndex sh = indices[i];
        ShuffleIndex& dst = idxs[i % tSegment];
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
            const auto idx = ShuffleIndex{u8(u8(sh) - (i / tSegment * tSegment))};
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
      return ShuffleIndices<value_size, tSegment>{.indices = idxs, .subzero = subz};
    }
  }
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, lane_size>> single_lane() const {
    return repeated<lane_size>();
  }
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, 2 * lane_size>>
  double_lane() const {
    return repeated<2 * lane_size>();
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
            case any_sh: break;
            case zero_sh: {
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
          idxs[i] = haszero ? zero_sh : any_sh;
        }
      }

      return Dst{.indices = idxs, .subzero = subz};
    }
  }

  [[nodiscard]] constexpr BlendZeros<tValueBytes, tSize> blend_zeros() const {
    auto f = [](ShuffleIndex sh) {
      switch (sh) {
        case any_sh: return any_bz;
        case zero_sh: return zero_bz;
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

template<AnyShuffleIndices auto tIdxs>
struct PairShufflerTrait;
template<AnyShuffleIndices auto tIdxs>
using PairShuffler = PairShufflerTrait<tIdxs>::Shuffler;

template<ShuffleIndex... tIdxs, AnyVector TVec>
requires(TVec::size == sizeof...(tIdxs))
inline TVec shuffle(TVec vec) {
  static constexpr auto idxs = ShuffleIndicesFor<TVec>{.indices = {tIdxs...}};
  return Shuffler<idxs>::apply(vec, auto_tag<idxs>);
}
template<ShuffleIndex... tIdxs, AnyVector TVec>
requires(TVec::size == sizeof...(tIdxs))
inline TVec pair_shuffle(TVec a, TVec b) {
  static constexpr auto idxs = ShuffleIndicesFor<TVec>{.indices = {tIdxs...}};
  return PairShuffler<idxs>::apply(a, b, auto_tag<idxs>);
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

struct SubShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  using Base = Shuffler<tSh.sub_extended()>;

  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return Base<tSh>::is_applicable(auto_tag<tSh.sub_extended()>);
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    return TVec{Base<tSh>::apply(vec.full, auto_tag<tSh.sub_extended()>)};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    return Base<tSh>::cost(auto_tag<tSh.sub_extended()>);
  }
};
template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size < register_bytes.front()))
struct ShufflerTrait<tSh> {
  using Shuffler = SubShuffler;
};

// A pair shuffler that just shuffles one of the vectors
struct PairShufflerSingle : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return tSh.indices_in_vector(0).has_value() || tSh.indices_in_vector(1).has_value();
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) {
    static constexpr auto a_sh = tSh.indices_in_vector(0);
    static constexpr auto b_sh = tSh.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::apply(a, auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::apply(b, auto_tag<b_sh.value()>);
    }
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto a_sh = tSh.indices_in_vector(0);
    constexpr auto b_sh = tSh.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::cost(auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::cost(auto_tag<b_sh.value()>);
    }
  }
};
// A pair shuffler that performs two shuffles and then blends
struct PairShufflerBlend : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec a, TVec b, AutoTag<tSh> /*tag*/) {
    static constexpr auto a_sh = tSh.indices_in_vector_fallback(0, any_sh);
    static constexpr auto b_sh = tSh.indices_in_vector_fallback(1, any_sh);

    return Blender<tSh.blend_vectors()>::apply(Shuffler<a_sh>::apply(a, auto_tag<a_sh>),
                                               Shuffler<b_sh>::apply(b, auto_tag<b_sh>),
                                               auto_tag<tSh.blend_vectors()>);
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto a_sh = tSh.indices_in_vector_fallback(0, any_sh);
    constexpr auto b_sh = tSh.indices_in_vector_fallback(1, any_sh);

    const auto [c00, c01] = Shuffler<a_sh>::cost(auto_tag<a_sh>);
    const auto [c10, c11] = Shuffler<b_sh>::cost(auto_tag<b_sh>);
    const auto [c20, c21] = Blender<tSh.blend_vectors()>::cost(auto_tag<tSh.blend_vectors()>);
    return std::make_pair(c00 + c10 + c20, c01 + c11 + c21);
  }
};
template<AnyShuffleIndices auto tSh>
struct PairShufflerTrait {
  using Shuffler = CheapestType<tSh, PairShufflerSingle, PairShufflerBlend>;
};

struct SuperShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto tSh>
  static constexpr bool is_applicable(AutoTag<tSh> /*tag*/) {
    return true;
  }
  template<AnyVector TVec, ShuffleIndicesFor<TVec> tSh>
  static TVec apply(TVec vec, AutoTag<tSh> /*tag*/) {
    static constexpr auto lower_sh = tSh.half_raw(0);
    static constexpr auto upper_sh = tSh.half_raw(1);

    const auto lower = PairShuffler<lower_sh>::apply(vec.lower, vec.upper, auto_tag<lower_sh>);
    const auto upper = PairShuffler<upper_sh>::apply(vec.lower, vec.upper, auto_tag<upper_sh>);
    return TVec{.lower = lower, .upper = upper};
  }
  template<AnyShuffleIndices auto tSh>
  static constexpr std::pair<f64, f64> cost(AutoTag<tSh> /*tag*/) {
    constexpr auto lower_sh = tSh.half_raw(0);
    constexpr auto upper_sh = tSh.half_raw(1);
    const auto [c0a, c1a] = PairShuffler<lower_sh>::cost(auto_tag<lower_sh>);
    const auto [c0b, c1b] = PairShuffler<upper_sh>::cost(auto_tag<upper_sh>);
    return {c0a + c0b, c1a + c1b};
  }
};

template<AnyShuffleIndices auto tSh>
requires((tSh.value_size * tSh.size > register_bytes.back()))
struct ShufflerTrait<tSh> {
  using Shuffler = SuperShuffler;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHUFFLE_STATIC_HPP
