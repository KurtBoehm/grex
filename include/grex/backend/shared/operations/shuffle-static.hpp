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

#include "grex/backend/active/sizes.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/blend-static.hpp"
#include "grex/backend/shared/operations/blend-zero-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<std::size_t ValueBytes, std::size_t N>
struct ShuffleIndices {
  static constexpr std::size_t value_size = ValueBytes;
  static constexpr std::size_t size = N;
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
    return (is_index(sh) && is_in_lane(i, static_cast<u8>(sh))
              ? std::make_optional(static_cast<u8>(sh))
              : std::nullopt);
  }

  constexpr ShuffleIndex operator[](std::size_t i) const {
    return indices[i];
  }

  [[nodiscard]] constexpr bool requires_zeroing() const {
    return subzero ||
           static_apply<size>([&]<std::size_t... I> { return (... || (indices[I] == zero_sh)); });
  }

  [[nodiscard]] constexpr int imm8() const
  requires(size == 4)
  {
    return static_apply<size>([&]<std::size_t... I> {
      auto f = [](int i, ShuffleIndex sh) { return is_index(sh) ? static_cast<int>(sh) : i; };
      return (0 + ... + (f(I, indices[I]) << (2 * I)));
    });
  }

  template<bool SignedIdxs = true>
  [[nodiscard]] GREX_ALWAYS_INLINE auto vector(BoolTag<SignedIdxs> signed_idxs = {}) const {
    return static_apply<size>([&]<std::size_t... I> GREX_ALWAYS_INLINE {
      using Val = std::conditional_t<signed_idxs, SignedInt<value_size>, UnsignedInt<value_size>>;
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? Val(sh) : Val(-1); };
      return set(type_tag<NativeVector<Val, size>>, f(indices[I])...);
    });
  }

  [[nodiscard]] GREX_ALWAYS_INLINE auto mask() const {
    return static_apply<size>([&]<std::size_t... I> GREX_ALWAYS_INLINE {
      return set(type_tag<NativeMask<SignedInt<value_size>, size>>, is_index(indices[I])...);
    });
  }

  [[nodiscard]] constexpr auto laned_indices() const {
    using Val = SignedInt<value_size>;
    using Opt = std::optional<std::array<Val, size>>;

    if (!is_lane_local()) {
      return Opt{};
    }
    return static_apply<size>([&]<std::size_t... I> {
      auto f = [](ShuffleIndex sh) { return is_index(sh) ? Val(sh) : Val{-1}; };
      return Opt{std::array{f(indices[I])...}};
    });
  }

  [[nodiscard]] constexpr auto intralane_indices() const {
    return static_apply<size>([&]<std::size_t... I> {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        using Val = SignedInt<value_size>;
        return (is_index(sh) && is_in_lane(i, static_cast<u8>(sh))
                  ? static_cast<Val>(static_cast<u8>(sh))
                  : Val{-1});
      };
      return std::array{f(I, indices[I])...};
    });
  }
  [[nodiscard]] constexpr auto extralane_indices() const {
    static_assert(size == 2 * lane_size,
                  "This function is designed for 256-bit, i.e. two-laned, vectors!");
    using Val = SignedInt<value_size>;
    return static_apply<size>([&]<std::size_t... I> {
      auto f = [](std::size_t i, ShuffleIndex sh) {
        return (is_index(sh) && !is_in_lane(i, static_cast<u8>(sh))
                  ? static_cast<Val>(static_cast<u8>(sh))
                  : Val{-1});
      };
      return std::array{f(I, indices[I])...};
    });
  }

  template<std::size_t Segment>
  [[nodiscard]] constexpr bool is_segment_local() const {
    for (std::size_t i = 0; i < size; ++i) {
      const auto sh = indices[i];
      if (!is_index(sh)) {
        continue;
      }
      const auto idx = u8(sh);
      const auto lane_off = i / Segment * Segment;
      if (idx < lane_off || lane_off + Segment <= idx) {
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
    return static_apply<lane_size>([&]<std::size_t... I> {
      return ShuffleIndices<value_size, lane_size>{
        .indices = std::array{((I < size) ? indices[I] : any_sh)...},
        .subzero = subzero,
      };
    });
  }

  [[nodiscard]] constexpr Half half_raw(std::size_t half) const {
    return Half{
      .indices = static_apply<size / 2>(
        [&]<std::size_t... I> { return std::array{indices[I + half * size / 2]...}; }),
      .subzero = subzero,
    };
  }
  [[nodiscard]] constexpr Half half(std::size_t half) const {
    std::array<ShuffleIndex, size / 2> arr{};
    for (std::size_t i = 0; i < size / 2; ++i) {
      const ShuffleIndex sh = indices[i];
      if (is_index(sh) &&
          (static_cast<u8>(sh) < half * size / 2 || (half + 1) * size / 2 <= static_cast<u8>(sh))) {
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
        if (static_cast<u8>(sh) < index * size || (index + 1) * size <= static_cast<u8>(sh)) {
          return std::nullopt;
        }
        sh = ShuffleIndex(static_cast<u8>(sh) - index * size);
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
        sh = (index * size <= static_cast<u8>(sh) && static_cast<u8>(sh) < (index + 1) * size)
               ? ShuffleIndex(static_cast<u8>(sh) - index * size)
               : fallback;
      }
      arr[i] = sh;
    }
    return ShuffleIndices{.indices = arr, .subzero = subzero};
  }
  [[nodiscard]] constexpr Blend blend_vectors() const {
    auto f = [&](ShuffleIndex sh) {
      return (is_index(sh) && static_cast<u8>(sh) >= size) ? rhs_bl : lhs_bl;
    };
    const auto arr =
      static_apply<size>([&]<std::size_t... I> { return std::array{f(indices[I])...}; });
    return Blend{.ctrl = arr};
  }

  template<std::size_t Segment>
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, Segment>> repeated() const {
    static_assert(size >= Segment);
    if constexpr (size == Segment) {
      return *this;
    } else {
      if (!is_segment_local<Segment>()) {
        return std::nullopt;
      }
      std::array<ShuffleIndex, Segment> idxs = static_apply<Segment>(
        [&]<std::size_t... I> { return std::array<ShuffleIndex, Segment>{indices[I]...}; });
      bool subz = subzero;
      for (std::size_t i = Segment; i < size; ++i) {
        const ShuffleIndex sh = indices[i];
        ShuffleIndex& dst = idxs[i % Segment];
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
            const auto idx =
              ShuffleIndex{static_cast<u8>(static_cast<u8>(sh) - (i / Segment * Segment))};
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
      return ShuffleIndices<value_size, Segment>{.indices = idxs, .subzero = subz};
    }
  }
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, lane_size>> single_lane() const {
    return repeated<lane_size>();
  }
  [[nodiscard]] constexpr std::optional<ShuffleIndices<value_size, 2 * lane_size>>
  double_lane() const {
    return repeated<2 * lane_size>();
  }

  template<std::size_t DstValueBytes>
  friend constexpr std::optional<ShuffleIndices<DstValueBytes, N * ValueBytes / DstValueBytes>>
  convert(const ShuffleIndices& self) {
    constexpr auto dst_size = N * ValueBytes / DstValueBytes;
    using Dst = ShuffleIndices<DstValueBytes, dst_size>;

    if constexpr (DstValueBytes == ValueBytes) {
      return self;
    } else if constexpr (DstValueBytes < ValueBytes) {
      // simply multiply the entries with factor and add their chunk index
      constexpr auto factor = ValueBytes / DstValueBytes;
      auto f = [&](ShuffleIndex sh, std::size_t chunki) {
        if (is_index(sh)) {
          return ShuffleIndex(static_cast<u8>(sh) * factor + chunki);
        }
        return sh;
      };
      const auto idxs = static_apply<dst_size>(
        [&]<std::size_t... I> { return std::array{f(self.indices[I / factor], I % factor)...}; });
      return Dst{.indices = idxs, .subzero = self.subzero};
    } else {
      // check whether the indices in each chunk that is converted to one index
      // start at a multiple of `factor` and are ascending from there (apart from any/zero)
      constexpr auto factor = DstValueBytes / ValueBytes;
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
              const auto srci = static_cast<u8>(shi);
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

  [[nodiscard]] constexpr BlendZeroSelectors<ValueBytes, N> blend_zeros() const {
    auto f = [](ShuffleIndex sh) {
      switch (sh) {
        case any_sh: return any_bz;
        case zero_sh: return zero_bz;
        default: return keep_bz;
      }
    };
    return static_apply<N>(
      [&]<std::size_t... I> { return BlendZeroSelectors<ValueBytes, N>{f(indices[I])...}; });
  }
};
template<AnyVector Vec>
using ShuffleIndicesFor = ShuffleIndices<sizeof(typename Vec::Value), Vec::size>;

template<typename T>
struct AnyShuffleIndicesTrait : public std::false_type {};
template<std::size_t ValueBytes, std::size_t N>
struct AnyShuffleIndicesTrait<ShuffleIndices<ValueBytes, N>> : public std::true_type {};
template<typename T>
concept AnyShuffleIndices = AnyShuffleIndicesTrait<T>::value;

template<AnyShuffleIndices auto I>
struct ShufflerTrait;
template<AnyShuffleIndices auto I>
using Shuffler = ShufflerTrait<I>::Shuffler;

template<AnyShuffleIndices auto I>
struct PairShufflerTrait;
template<AnyShuffleIndices auto I>
using PairShuffler = PairShufflerTrait<I>::Shuffler;

template<ShuffleIndex... I, AnyVector Vec>
requires(Vec::size == sizeof...(I))
inline Vec shuffle(Vec vec) {
  static constexpr auto idxs = ShuffleIndicesFor<Vec>{.indices = {I...}};
  return Shuffler<idxs>::apply(vec, auto_tag<idxs>);
}
template<ShuffleIndex... I, AnyVector Vec>
requires(Vec::size == sizeof...(I))
inline Vec pair_shuffle(Vec a, Vec b) {
  static constexpr auto idxs = ShuffleIndicesFor<Vec>{.indices = {I...}};
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
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return static_apply<SI.size>(
      []<std::size_t... I> { return (... && (!is_index(SI[I]) || u8(SI[I]) == I)); });
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    static_assert(is_applicable(auto_tag<SI>));
    return ZeroBlender<SI.blend_zeros()>::apply(vec, auto_tag<SI.blend_zeros()>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    static_assert(is_applicable(auto_tag<SI>));
    return ZeroBlender<SI.blend_zeros()>::cost(auto_tag<SI>);
  }
};

struct SubShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  using Base = Shuffler<SI.sub_extended()>;

  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return Base<SI>::is_applicable(auto_tag<SI.sub_extended()>);
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    return Vec{Base<SI>::apply(vec.full, auto_tag<SI.sub_extended()>)};
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*tag*/) {
    return Base<SI>::cost(auto_tag<SI.sub_extended()>);
  }
};
template<AnyShuffleIndices auto SI>
requires((SI.value_size * SI.size < register_bytes.front())) // NOLINT(*-redundant-parentheses)
struct ShufflerTrait<SI> {
  using Shuffler = SubShuffler;
};

// A pair shuffler that just shuffles one of the vectors
struct PairShufflerSingle : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return SI.indices_in_vector(0).has_value() || SI.indices_in_vector(1).has_value();
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec a, Vec b, AutoTag<SI> /*tag*/) {
    static constexpr auto a_sh = SI.indices_in_vector(0);
    static constexpr auto b_sh = SI.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::apply(a, auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::apply(b, auto_tag<b_sh.value()>);
    }
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*tag*/) {
    constexpr auto a_sh = SI.indices_in_vector(0);
    constexpr auto b_sh = SI.indices_in_vector(1);

    if constexpr (a_sh.has_value()) {
      return Shuffler<a_sh.value()>::cost(auto_tag<a_sh.value()>);
    } else {
      return Shuffler<b_sh.value()>::cost(auto_tag<b_sh.value()>);
    }
  }
};
// A pair shuffler that performs two shuffles and then blends
struct PairShufflerBlend : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec a, Vec b, AutoTag<SI> /*tag*/) {
    static constexpr auto a_sh = SI.indices_in_vector_fallback(0, any_sh);
    static constexpr auto b_sh = SI.indices_in_vector_fallback(1, any_sh);

    return Blender<SI.blend_vectors()>::apply(Shuffler<a_sh>::apply(a, auto_tag<a_sh>),
                                              Shuffler<b_sh>::apply(b, auto_tag<b_sh>),
                                              auto_tag<SI.blend_vectors()>);
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*tag*/) {
    constexpr auto a_sh = SI.indices_in_vector_fallback(0, any_sh);
    constexpr auto b_sh = SI.indices_in_vector_fallback(1, any_sh);

    const auto [c00, c01] = Shuffler<a_sh>::cost(auto_tag<a_sh>);
    const auto [c10, c11] = Shuffler<b_sh>::cost(auto_tag<b_sh>);
    const auto [c20, c21] = Blender<SI.blend_vectors()>::cost(auto_tag<SI.blend_vectors()>);
    return {.inv_throughput = c00 + c10 + c20, .latency = c01 + c11 + c21};
  }
};
template<AnyShuffleIndices auto SI>
struct PairShufflerTrait {
  using Shuffler = CheapestType<SI, PairShufflerSingle, PairShufflerBlend>;
};

struct SuperShuffler : public BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    static constexpr auto lower_sh = SI.half_raw(0);
    static constexpr auto upper_sh = SI.half_raw(1);

    const auto lower = PairShuffler<lower_sh>::apply(vec.lower, vec.upper, auto_tag<lower_sh>);
    const auto upper = PairShuffler<upper_sh>::apply(vec.lower, vec.upper, auto_tag<upper_sh>);
    return Vec{.lower = lower, .upper = upper};
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*tag*/) {
    constexpr auto lower_sh = SI.half_raw(0);
    constexpr auto upper_sh = SI.half_raw(1);
    const auto [c0a, c1a] = PairShuffler<lower_sh>::cost(auto_tag<lower_sh>);
    const auto [c0b, c1b] = PairShuffler<upper_sh>::cost(auto_tag<upper_sh>);
    return {.inv_throughput = c0a + c0b, .latency = c1a + c1b};
  }
};

template<AnyShuffleIndices auto SI>
requires((SI.value_size * SI.size > register_bytes.back())) // NOLINT(*-redundant-parentheses)
struct ShufflerTrait<SI> {
  using Shuffler = SuperShuffler;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_SHUFFLE_STATIC_HPP
