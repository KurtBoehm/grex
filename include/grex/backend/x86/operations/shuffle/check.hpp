// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_CHECK_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_CHECK_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <utility>

#include "thesauros/ranges.hpp"
#include "thesauros/static-ranges.hpp"

#include "grex/base/defs.hpp"

namespace grex::backend {
enum struct OptZeroing : u8 { none, no_zeroing, zeroing };
inline constexpr OptZeroing none_oz = OptZeroing::none;
inline constexpr OptZeroing no_zeroing_oz = OptZeroing::no_zeroing;
inline constexpr OptZeroing zeroing_oz = OptZeroing::zeroing;

struct Rotate {
  std::size_t rot;
  bool rotzeroing;
  OptZeroing shleft;
  OptZeroing shright;

  constexpr bool operator==(const Rotate&) const = default;
};

template<std::size_t tElementBytes, std::size_t tSize>
struct ShuffleChecker {
  static constexpr std::size_t element_bytes = tElementBytes;
  static constexpr std::size_t size = tSize;
  static constexpr std::size_t lane_num = element_bytes * tSize / 16;
  static constexpr std::size_t lane_size = tSize / lane_num;
  using Sh = ShuffleIndex;
  using LanePattern = std::array<Sh, lane_size>;

private:
  static constexpr auto star_and = thes::star::left_reduce(std::logical_and{}, true);
  static constexpr auto star_or = thes::star::left_reduce(std::logical_or{}, false);
  [[nodiscard]] constexpr bool indices_check(auto f, auto r) const {
    return indices | thes::star::transform(f) | r;
  };
  [[nodiscard]] constexpr auto index_trans(auto f) const {
    return thes::star::index_transform<0, size>(
      [this, f](auto i) { return f(i, std::get<i>(indices)); });
  };

public:
  std::array<ShuffleIndex, tSize> indices;

  // check if there is an out-of-range index
  [[nodiscard]] constexpr bool out_of_range() const {
    return indices_check([](Sh sh) { return is_index(sh) && u8(sh) >= size; }, star_or);
  }

  // check if everything is “irrelevant” or retains the previous index: no-op
  [[nodiscard]] constexpr bool noop() const {
    return index_trans([&](auto i, Sh sh) { return sh == any_sh || std::size_t(sh) == i; }) |
           star_and;
  }

  // check if everything is “irrelevant” or “zero”: everything can be zero
  [[nodiscard]] constexpr bool all_zero() const {
    return indices_check([](Sh sh) { return !is_index(sh); }, star_and);
  }

  // check if a permutation is required
  [[nodiscard]] constexpr bool permute() const {
    return index_trans([&](auto i, Sh sh) { return !is_index(sh) || std::size_t(sh) != i; }) |
           star_or;
  }

  // check if zeroing is required
  [[nodiscard]] constexpr bool zeroing() const {
    return indices_check([](Sh sh) { return sh == zero_sh; }, star_or);
  }

  // check if there is a single element that can be broadcast
  // returns (index of the element to be broadcast, whether additional zeroing is required)
  // this function assume that all_zero is false
  [[nodiscard]] constexpr std::optional<std::pair<u8, bool>> broadcast() const {
    const auto ish = *(indices | std::views::filter([](Sh sh) { return is_index(sh); })).begin();
    if (indices_check([ish](Sh sh) { return !is_index(sh) || sh == ish; }, star_and)) {
      return std::make_pair(u8(ish), zeroing());
    }
    return std::nullopt;
  }

  // check if a permutation with larger blocks is possible,
  // potentially with additional zeroing afterwards
  [[nodiscard]] constexpr OptZeroing large_block() const {
    const auto large_valid = thes::star::index_transform<tSize / 2>([&](auto i) {
      const auto sh0 = indices[2 * i];
      const auto sh1 = indices[2 * i + 1];
      return !is_index(sh0) || (u8(sh0) % 2 == 0 && (!is_index(sh1) || u8(sh1) == u8(sh0) + 1));
    });
    if (large_valid | star_and) {
      const auto addz = thes::star::index_transform<tSize / 2>([&](auto i) {
        const auto sh0 = indices[2 * i];
        const auto sh1 = indices[2 * i + 1];
        return (sh0 == zero_sh && is_index(sh1)) || (is_index(sh0) && sh1 == zero_sh);
      });
      return (addz | star_or) ? zeroing_oz : no_zeroing_oz;
    }
    return none_oz;
  }

  // check if the permutation crosses lanes
  [[nodiscard]] constexpr bool cross_lane() const {
    auto op = [](auto i, Sh sh) {
      return is_index(sh) && i / lane_size != std::size_t(sh) / lane_size;
    };
    return index_trans(op) | star_or;
  }

  // check if the permutation is the same in each lane when treating “zero” like “any”,
  // which might lead to additional zeroing being required (which is comparatively cheap)
  [[nodiscard]] constexpr std::optional<LanePattern> same_pattern() const {
    if (cross_lane()) {
      // part of the permutation crosses lanes, i.e. the pattern cannot be the same in each lane
      return std::nullopt;
    }
    LanePattern pattern{};
    for (std::size_t i = 0; i < lane_size; ++i) {
      Sh ip = any_sh;
      for (std::size_t lane = 0; lane < lane_num; ++lane) {
        const Sh ix = indices[lane * lane_size + i];
        if (is_index(ix)) {
          // this subtraction is valid because “cross_lane” is false
          const auto ux = u8(ix) - lane * lane_size;
          if (is_index(ip) && u8(ip) != ux) {
            // a different index was already stored
            return std::nullopt;
          }
          ip = Sh(ux);
        }
      }
      pattern[i] = ip;
    }
    return pattern;
  }

  // check if the permutation can be realized by zero-extending the lower half
  [[nodiscard]] constexpr OptZeroing zext() const {
    if (index_trans([](auto i, Sh sh) { return !is_index(sh) || std::size_t(sh) * 2 == i; }) |
        star_and) {
      const auto addz = index_trans([](auto i, Sh sh) { return i % 2 == 0 && sh == zero_sh; });
      return (addz | star_or) ? zeroing_oz : no_zeroing_oz;
    }
    return none_oz;
  }

  [[nodiscard]] constexpr OptZeroing compress() const {
    // the lower bound for the next index
    u8 lb = 0;
    // when there is a zero index, that only requires zeroing if something non-zero comes after
    // potentially zeroing
    bool pzeroing = false;
    // definitely zeroing
    bool zeroing = false;
    for (const std::size_t i : thes::range(size)) {
      const Sh ix = indices[i];
      if (is_index(ix)) {
        const auto ux = u8(ix);
        // check whether the index is greater than the last actual index
        // + the number of non-index values in between, which corresponds to lb - 1
        if (ux < lb) {
          // The index is too small, compress cannot be used
          return none_oz;
        }
        // if potential zeroing has occurred before, it is now clear that zeroing is required
        // due to the following non-zero value
        zeroing = zeroing || pzeroing;
        // the index is valid, update the lower bound for the next index
        lb = ux + 1;
      } else {
        // ix is zero or any, which requires an arbitrary value at this index
        ++lb;
        // if ix is zero, zeroing is required if another non-zero value follows
        pzeroing = pzeroing || ix == zero_sh;
      }
    }
    return zeroing ? zeroing_oz : no_zeroing_oz;
  }

  [[nodiscard]] constexpr OptZeroing expand() const {
    // lower bound: the last actual index + 1
    u8 lb = 0;
    // upper bound: the last actual index + the number of non-indices since then
    u8 ub = 0;
    // the number of “any” elements since the last actual index
    u8 anum = 0;
    // whether explicit zeroing is necessary
    bool zeroing = false;
    for (const std::size_t i : thes::range(size)) {
      const Sh ix = indices[i];
      if (is_index(ix)) {
        const auto ux = u8(ix);
        // check whether the index is at least the last actual index + 1
        // and no more than that number + the number non-index elements in between
        // (which could be filled with dummy values)
        if (ux < lb || ux > ub) {
          // if that is not the case, expand cannot be used
          return none_oz;
        }
        // force explicit zeroing if the number of dummy values in the range since
        // the last actual index is greater than the number of “any” element in this range,
        // i.e. some of the dummy values have to be zeroed explicitly
        zeroing = zeroing || (ux - lb > anum);
        // The next actual index has to be ux + 1 for expand to work
        lb = ub = ux + 1;
        // reset number of “any” elements
        anum = 0;
      } else {
        // ix is zero or any, increment the upper bound to allow a dummy to be placed here
        ++ub;
        // increment the number of “any” elements if appropriate
        anum += u8(ix == any_sh);
      }
    }
    return zeroing ? zeroing_oz : no_zeroing_oz;
  }

  // if applicable, the number of bytes by which the vector needs to be rotated
  // to achieve the desired permutation and whether zeroing is required
  // this function assume that all_zero is false
  [[nodiscard]] constexpr std::optional<Rotate> rotate() const {
    const std::optional<LanePattern> opattern = same_pattern();
    if (!opattern.has_value()) {
      return std::nullopt;
    }
    // the lane pattern
    const LanePattern pattern = opattern.value();
    // the amount of rotating required
    std::optional<std::size_t> orot{};
    for (const std::size_t i : thes::range(lane_size)) {
      const Sh ix = pattern[i];
      if (!is_index(ix)) {
        continue;
      }
      const auto ux = u8(ix);
      // The amount of rightward rotation is ux - i
      const std::size_t a = std::size_t(ux + lane_size - i) % lane_size;
      if (orot.has_value() && orot != a) {
        // a different amount has already been determined, i.e. rotation is not possible
        return std::nullopt;
      }
      // either the same or no amount is stored, updating is safe in either case
      orot = a;
    }
    // rotation can only be empty if there are no actual indices, which we assume not to be the case
    const std::size_t rot = orot.value();

    // check whether a left shift is possible
    const std::size_t lrot = (lane_size - rot) % lane_size;
    OptZeroing shleft = no_zeroing_oz;
    for (const std::size_t i : thes::range(lrot)) {
      // check if first rot elements are “any” or “zero”
      if (is_index(pattern[i])) {
        shleft = none_oz;
      }
    }
    if (shleft != none_oz) {
      for (const std::size_t i : thes::range(lrot, lane_size)) {
        if (pattern[i] == zero_sh) {
          // explicit zeroing needed
          shleft = zeroing_oz;
        }
      }
    }

    // check whether a right shift is possible
    OptZeroing shright = no_zeroing_oz;
    for (const std::size_t i : thes::range(lane_size - rot, lane_size)) {
      // check if last (lane_size - rot) elements are “any” or “zero”
      if (is_index(pattern[i])) {
        shright = none_oz;
      }
    }
    if (shright != none_oz) {
      for (const std::size_t i : thes::range(lane_size - rot)) {
        if (pattern[i] == zero_sh) {
          // explicit zeroing needed
          shright = zeroing_oz;
        }
      }
    }

    return Rotate{.rot = rot, .rotzeroing = zeroing(), .shleft = shleft, .shright = shright};
  }

private:
  // ixop should return true iff the pattern is not invalidated
  [[nodiscard]] constexpr bool laneop(auto ixop) const {
    const std::optional<LanePattern> opattern = same_pattern();
    if (!opattern.has_value()) {
      return false;
    }
    // the lane pattern
    const LanePattern pattern = opattern.value();

    for (const std::size_t i : thes::range(lane_size)) {
      const Sh ix = pattern[i];
      if (!ixop(i, ix)) {
        return false;
      }
    }
    return true;
  }

public:
  // check whether pairs of adjacent elements can just be swapped
  // zeroing required when “zeroing” is true
  [[nodiscard]] constexpr bool swap() const {
    return laneop([](const std::size_t i, Sh ix) { return !is_index(ix) || u8(ix) == (i ^ 1U); });
  }
  // check whether the most significant elements in each lane can be duplicated to use unpackhi
  // zeroing required when “zeroing” is true
  [[nodiscard]] constexpr bool unpackhi() const {
    return laneop(
      [](const std::size_t i, Sh ix) { return !is_index(ix) || u8(ix) == (lane_size + i) / 2; });
  }
  // check whether the least significant elements in each lane can be duplicated to use unpacklo
  // zeroing required when “zeroing” is true
  [[nodiscard]] constexpr bool unpacklo() const {
    return laneop([](const std::size_t i, Sh ix) { return !is_index(ix) || u8(ix) == i / 2; });
  }

  // zeroing required when “zeroing” is true
  [[nodiscard]] constexpr std::optional<u32> pshufd() const {
    if constexpr (element_bytes < 4) {
      return std::nullopt;
    }

    const std::optional<LanePattern> opattern = same_pattern();
    if (!opattern.has_value()) {
      return std::nullopt;
    }
    // the lane pattern
    const LanePattern pattern = opattern.value();
    u32 ctrl{};

    for (const std::size_t i : thes::range(lane_size)) {
      const Sh ix = pattern[i];
      if constexpr (element_bytes == 4) {
        ctrl |= (u32(ix) % 4) << (2 * i);
      }
      if constexpr (element_bytes == 8) {
        // if ix == 0, this is 4 = 0b0100, if ix == 1, it is 14 = 0b1110
        ctrl |= ((u32(ix) % 2) * 10 + 4) << (4 * i);
      }
    }
    return ctrl;
  }

  // zeroing required when “zeroing” is true
  [[nodiscard]] constexpr std::optional<std::size_t> rotate_big() const {
    // the same logic as in “rotate”, but using the overall indices and size instead those
    // of each lane
    std::optional<std::size_t> rot{};
    for (const std::size_t i : thes::range(size)) {
      const Sh ix = indices[i];
      if (!is_index(ix)) {
        continue;
      }
      const auto ux = u8(ix);
      const std::size_t a = std::size_t(ux + size - i) % size;
      if (rot.has_value() && rot != a) {
        return std::nullopt;
      }
      rot = a;
    }
    return rot;
  }
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_SHUFFLE_CHECK_HPP
