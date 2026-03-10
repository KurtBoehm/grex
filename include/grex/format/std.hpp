// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_FORMAT_STD_HPP
#define INCLUDE_GREX_FORMAT_STD_HPP

#include <format>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#include <cstddef>

#include "grex/types.hpp"

template<grex::Vectorizable T, std::size_t tSize>
struct std::formatter<grex::Vector<T, tSize>> : public std::formatter<std::array<T, tSize>> {
  std::format_context::iterator format(grex::Vector<T, tSize> v, std::format_context& ctx) const {
    return formatter<std::array<T, tSize>>::format(v.as_array(), ctx);
  }
};
template<grex::Vectorizable T, std::size_t tSize>
struct std::formatter<grex::Mask<T, tSize>> : public std::formatter<std::array<bool, tSize>> {
  std::format_context::iterator format(grex::Mask<T, tSize> m, std::format_context& ctx) const {
    return formatter<std::array<bool, tSize>>::format(m.as_array(), ctx);
  }
};

template<>
struct std::formatter<grex::ShuffleIndex> {
  static constexpr const char* parse(std::format_parse_context& ctx) {
    return ctx.begin();
  }
  static auto format(grex::ShuffleIndex sh, std::format_context& ctx) {
    switch (sh) {
      case grex::any_sh: return std::format_to(ctx.out(), "any_sh");
      case grex::zero_sh: return std::format_to(ctx.out(), "zero_sh");
      default: return std::format_to(ctx.out(), "{}_sh", int(sh));
    }
  }
};
#endif

template<>
struct std::formatter<grex::BlendZeroSelector> {
  static constexpr const char* parse(std::format_parse_context& ctx) {
    return ctx.begin();
  }
  static auto format(grex::BlendZeroSelector bz, std::format_context& ctx) {
    switch (bz) {
      case grex::zero_bz: return std::format_to(ctx.out(), "zero_bz");
      case grex::keep_bz: return std::format_to(ctx.out(), "keep_bz");
      case grex::any_bz: return std::format_to(ctx.out(), "any_bz");
      default: return std::format_to(ctx.out(), "???");
    }
  }
};

template<>
struct std::formatter<grex::BlendSelector> {
  static constexpr const char* parse(std::format_parse_context& ctx) {
    return ctx.begin();
  }
  static auto format(grex::BlendSelector bl, std::format_context& ctx) {
    switch (bl) {
      case grex::lhs_bl: return std::format_to(ctx.out(), "lhs_bl");
      case grex::rhs_bl: return std::format_to(ctx.out(), "rhs_bl");
      case grex::any_bl: return std::format_to(ctx.out(), "any_bl");
      default: return std::format_to(ctx.out(), "???");
    }
  }
};

#endif // INCLUDE_GREX_FORMAT_STD_HPP
