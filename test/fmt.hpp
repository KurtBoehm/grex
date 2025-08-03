// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_FMT_HPP
#define TEST_FMT_HPP

#include "fmt/base.h"

#include "grex/base/defs.hpp"

template<>
struct fmt::formatter<grex::ShuffleIndex> {
  static constexpr const char* parse(fmt::parse_context<>& ctx) {
    return ctx.begin();
  }
  static auto format(grex::ShuffleIndex sh, format_context& ctx) {
    switch (sh) {
      case grex::any_sh: return fmt::format_to(ctx.out(), "any_sh");
      case grex::zero_sh: return fmt::format_to(ctx.out(), "zero_sh");
      default: return fmt::format_to(ctx.out(), "{}_sh", int(sh));
    }
  }
};

template<>
struct fmt::formatter<grex::BlendZero> {
  static constexpr const char* parse(fmt::parse_context<>& ctx) {
    return ctx.begin();
  }
  static auto format(grex::BlendZero bz, format_context& ctx) {
    switch (bz) {
      case grex::zero_bz: return fmt::format_to(ctx.out(), "zero_bz");
      case grex::keep_bz: return fmt::format_to(ctx.out(), "keep_bz");
      case grex::any_bz: return fmt::format_to(ctx.out(), "any_bz");
      default: return fmt::format_to(ctx.out(), "???");
    }
  }
};

template<>
struct fmt::formatter<grex::BlendSelector> {
  static constexpr const char* parse(fmt::parse_context<>& ctx) {
    return ctx.begin();
  }
  static auto format(grex::BlendSelector bl, format_context& ctx) {
    switch (bl) {
      case grex::lhs_bl: return fmt::format_to(ctx.out(), "lhs_bl");
      case grex::rhs_bl: return fmt::format_to(ctx.out(), "rhs_bl");
      case grex::any_bl: return fmt::format_to(ctx.out(), "any_bl");
      default: return fmt::format_to(ctx.out(), "???");
    }
  }
};

#endif // TEST_FMT_HPP
