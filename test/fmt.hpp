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

#endif // TEST_FMT_HPP
