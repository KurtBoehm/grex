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
