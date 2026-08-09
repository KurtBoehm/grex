// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_FORMAT_FMTLIB_HPP
#define INCLUDE_GREX_FORMAT_FMTLIB_HPP

#include <fmt/base.h>
#include <fmt/ranges.h>

#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"
#include "grex/format/base.hpp"

#if !GREX_BACKEND_SCALAR
#include <array>
#include <cstddef>

#include "grex/types.hpp"

template<grex::Vectorizable T, std::size_t tSize>
struct fmt::formatter<grex::Vector<T, tSize>>
    : fmt::formatter<std::array<grex::format_impl::Formatted<T>, tSize>> {
  fmt::format_context::iterator format(grex::Vector<T, tSize> v, fmt::format_context& ctx) const {
    return fmt::formatter<std::array<grex::format_impl::Formatted<T>, tSize>>::format(
      grex::format_impl::formatted_array(v), ctx);
  }
};
template<grex::Vectorizable T, std::size_t tSize, typename TChar>
struct fmt::is_tuple_formattable<grex::Vector<T, tSize>, TChar> {
  static constexpr bool value = false;
};

template<grex::Vectorizable T, std::size_t tSize>
struct fmt::formatter<grex::Mask<T, tSize>> : fmt::formatter<std::array<bool, tSize>> {
  fmt::format_context::iterator format(grex::Mask<T, tSize> m, fmt::format_context& ctx) const {
    return fmt::formatter<std::array<bool, tSize>>::format(m.as_array(), ctx);
  }
};
template<grex::Vectorizable T, std::size_t tSize, typename TChar>
struct fmt::is_tuple_formattable<grex::Mask<T, tSize>, TChar> {
  static constexpr bool value = false;
};

template<>
struct fmt::formatter<grex::ShuffleIndex> {
  static constexpr const char* parse(fmt::parse_context<>& ctx) {
    return ctx.begin();
  }
  static auto format(grex::ShuffleIndex sh, fmt::format_context& ctx) {
    switch (sh) {
      case grex::any_sh: return fmt::format_to(ctx.out(), "any_sh");
      case grex::zero_sh: return fmt::format_to(ctx.out(), "zero_sh");
      default: return fmt::format_to(ctx.out(), "{}_sh", int(sh));
    }
  }
};
#endif

/**
 * Binary16 values are formatted as the binary32 values they are equal to.
 *
 * Both customization points are provided: `format_as` is the only one that `{fmt}` applies to types
 * that are not class types, which a native `_Float16` is not, while `format` is what the range
 * formatter uses for the elements of a range of binary16 values.
 */
template<>
struct fmt::formatter<grex::f16> : fmt::formatter<grex::f32> {
  static grex::f32 format_as(grex::f16 v) {
    return grex::f16_to_f32(v);
  }
  fmt::format_context::iterator format(grex::f16 v, fmt::format_context& ctx) const {
    return fmt::formatter<grex::f32>::format(grex::f16_to_f32(v), ctx);
  }
};

template<>
struct fmt::formatter<grex::BlendZeroSelector> {
  static constexpr const char* parse(fmt::parse_context<>& ctx) {
    return ctx.begin();
  }
  static auto format(grex::BlendZeroSelector bz, fmt::format_context& ctx) {
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
  static auto format(grex::BlendSelector bl, fmt::format_context& ctx) {
    switch (bl) {
      case grex::lhs_bl: return fmt::format_to(ctx.out(), "lhs_bl");
      case grex::rhs_bl: return fmt::format_to(ctx.out(), "rhs_bl");
      case grex::any_bl: return fmt::format_to(ctx.out(), "any_bl");
      default: return fmt::format_to(ctx.out(), "???");
    }
  }
};

#endif // INCLUDE_GREX_FORMAT_FMTLIB_HPP
