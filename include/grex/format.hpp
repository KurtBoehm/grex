// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_FORMAT_HPP
#define INCLUDE_GREX_FORMAT_HPP

#include <cstddef>

#include <fmt/base.h>
#include <fmt/format.h>

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex {
template<typename TRange>
struct BaseFmt : public fmt::nested_formatter<typename TRange::Value> {
  auto format(const TRange& r, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      it = fmt::format_to(it, "[");
      static_apply<TRange::size>([&]<std::size_t... tIdxs>() {
        auto f = [&](auto i) {
          if constexpr (i > 0) {
            it = fmt::format_to(it, ", ");
          }
          it = fmt::format_to(it, "{}", this->nested(r.get(i)));
        };
        (..., f(index_tag<tIdxs>));
      });
      it = fmt::format_to(it, "]");
      return it;
    });
  }
};
} // namespace grex

template<grex::Vectorizable T, std::size_t tSize>
struct fmt::formatter<grex::Vector<T, tSize>> : public grex::BaseFmt<grex::Vector<T, tSize>> {};
template<grex::Vectorizable T, std::size_t tSize>
struct fmt::formatter<grex::Mask<T, tSize>> : public grex::BaseFmt<grex::Mask<T, tSize>> {};

#endif // INCLUDE_GREX_FORMAT_HPP
