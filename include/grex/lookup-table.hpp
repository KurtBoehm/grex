// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_LOOKUP_TABLE_HPP

#include <array>
#include <cstddef>

#include "grex/backend/active/operations/lookup-table.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/base.hpp"
#include "grex/tags.hpp"

#if !GREX_BACKEND_SCALAR
#include <span>

#include "grex/operations-tagged.hpp"
#include "grex/types.hpp"
#endif

namespace grex {
template<Vectorizable T, std::size_t N>
struct LookupTable {
  using Index = UnsignedInt<sizeof(T)>;

  LookupTable() = default;
  explicit LookupTable(std::array<T, N> data) : data_{std::move(data)} {}

  [[nodiscard, GREX_ALWAYS_INLINE]] T lookup(std::size_t i) const {
    return data_[i];
  }
  template<UnsignedIntVectorizable Idx>
  [[nodiscard, GREX_ALWAYS_INLINE]] T lookup(Idx i, grex::AnyScalarTag auto /*vtag*/) const {
    return data_[i];
  }

#if !GREX_BACKEND_SCALAR
  template<UnsignedIntVectorizable Idx, std::size_t IN>
  [[nodiscard, GREX_ALWAYS_INLINE]] grex::Vector<T, IN> lookup(grex::Vector<Idx, IN> i,
                                                               grex::AnyVectorTag auto vtag) const {
    return grex::gather(std::span{data_}, i, vtag);
  }
#endif

private:
  std::array<T, N> data_{};
};

#if !GREX_BACKEND_SCALAR
template<Vectorizable T, std::size_t N>
requires(backend::LookupTable<T, N>::is_enabled)
struct LookupTable<T, N> {
  using Table = backend::LookupTable<T, N>;
  using Index = Table::Index;

  LookupTable() = default;
  explicit LookupTable(std::array<T, N> data) : data_{std::move(data)}, table_{data_} {}

  [[nodiscard, GREX_ALWAYS_INLINE]] T lookup(std::size_t i) const {
    return data_[i];
  }
  template<UnsignedIntVectorizable Idx>
  [[nodiscard, GREX_ALWAYS_INLINE]] T lookup(Idx i, grex::AnyScalarTag auto /*vtag*/) const {
    return data_[i];
  }

  template<UnsignedIntVectorizable Idx, std::size_t IN>
  [[nodiscard, GREX_ALWAYS_INLINE]] grex::Vector<T, IN> lookup(grex::Vector<Idx, IN> i,
                                                               grex::AnyVectorTag auto vtag) const {
    return vtag.mask(grex::Vector<T, IN>{table_.lookup(i.backend())});
  }

private:
  std::array<T, N> data_{};
  Table table_{};
};
#endif
} // namespace grex

#endif // INCLUDE_GREX_LOOKUP_TABLE_HPP
