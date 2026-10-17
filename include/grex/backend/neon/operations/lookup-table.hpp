// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOOKUP_TABLE_HPP

#include <array>
#include <cstddef>

#include <arm_neon.h>

#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/backend/neon/operations/interleave.hpp"
#include "grex/backend/shared/operations/lookup-table.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
struct LookupTable {
  static constexpr bool is_enabled = false;
};

/** If the table consists of at most two native vectors, use `shuffle`. */
template<Vectorizable T, std::size_t N>
requires(N <= 2 * max_native_size<T>)
struct LookupTable<T, N> : ShuffleLookupTable<T, N> {
  using Index = u8;
  static constexpr bool is_enabled = true;

  using ShuffleLookupTable<T, N>::ShuffleLookupTable;
};

/**
 * The SIMD component of a `grex::LookupTable` implemented based on 16-bit planes: The values are
 * de-interleaved into `B = sizeof(T) / 2` `u16` in the constructor so that `B` two-table `TBL`
 * instructions can be applied to the planes independently, with the results only needing to be
 * interleaved again.
 */
template<Vectorizable T, std::size_t N>
requires(N == 16 && sizeof(T) > 2)
struct LookupTable<T, N> {
  using UnValue = UnsignedInt<sizeof(T)>;
  using Planes = std::array<VectorFor<u16, 16>, sizeof(T) / 2>;
  using Index = u16;
  static constexpr bool is_enabled = true;

  LookupTable() = default;
  explicit LookupTable(const std::array<T, N>& table)
      : table_{deinterleave<u16>(as<UnValue>(load(table.data(), type_tag<VectorFor<T, N>>)))} {}

  template<UnsignedIntVector Idx>
  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, size_of<Idx>> lookup(Idx idxs) const {
    return static_apply<sizeof(T) / 2>(
      [&]<std::size_t... I> { return as<T>(interleave(shuffle(std::get<I>(table_), idxs)...)); });
  }

private:
  Planes table_{};
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_LOOKUP_TABLE_HPP
