// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP

#include <cstddef>

#include "grex/backend/shared/operations/lookup-table.hpp"
#include "grex/base.hpp"

namespace grex::backend {
template<Vectorizable T, std::size_t N>
struct LookupTable {
  static constexpr bool is_enabled = false;
};

/** If the table is at most twice the native size, use `shuffle`. */
template<Vectorizable T, std::size_t N>
requires(N <= 2 * max_native_size<T>)
struct LookupTable<T, N> : ShuffleLookupTable<T, N> {
  using Index = u8;
  static constexpr bool is_enabled = true;

  using ShuffleLookupTable<T, N>::ShuffleLookupTable;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_LOOKUP_TABLE_HPP
