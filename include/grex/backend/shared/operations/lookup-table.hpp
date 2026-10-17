// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOOKUP_TABLE_HPP

#include <array>
#include <cstddef>

#include "grex/backend/active/operations/load.hpp"
#include "grex/backend/active/operations/set.hpp"
#include "grex/backend/active/operations/shuffle.hpp"
#include "grex/backend/base.hpp"
#include "grex/backend/choosers.hpp"
#include "grex/base.hpp"

namespace grex::backend {
/** The SIMD component of a `grex::LookupTable` implemented based on `shuffle`. */
template<Vectorizable T, std::size_t N>
struct ShuffleLookupTable {
  using Vec = VectorFor<T, N>;

  ShuffleLookupTable() = default;
  explicit ShuffleLookupTable(const std::array<T, N>& table)
      : table_{load(table.data(), type_tag<Vec>)} {}

  template<UnsignedIntVector Idx>
  [[nodiscard, GREX_ALWAYS_INLINE]] VectorFor<T, size_of<Idx>> lookup(Idx idxs) const {
    return shuffle(table_, idxs);
  }

private:
  Vec table_ = zeros(type_tag<Vec>);
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_SHARED_OPERATIONS_LOOKUP_TABLE_HPP
