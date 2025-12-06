// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_LOOKUP_TABLE_HPP
#define INCLUDE_GREX_LOOKUP_TABLE_HPP

#include <array>
#include <cstddef>
#include <span>

#include "grex/base.hpp"
#include "grex/operations-tagged.hpp"
#include "grex/tags.hpp"
#include "grex/types.hpp"

namespace grex {
template<Vectorizable T, std::size_t tSize>
struct LookupTable {
  LookupTable() = default;
  explicit LookupTable(std::array<T, tSize> data) : data_{std::move(data)} {}

  T lookup(std::size_t i) const {
    return data_[i];
  }
  template<UnsignedIntVectorizable TIdx>
  T lookup(TIdx i, grex::AnyScalarTag auto /*vtag*/) const {
    return data_[i];
  }

  template<UnsignedIntVectorizable TIdx, std::size_t tVecSize>
  grex::Vector<T, tVecSize> lookup(grex::Vector<TIdx, tVecSize> i,
                                   grex::AnyVectorTag auto vtag) const {
    return grex::gather(std::span{data_}, i, vtag);
  }

private:
  std::array<T, tSize> data_{};
};

template<Vectorizable T, std::size_t tSize>
requires(sizeof(T) * tSize <= 64)
struct LookupTable<T, tSize> {
  using VectorData = grex::Vector<T, tSize>;

  LookupTable() = default;
  explicit LookupTable(std::array<T, tSize> data)
      : data_{std::move(data)}, vdata_{VectorData::load(data_.data())} {}

  T lookup(std::size_t i) const {
    return data_[i];
  }
  template<UnsignedIntVectorizable TIdx>
  T lookup(TIdx i, grex::AnyScalarTag auto /*vtag*/) const {
    return data_[i];
  }

  template<UnsignedIntVectorizable TIdx, std::size_t tVecSize>
  grex::Vector<T, tVecSize> lookup(grex::Vector<TIdx, tVecSize> i,
                                   grex::AnyVectorTag auto vtag) const {
    return vtag.mask(grex::shuffle(vdata_, i));
  }

private:
  std::array<T, tSize> data_{};
  VectorData vdata_ = VectorData::zeros();
};
} // namespace grex

#endif // INCLUDE_GREX_LOOKUP_TABLE_HPP
