// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TAGS_HPP
#define INCLUDE_GREX_TAGS_HPP

#include <cstddef>

#include "grex/base/defs.hpp"
#include "grex/types.hpp"

namespace grex::tag {
/////////////////
// scalar tags //
/////////////////

template<Vectorizable T>
struct TypedScalar;

struct Scalar {
  using Whole = Scalar;
  static constexpr std::size_t size = 1;

  template<Vectorizable T>
  [[nodiscard]] TypedScalar<T> instantiate(TypeTag<T> /*tag*/) const;
  template<Vectorizable T>
  [[nodiscard]] TypedScalar<T> cast(TypeTag<T> /*tag*/) const;

  template<typename T>
  [[nodiscard]] constexpr T mask(T x) const {
    return x;
  }
};

template<Vectorizable T>
struct TypedScalar : public Scalar {
  using Value = T;
};

template<Vectorizable T>
[[nodiscard]] TypedScalar<T> Scalar::instantiate(TypeTag<T> /*tag*/) const {
  return {};
}
template<Vectorizable T>
[[nodiscard]] TypedScalar<T> Scalar::cast(TypeTag<T> /*tag*/) const {
  return {};
}

///////////////
// full tags //
///////////////

template<Vectorizable T, std::size_t tSize>
struct TypedFull;

template<std::size_t tSize>
struct Full {
  using Whole = Full;
  static constexpr std::size_t size = tSize;

  template<Vectorizable T>
  [[nodiscard]] TypedFull<T, tSize> instantiate(TypeTag<T> /*tag*/) const;
  template<Vectorizable T>
  [[nodiscard]] TypedFull<T, tSize> cast(TypeTag<T> /*tag*/) const;

  template<SizedVector<size> TVec>
  [[nodiscard]] TVec mask(TVec v) const {
    return v;
  }
  template<SizedMask<size> TMask>
  [[nodiscard]] TMask mask(TMask m) const {
    return m;
  }

  [[nodiscard]] std::size_t part() const {
    return size;
  }
};

template<Vectorizable T, std::size_t tSize>
struct TypedFull : public Full<tSize> {
  using Value = T;
};

template<std::size_t tSize>
template<Vectorizable T>
[[nodiscard]] TypedFull<T, tSize> Full<tSize>::instantiate(TypeTag<T> /*tag*/) const {
  return {};
}
template<std::size_t tSize>
template<Vectorizable T>
[[nodiscard]] TypedFull<T, tSize> Full<tSize>::cast(TypeTag<T> /*tag*/) const {
  return {};
}

////////////////////////////////////
// partial tags incl. masked tags //
////////////////////////////////////

template<Vectorizable T, std::size_t tSize>
struct TypedMasked {
  using Value = T;
  using Whole = TypedFull<T, tSize>;
  static constexpr std::size_t size = tSize;

  explicit TypedMasked(Mask<Value, tSize> mask) : mask_(mask) {}

  template<Vectorizable TOther>
  [[nodiscard]] TypedMasked<TOther, tSize> cast(TypeTag<TOther> /*tag*/) const {
    return {mask_};
  }

  [[nodiscard]] Vector<T, tSize> mask(Vector<T, tSize> v) const {
    return blend_zero(mask_, v);
  }
  [[nodiscard]] Mask<Value, tSize> mask(Mask<Value, tSize> m) const {
    return mask_ && m;
  }

  [[nodiscard]] Mask<Value, tSize> mask() const {
    return mask_;
  }

private:
  Mask<Value, tSize> mask_;
};

template<std::size_t tSize>
struct Part {
  using Whole = Full<tSize>;
  static constexpr std::size_t size = tSize;

  explicit constexpr Part(std::size_t part) : part_(part) {}

  template<typename T>
  [[nodiscard]] TypedMasked<T, size> instantiate(TypeTag<T> /*tag*/) const {
    return TypedMasked<T, size>{make_mask<Mask<T, size>>()};
  }

  template<SizedMask<size> TMask>
  [[nodiscard]] TMask mask(TMask m) const {
    return m && make_mask<TMask>();
  }
  template<SizedVector<size> TVec>
  [[nodiscard]] TVec mask(TVec v) const {
    return v.cutoff(part_);
  }

  [[nodiscard]] std::size_t part() const {
    return part_;
  }

private:
  template<typename TMask>
  auto make_mask(TypeTag<TMask> /*tag*/ = {}) const {
    return TMask::cutoff_mask(part_);
  }

  std::size_t part_;
};
} // namespace grex::tag

#endif // INCLUDE_GREX_TAGS_HPP
