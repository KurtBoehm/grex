// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TYPES_HPP
#define INCLUDE_GREX_TYPES_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include "thesauros/static-ranges/definitions.hpp"
#include "thesauros/types.hpp"

#include "grex/backend.hpp"
#include "grex/base.hpp"

namespace grex {
template<Vectorizable T, std::size_t tSize>
struct Mask {
  using Backend = backend::Mask<T, tSize>;
  static constexpr std::size_t size = tSize;
  static constexpr thes::star::PrintableMarker printable{};

  Mask() : mask_{backend::zero(thes::type_tag<Backend>)} {}
  explicit Mask(T value) : mask_{backend::broadcast(value, thes::type_tag<Backend>)} {}
  template<typename... Ts>
  requires(((sizeof...(Ts) == tSize) && ... && std::same_as<Ts, bool>))
  explicit Mask(Ts... values) : mask_{backend::set(T{values}..., thes::type_tag<Backend>)} {}

private:
  explicit Mask(Backend v) : mask_(v) {}
  Backend mask_;
};

template<Vectorizable T, std::size_t tSize>
struct Vector {
  using Value = T;
  using Mask = grex::Mask<T, tSize>;
  using Backend = backend::Vector<T, tSize>;
  static constexpr std::size_t size = tSize;
  static constexpr thes::star::PrintableMarker printable{};

  Vector() : vec_{backend::zero(thes::type_tag<Backend>)} {}
  explicit Vector(T value) : vec_{backend::broadcast(value, thes::type_tag<Backend>)} {}
  template<typename... Ts>
  requires(sizeof...(Ts) == tSize)
  explicit Vector(Ts... values) : vec_{backend::set(T{values}..., thes::type_tag<Backend>)} {}

  friend Vector operator+(Vector a, Vector b) {
    return Vector{backend::add(a.vec_, b.vec_)};
  }
  friend Vector operator-(Vector a, Vector b) {
    return Vector{backend::subtract(a.vec_, b.vec_)};
  }

  void store(T* value) const {
    backend::store(value, vec_);
  }
  void store_aligned(T* value) const {
    backend::store_aligned(value, vec_);
  }

  T operator[](std::size_t i) const {
    return backend::extract(vec_, i);
  }
  T get(thes::AnyIndexTag auto i) const {
    return backend::extract(vec_, i);
  }
  std::array<T, tSize> as_array() const {
    std::array<T, tSize> out{};
    store(out.data());
    return out;
  }

private:
  explicit Vector(Backend v) : vec_(v) {}
  Backend vec_;
};
} // namespace grex

#endif // INCLUDE_GREX_TYPES_HPP
