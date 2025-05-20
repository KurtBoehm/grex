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
  explicit Mask(Ts... values) : mask_{backend::set(T(values)..., thes::type_tag<Backend>)} {}
  explicit Mask(Backend v) : mask_(v) {}

  static Mask cutoff_mask(std::size_t i) {
    return Mask{backend::cutoff_mask(i, thes::type_tag<Backend>)};
  }

  Mask operator!() const {
    return Mask{backend::logical_not(mask_)};
  }
  friend Mask operator&&(Mask a, Mask b) {
    return Mask{backend::logical_and(a.mask_, b.mask_)};
  }
  friend Mask operator||(Mask a, Mask b) {
    return Mask{backend::logical_or(a.mask_, b.mask_)};
  }

  friend Mask operator!=(Mask a, Mask b) {
    return Mask{backend::logical_xor(a.mask_, b.mask_)};
  }
  friend Mask operator==(Mask a, Mask b) {
    return !(a != b);
  }

  bool operator[](std::size_t i) const {
    return backend::extract(mask_, i);
  }
  bool get(thes::AnyIndexTag auto i) const {
    return backend::extract(mask_, i);
  }

  Backend backend() const {
    return mask_;
  }

private:
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
  explicit Vector(Backend v) : vec_(v) {}

  static Vector load(const T* ptr) {
    return Vector{backend::load(ptr, thes::index_tag<size>)};
  }
  static Vector load_aligned(const T* ptr) {
    return Vector{backend::load_aligned(ptr, thes::index_tag<size>)};
  }

  static Vector indices() {
    return Vector{backend::indices(thes::type_tag<Backend>)};
  }
  static Vector indices(T start) {
    return indices() + Vector{start};
  }

  Vector operator-() const {
    return Vector{backend::negate(vec_)};
  }
  friend Vector operator+(Vector a, Vector b) {
    return Vector{backend::add(a.vec_, b.vec_)};
  }
  friend Vector operator-(Vector a, Vector b) {
    return Vector{backend::subtract(a.vec_, b.vec_)};
  }
  friend Vector operator*(Vector a, Vector b) {
    return Vector{backend::multiply(a.vec_, b.vec_)};
  }
  friend Vector operator/(Vector a, Vector b)
  requires(std::floating_point<T>)
  {
    return Vector{backend::divide(a.vec_, b.vec_)};
  }

  Vector cutoff(std::size_t i) const {
    return Vector{backend::cutoff(i, vec_)};
  }

  T operator[](std::size_t i) const {
    return backend::extract(vec_, i);
  }
  T get(thes::AnyIndexTag auto i) const {
    return backend::extract(vec_, i);
  }
  Vector insert(std::size_t i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }

  void store(T* value) const {
    backend::store(value, vec_);
  }
  void store_aligned(T* value) const {
    backend::store_aligned(value, vec_);
  }
  std::array<T, tSize> as_array() const {
    std::array<T, tSize> out{};
    store(out.data());
    return out;
  }

  Vector operator~() const
  requires(std::integral<T>)
  {
    return Vector{backend::bitwise_not(vec_)};
  }
  friend Vector operator&(Vector a, Vector b)
  requires(std::integral<T>)
  {
    return Vector{backend::bitwise_and(a.vec_, b.vec_)};
  }
  friend Vector operator|(Vector a, Vector b)
  requires(std::integral<T>)
  {
    return Vector{backend::bitwise_or(a.vec_, b.vec_)};
  }
  friend Vector operator^(Vector a, Vector b)
  requires(std::integral<T>)
  {
    return Vector{backend::bitwise_xor(a.vec_, b.vec_)};
  }

  friend Mask operator==(Vector a, Vector b) {
    return Mask{backend::compare_eq(a.vec_, b.vec_)};
  }
  friend Mask operator!=(Vector a, Vector b) {
    return Mask{backend::compare_neq(a.vec_, b.vec_)};
  }
  friend Mask operator<(Vector a, Vector b) {
    return Mask{backend::compare_lt(a.vec_, b.vec_)};
  }
  friend Mask operator>(Vector a, Vector b) {
    return Mask{backend::compare_lt(b.vec_, a.vec_)};
  }
  friend Mask operator>=(Vector a, Vector b) {
    return Mask{backend::compare_ge(a.vec_, b.vec_)};
  }
  friend Mask operator<=(Vector a, Vector b) {
    return Mask{backend::compare_ge(b.vec_, a.vec_)};
  }

  Backend backend() const {
    return vec_;
  }

private:
  Backend vec_;
};

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> abs(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::abs(v.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> min(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::min(a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> max(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::max(a.backend(), b.backend())};
}

template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Mask<T, tSize> is_finite(Vector<T, tSize> v) {
  return Mask<T, tSize>{backend::is_finite(v.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline T horizontal_add(Vector<T, tSize> v) {
  return backend::horizontal_add(v.backend());
}
template<Vectorizable T, std::size_t tSize>
inline T horizontal_min(Vector<T, tSize> v) {
  return backend::horizontal_min(v.backend());
}
template<Vectorizable T, std::size_t tSize>
inline T horizontal_max(Vector<T, tSize> v) {
  return backend::horizontal_max(v.backend());
}
template<Vectorizable T, std::size_t tSize>
inline T horizontal_and(Mask<T, tSize> m) {
  return backend::horizontal_and(m.backend());
}

template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Vector<T, tSize> fmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmadd(a.backend(), b.backend(), c.backend())};
}
template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Vector<T, tSize> fmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmsub(a.backend(), b.backend(), c.backend())};
}
template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Vector<T, tSize> fnmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmadd(a.backend(), b.backend(), c.backend())};
}
template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Vector<T, tSize> fnmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmsub(a.backend(), b.backend(), c.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend_zero(Mask<T, tSize> mask, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero(mask.backend(), v1.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend(Mask<T, tSize> mask, Vector<T, tSize> v0, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend(mask.backend(), v0.backend(), v1.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> mask_add(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_add(mask.backend(), a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> mask_subtract(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_subtract(mask.backend(), a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> mask_multiply(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_multiply(mask.backend(), a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
requires(std::floating_point<T>)
inline Vector<T, tSize> mask_divide(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_divide(mask.backend(), a.backend(), b.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> shuffle(Vector<T, tSize> v,
                                thes::TypedValueTag<std::array<ShuffleIndex, tSize>> auto idxs) {
  return Vector<T, tSize>{backend::shuffle(v.backend(), idxs)};
}
} // namespace grex

#endif // INCLUDE_GREX_TYPES_HPP
