// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TYPES_HPP
#define INCLUDE_GREX_TYPES_HPP

#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "grex/backend.hpp"
#include "grex/base.hpp"

namespace grex {
using backend::has_fma;
using backend::native_sizes;
using backend::register_bits;
using backend::register_bytes;

template<Vectorizable T, std::size_t tSize>
struct Mask {
  using Value = bool;
  using VectorValue = T;
  using Backend = backend::MaskFor<T, tSize>;
  static constexpr std::size_t size = tSize;

  Mask() : mask_{backend::zeros(type_tag<Backend>)} {}
  explicit Mask(bool value) : mask_{backend::broadcast(value, type_tag<Backend>)} {}
  template<typename... Ts>
  requires(((sizeof...(Ts) == tSize) && ... && std::same_as<Ts, bool>))
  explicit Mask(Ts... values) : mask_{backend::set(type_tag<Backend>, values...)} {}
  explicit Mask(Backend v) : mask_(v) {}

  static Mask zeros() {
    return Mask{backend::zeros(type_tag<Backend>)};
  }
  static Mask ones() {
    return Mask{backend::ones(type_tag<Backend>)};
  }
  static Mask cutoff_mask(std::size_t i) {
    return Mask{backend::cutoff_mask(i, type_tag<Backend>)};
  }

  template<Vectorizable TDst>
  Mask<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Mask<TDst, tSize>{backend::convert(mask_, type_tag<TDst>)};
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
    return Mask{backend::compare_eq(a.mask_, b.mask_)};
  }

  bool operator[](std::size_t i) const {
    return backend::extract(mask_, i);
  }
  // this form of get is required for {fmt} to use its formatter for tuple-like types
  template<std::size_t tIdx>
  friend bool get(const Mask& m) {
    return backend::extract(m.mask_, tIdx);
  }
  Mask insert(std::size_t i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }
  Mask insert(AnyIndexTag auto i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }

  Backend backend() const {
    return mask_;
  }

private:
  Backend mask_;
};

template<Vectorizable T, std::size_t tSize>
struct Vector;

// This silly little base class allows the constructor arguments to be declared as all being
// of type T, which leads to integer literals being interpreted as the correct type,
// among other benefits
template<Vectorizable T, typename TIdxs>
struct VectorBase;
template<Vectorizable T, std::size_t... tIdxs>
struct VectorBase<T, std::index_sequence<tIdxs...>> {
  static constexpr std::size_t size = sizeof...(tIdxs);
  using Backend = backend::VectorFor<T, size>;
  friend Vector<T, size>;

  explicit VectorBase(IdxType<tIdxs, T>... values)
      : vec_{backend::set(type_tag<Backend>, values...)} {}
  explicit VectorBase(Backend v) : vec_(v) {}

private:
  Backend vec_;
};

template<Vectorizable T, std::size_t tSize>
struct Vector : public VectorBase<T, std::make_index_sequence<tSize>> {
  using Value = T;
  using Mask = grex::Mask<T, tSize>;
  using Backend = backend::VectorFor<T, tSize>;
  static constexpr std::size_t size = tSize;

  using Base = VectorBase<T, std::make_index_sequence<tSize>>;
  using Base::Base;

  Vector() : Base{backend::zeros(type_tag<Backend>)} {}
  explicit Vector(T value) : Base{backend::broadcast(value, type_tag<Backend>)} {}

  static Vector expanded_any(T x) {
    return Vector{backend::expand_any(backend::Scalar<T>{x}, index_tag<tSize>)};
  }
  static Vector expanded_zero(T x) {
    return Vector{backend::expand_zero(backend::Scalar<T>{x}, index_tag<tSize>)};
  }

  static Vector load(const T* ptr) {
    return Vector{backend::load(ptr, type_tag<Backend>)};
  }
  static Vector load_aligned(const T* ptr) {
    return Vector{backend::load_aligned(ptr, type_tag<Backend>)};
  }
  static Vector load_part(const T* ptr, std::size_t num) {
    return Vector{backend::load_part(ptr, num, type_tag<Backend>)};
  }

  template<std::size_t tSrcBytes>
  static Vector load_multibyte(const std::byte* data, IndexTag<tSrcBytes> src_bytes) {
    const auto* raw = reinterpret_cast<const u8*>(data);
    return Vector{backend::load_multibyte(raw, src_bytes, type_tag<Backend>)};
  }
  template<MultiByteIterator TIt>
  static Vector load_multibyte(TIt it) {
    return load_multibyte(it.raw(), index_tag<TIt::Container::element_bytes>);
  }

  static Vector undefined() {
    return Vector{backend::undefined(type_tag<Backend>)};
  }
  static Vector zeros() {
    return Vector{backend::zeros(type_tag<Backend>)};
  }
  static Vector indices() {
    return Vector{backend::indices(type_tag<Backend>)};
  }
  static Vector indices(T start) {
    return indices() + Vector{start};
  }

  Vector operator-() const {
    return Vector{backend::negate(vec_)};
  }
  Vector operator~() const
  requires(IntVectorizable<T>)
  {
    return Vector{backend::bitwise_not(vec_)};
  }

#define GREX_VECTOR_BINOP(OP, REQ, NAME) \
  friend Vector operator OP(Vector a, Vector b) REQ { \
    return Vector{backend::NAME(a.vec_, b.vec_)}; \
  } \
  friend Vector operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  friend Vector operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  } \
  Vector& operator OP##=(Vector b) REQ { \
    return *this = *this OP b; \
  } \
  Vector& operator OP##=(Value b) REQ { \
    return *this = *this OP b; \
  }

  GREX_VECTOR_BINOP(+, , add)
  GREX_VECTOR_BINOP(-, , subtract)
  GREX_VECTOR_BINOP(*, , multiply)
  GREX_VECTOR_BINOP(/, requires(FloatVectorizable<T>), divide)
  GREX_VECTOR_BINOP(&, requires(IntVectorizable<T>), bitwise_and)
  GREX_VECTOR_BINOP(|, requires(IntVectorizable<T>), bitwise_or)
  GREX_VECTOR_BINOP(^, requires(IntVectorizable<T>), bitwise_xor)
#undef GREX_VECTOR_BINOP

  Vector cutoff(std::size_t i) const {
    return Vector{backend::cutoff(i, vec_)};
  }
  template<Vectorizable TDst>
  Vector<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Vector<TDst, tSize>{backend::convert(vec_, type_tag<TDst>)};
  }

  T operator[](std::size_t i) const {
    return backend::extract(vec_, i);
  }
  // this form of get is required for {fmt} to use its formatter for tuple-like types
  template<std::size_t tIdx>
  friend T get(const Vector& v) {
    return backend::extract(v.vec_, tIdx);
  }
  Vector insert(std::size_t i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }
  Vector insert(AnyIndexTag auto i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }

  void store(T* value) const {
    backend::store(value, vec_);
  }
  void store_aligned(T* value) const {
    backend::store_aligned(value, vec_);
  }
  void store_part(T* value, std::size_t num) const {
    backend::store_part(value, vec_, num);
  }

#define GREX_VECTOR_CMP_BINOP(OP, REQ, BACKEND) \
  friend Mask operator OP(Vector a, Vector b) REQ { \
    return Mask{BACKEND}; \
  } \
  friend Mask operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  friend Mask operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  }

  GREX_VECTOR_CMP_BINOP(==, , backend::compare_eq(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(!=, , backend::compare_neq(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(<, , backend::compare_lt(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(>, , backend::compare_lt(b.vec_, a.vec_))
  GREX_VECTOR_CMP_BINOP(>=, , backend::compare_ge(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(<=, , backend::compare_ge(b.vec_, a.vec_))
#undef GREX_VECTOR_CMP_BINOP

  template<std::size_t tDstSize>
  Vector<T, tDstSize> expand_any(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_any(vec_, index_tag<tDstSize>)};
  }
  template<std::size_t tDstSize>
  Vector<T, tDstSize> expand_zero(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_zero(vec_, index_tag<tDstSize>)};
  }

  Vector shingle_up() const {
    return Vector{backend::shingle_up(vec_)};
  }
  Vector shingle_up(Value front) const {
    return Vector{backend::shingle_up(backend::Scalar<T>{front}, vec_)};
  }
  Vector shingle_down() const {
    return Vector{backend::shingle_down(vec_)};
  }
  Vector shingle_down(Value back) const {
    return Vector{backend::shingle_down(vec_, backend::Scalar<T>{back})};
  }

  Backend backend() const {
    return vec_;
  }

private:
  using Base::vec_;
};

template<Vectorizable T, std::size_t tSize>
inline Mask<T, tSize> andnot(Mask<T, tSize> a, Mask<T, tSize> b) {
  return Mask<T, tSize>{backend::logical_andnot(a.backend(), b.backend())};
}

// traits
template<typename T>
struct MaskTrait : public std::false_type {};
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<Mask<T, tSize>> : public std::true_type {
  using VectorFor = Vector<T, tSize>;
};
template<typename T>
struct VectorTrait : public std::false_type {};
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<Vector<T, tSize>> : public std::true_type {
  using MaskFor = Mask<T, tSize>;
};

// mask concepts
template<typename T>
concept AnyMask = MaskTrait<T>::value;
template<typename T, std::size_t tSize>
concept SizedMask = MaskTrait<T>::value && T::size == tSize;

// vector concepts
template<typename TVec>
concept AnyVector = VectorTrait<TVec>::value;
template<typename TVec, typename TVal>
concept ValuedVector = VectorTrait<TVec>::value && std::same_as<typename TVec::Value, TVal>;
template<typename TVec, std::size_t tSize>
concept SizedVector = VectorTrait<TVec>::value && TVec::size == tSize;
template<typename TVec>
concept IntVector = AnyVector<TVec> && IntVectorizable<typename TVec::Value>;
template<typename TVec>
concept FpVector = AnyVector<TVec> && FloatVectorizable<typename TVec::Value>;

// type mappings
template<AnyVector TVec>
using MaskFor = VectorTrait<TVec>::MaskFor;
template<AnyMask TVec>
using VectorFor = MaskTrait<TVec>::VectorFor;

template<SignedVectorizable T, std::size_t tSize>
inline Vector<T, tSize> abs(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::abs(v.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> sqrt(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::sqrt(v.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> min(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::min(a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> max(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::max(a.backend(), b.backend())};
}

template<FloatVectorizable T, std::size_t tSize>
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
inline bool horizontal_and(Mask<T, tSize> m) {
  return backend::horizontal_and(m.backend());
}

template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmadd(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmsub(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fnmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmadd(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> fnmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmsub(a.backend(), b.backend(), c.backend())};
}

template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend_zero(Mask<T, tSize> mask, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero(mask.backend(), v1.backend())};
}
template<BlendZero... tBzs, Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend_zero(Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero<tBzs...>(v1.backend())};
}
template<Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend(Mask<T, tSize> mask, Vector<T, tSize> v0, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend(mask.backend(), v0.backend(), v1.backend())};
}
template<BlendSelector... tBls, Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> blend(Vector<T, tSize> v0, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend<tBls...>(v0.backend(), v1.backend())};
}

template<ShuffleIndex... tIdxs, Vectorizable T, std::size_t tSize>
inline Vector<T, tSize> shuffle(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::shuffle<tIdxs...>(v.backend())};
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
template<FloatVectorizable T, std::size_t tSize>
inline Vector<T, tSize> mask_divide(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_divide(mask.backend(), a.backend(), b.backend())};
}

template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
inline Vector<TValue, tSize> gather(std::span<const TValue, tExtent> data,
                                    Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::gather(data, indices.backend())};
}
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
inline Vector<TValue, tSize> mask_gather(std::span<const TValue, tExtent> data,
                                         Mask<TValue, tSize> mask, Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::mask_gather(data, mask.backend(), indices.backend())};
}
} // namespace grex

// implement the tuple-like interface for structured bindings and automatic {fmt} formatting
template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Vector<T, tSize>> : public std::integral_constant<std::size_t, tSize> {
};
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Vector<T, tSize>> {
  using type = const T;
};

template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Mask<T, tSize>> : public std::integral_constant<std::size_t, tSize> {};
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Mask<T, tSize>> {
  using type = const bool;
};

#endif // INCLUDE_GREX_TYPES_HPP
