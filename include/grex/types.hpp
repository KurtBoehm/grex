// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TYPES_HPP
#define INCLUDE_GREX_TYPES_HPP

#include "grex/backend.hpp"

namespace grex {
template<Vectorizable T>
static constexpr std::size_t max_native_size = backend::max_native_size<T>;
template<Vectorizable T>
static constexpr std::size_t min_native_size = backend::min_native_size<T>;
} // namespace grex

#if !GREX_BACKEND_SCALAR
#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "grex/base.hpp"

namespace grex {
template<Vectorizable T>
static constexpr std::array native_sizes = backend::native_sizes<T>;
static constexpr std::array register_bits = backend::register_bits;
static constexpr std::array register_bytes = backend::register_bytes;

template<Vectorizable T, std::size_t tSize>
struct Mask {
  using Value = bool;
  using VectorValue = T;
  using Backend = backend::MaskFor<T, tSize>;
  static constexpr std::size_t size = tSize;

  GREX_ALWAYS_INLINE Mask() : mask_{backend::zeros(type_tag<Backend>)} {}
  GREX_ALWAYS_INLINE explicit Mask(bool value)
      : mask_{backend::broadcast(value, type_tag<Backend>)} {}
  template<typename... Ts>
  requires(((sizeof...(Ts) == tSize) && ... && std::same_as<Ts, bool>))
  GREX_ALWAYS_INLINE explicit Mask(Ts... values)
      : mask_{backend::set(type_tag<Backend>, values...)} {}
  GREX_ALWAYS_INLINE explicit Mask(Backend v) : mask_(v) {}

  GREX_ALWAYS_INLINE static Mask zeros() {
    return Mask{backend::zeros(type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Mask ones() {
    return Mask{backend::ones(type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Mask cutoff_mask(std::size_t i) {
    return Mask{backend::cutoff_mask(i, type_tag<Backend>)};
  }

  template<Vectorizable TDst>
  GREX_ALWAYS_INLINE Mask<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Mask<TDst, tSize>{backend::convert(mask_, type_tag<TDst>)};
  }

  GREX_ALWAYS_INLINE Mask operator!() const {
    return Mask{backend::logical_not(mask_)};
  }
  GREX_ALWAYS_INLINE friend Mask operator&&(Mask a, Mask b) {
    return Mask{backend::logical_and(a.mask_, b.mask_)};
  }
  GREX_ALWAYS_INLINE friend Mask operator||(Mask a, Mask b) {
    return Mask{backend::logical_or(a.mask_, b.mask_)};
  }

  GREX_ALWAYS_INLINE friend Mask operator!=(Mask a, Mask b) {
    return Mask{backend::logical_xor(a.mask_, b.mask_)};
  }
  GREX_ALWAYS_INLINE friend Mask operator==(Mask a, Mask b) {
    return Mask{backend::compare_eq(a.mask_, b.mask_)};
  }

  GREX_ALWAYS_INLINE bool operator[](std::size_t i) const {
    return backend::extract(mask_, i);
  }
  GREX_ALWAYS_INLINE bool operator[](AnyIndexTag auto i) const {
    return backend::extract(mask_, i);
  }
  template<std::size_t tIdx>
  GREX_ALWAYS_INLINE friend bool get(const Mask& m) {
    return backend::extract(m.mask_, index_tag<tIdx>);
  }
  GREX_ALWAYS_INLINE Mask insert(std::size_t i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }
  GREX_ALWAYS_INLINE Mask insert(AnyIndexTag auto i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }

  GREX_ALWAYS_INLINE Backend backend() const {
    return mask_;
  }
  GREX_ALWAYS_INLINE std::array<bool, size> as_array() const {
    return backend::to_array(mask_);
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

  GREX_ALWAYS_INLINE explicit VectorBase(IdxType<tIdxs, T>... values)
      : vec_{backend::set(type_tag<Backend>, values...)} {}
  GREX_ALWAYS_INLINE explicit VectorBase(Backend v) : vec_(v) {}

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

  GREX_ALWAYS_INLINE Vector() : Base{backend::zeros(type_tag<Backend>)} {}
  GREX_ALWAYS_INLINE explicit Vector(T value)
      : Base{backend::broadcast(value, type_tag<Backend>)} {}

  GREX_ALWAYS_INLINE static Vector expanded_any(T x) {
    return Vector{backend::expand_any(backend::Scalar<T>{x}, index_tag<tSize>)};
  }
  GREX_ALWAYS_INLINE static Vector expanded_zero(T x) {
    return Vector{backend::expand_zero(backend::Scalar<T>{x}, index_tag<tSize>)};
  }

  GREX_ALWAYS_INLINE static Vector load(const T* ptr) {
    return Vector{backend::load(ptr, type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Vector load_aligned(const T* ptr) {
    return Vector{backend::load_aligned(ptr, type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Vector load_part(const T* ptr, std::size_t num) {
    return Vector{backend::load_part(ptr, num, type_tag<Backend>)};
  }

  template<std::size_t tSrcBytes>
  GREX_ALWAYS_INLINE static Vector load_multibyte(const std::byte* data,
                                                  IndexTag<tSrcBytes> src_bytes)
  requires(UnsignedIntVectorizable<T> && tSrcBytes <= sizeof(Value))
  {
    const auto* raw = reinterpret_cast<const u8*>(data);
    return Vector{backend::load_multibyte(raw, src_bytes, type_tag<Backend>)};
  }
  template<MultiByteIterator TIt>
  GREX_ALWAYS_INLINE static Vector load_multibyte(TIt it)
  requires(UnsignedIntVectorizable<T>)
  {
    return load_multibyte(it.raw(), index_tag<TIt::Container::element_bytes>);
  }

  GREX_ALWAYS_INLINE static Vector undefined() {
    return Vector{backend::undefined(type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Vector zeros() {
    return Vector{backend::zeros(type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Vector indices() {
    return Vector{backend::indices(type_tag<Backend>)};
  }
  GREX_ALWAYS_INLINE static Vector indices(T start) {
    return indices() + Vector{start};
  }

  GREX_ALWAYS_INLINE Vector operator-() const {
    return Vector{backend::negate(vec_)};
  }
  GREX_ALWAYS_INLINE Vector operator~() const
  requires(IntVectorizable<T>)
  {
    return Vector{backend::bitwise_not(vec_)};
  }

#define GREX_VECTOR_BINOP(OP, REQ, NAME) \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, Vector b) REQ { \
    return Vector{backend::NAME(a.vec_, b.vec_)}; \
  } \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  GREX_ALWAYS_INLINE friend Vector operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  } \
  GREX_ALWAYS_INLINE Vector& operator OP##=(Vector b) REQ { \
    return *this = *this OP b; \
  } \
  GREX_ALWAYS_INLINE Vector& operator OP##=(Value b) REQ { \
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

#define GREX_VECTOR_SHOP(OP, REQ, NAME) \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, AnyIndexTag auto offset) REQ { \
    return Vector{backend::NAME(a.vec_, offset)}; \
  } \
  GREX_ALWAYS_INLINE Vector& operator OP##=(AnyIndexTag auto offset) REQ { \
    return *this = *this OP offset; \
  }

  GREX_VECTOR_SHOP(<<, , shift_left)
  GREX_VECTOR_SHOP(>>, , shift_right)
#undef GREX_VECTOR_SHOP

  GREX_ALWAYS_INLINE Vector cutoff(std::size_t i) const {
    return Vector{backend::cutoff(i, vec_)};
  }
  template<Vectorizable TDst>
  GREX_ALWAYS_INLINE Vector<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Vector<TDst, tSize>{backend::convert(vec_, type_tag<TDst>)};
  }

  GREX_ALWAYS_INLINE T operator[](std::size_t i) const {
    return backend::extract(vec_, i);
  }
  GREX_ALWAYS_INLINE T operator[](AnyIndexTag auto i) const {
    return backend::extract(vec_, i);
  }
  template<std::size_t tIdx>
  GREX_ALWAYS_INLINE friend T get(const Vector& v) {
    return backend::extract(v.vec_, index_tag<tIdx>);
  }

  GREX_ALWAYS_INLINE Vector insert(std::size_t i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }
  GREX_ALWAYS_INLINE Vector insert(AnyIndexTag auto i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }

  GREX_ALWAYS_INLINE void store(T* value) const {
    backend::store(value, vec_);
  }
  GREX_ALWAYS_INLINE void store_aligned(T* value) const {
    backend::store_aligned(value, vec_);
  }
  GREX_ALWAYS_INLINE void store_part(T* value, std::size_t num) const {
    backend::store_part(value, vec_, num);
  }

#define GREX_VECTOR_CMP_BINOP(OP, REQ, BACKEND) \
  GREX_ALWAYS_INLINE friend Mask operator OP(Vector a, Vector b) REQ { \
    return Mask{BACKEND}; \
  } \
  GREX_ALWAYS_INLINE friend Mask operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  GREX_ALWAYS_INLINE friend Mask operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  }

  GREX_VECTOR_CMP_BINOP(==, , backend::compare_eq(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(!=, , backend::compare_neq(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(<, , backend::compare_lt(a.vec_, b.vec_))
  GREX_VECTOR_CMP_BINOP(>, , backend::compare_lt(b.vec_, a.vec_))
  GREX_VECTOR_CMP_BINOP(<=, , backend::compare_ge(b.vec_, a.vec_))
  GREX_VECTOR_CMP_BINOP(>=, , backend::compare_ge(a.vec_, b.vec_))
#undef GREX_VECTOR_CMP_BINOP

  template<std::size_t tDstSize>
  GREX_ALWAYS_INLINE Vector<T, tDstSize> expand_any(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_any(vec_, index_tag<tDstSize>)};
  }
  template<std::size_t tDstSize>
  GREX_ALWAYS_INLINE Vector<T, tDstSize> expand_zero(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_zero(vec_, index_tag<tDstSize>)};
  }

  GREX_ALWAYS_INLINE Vector shingle_up() const {
    return Vector{backend::shingle_up(vec_)};
  }
  GREX_ALWAYS_INLINE Vector shingle_up(Value front) const {
    return Vector{backend::shingle_up(backend::Scalar<T>{front}, vec_)};
  }
  GREX_ALWAYS_INLINE Vector shingle_down() const {
    return Vector{backend::shingle_down(vec_)};
  }
  GREX_ALWAYS_INLINE Vector shingle_down(Value back) const {
    return Vector{backend::shingle_down(vec_, backend::Scalar<T>{back})};
  }

  GREX_ALWAYS_INLINE Backend backend() const {
    return vec_;
  }
  GREX_ALWAYS_INLINE std::array<Value, size> as_array() const {
    return backend::to_array(vec_);
  }

private:
  using Base::vec_;
};

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
concept SizedMask = AnyMask<T> && T::size == tSize;

// vector concepts
template<typename TVec>
concept AnyVector = VectorTrait<TVec>::value;
template<typename TVec, typename TVal>
concept ValuedVector = AnyVector<TVec> && std::same_as<typename TVec::Value, TVal>;
template<typename TVec, std::size_t tSize>
concept SizedVector = AnyVector<TVec> && TVec::size == tSize;
template<typename TVec>
concept IntVector = AnyVector<TVec> && IntVectorizable<typename TVec::Value>;
template<typename TVec>
concept FpVector = AnyVector<TVec> && FloatVectorizable<typename TVec::Value>;

// type mappings
template<AnyVector TVec>
using MaskFor = VectorTrait<TVec>::MaskFor;
template<AnyMask TMask>
using VectorFor = MaskTrait<TMask>::VectorFor;

template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Mask<T, tSize> andnot(Mask<T, tSize> a, Mask<T, tSize> b) {
  return Mask<T, tSize>{backend::logical_andnot(a.backend(), b.backend())};
}

template<SignedVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> abs(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::abs(v.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> sqrt(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::sqrt(v.backend())};
}

template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> min(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::min(a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> max(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::max(a.backend(), b.backend())};
}

template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Mask<T, tSize> is_finite(Vector<T, tSize> v) {
  return Mask<T, tSize>{backend::is_finite(v.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> make_finite(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::make_finite(v.backend())};
}

template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_add(Vector<T, tSize> v) {
  return backend::horizontal_add(v.backend());
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_min(Vector<T, tSize> v) {
  return backend::horizontal_min(v.backend());
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_max(Vector<T, tSize> v) {
  return backend::horizontal_max(v.backend());
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline bool horizontal_and(Mask<T, tSize> m) {
  return backend::horizontal_and(m.backend());
}

template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fmadd(Vector<T, tSize> a, Vector<T, tSize> b,
                                                 Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmadd(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fmsub(Vector<T, tSize> a, Vector<T, tSize> b,
                                                 Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmsub(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fnmadd(Vector<T, tSize> a, Vector<T, tSize> b,
                                                  Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmadd(a.backend(), b.backend(), c.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fnmsub(Vector<T, tSize> a, Vector<T, tSize> b,
                                                  Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmsub(a.backend(), b.backend(), c.backend())};
}

template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T extract_single(Vector<T, tSize> v) {
  return backend::extract_single(v.backend()).value;
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend_zero(Mask<T, tSize> mask, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero(mask.backend(), v1.backend())};
}
template<BlendZeroSelector... tBzs, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend_zero(Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero<tBzs...>(v1.backend())};
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend(Mask<T, tSize> mask, Vector<T, tSize> v0,
                                                 Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend(mask.backend(), v0.backend(), v1.backend())};
}
template<BlendSelector... tBls, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend(Vector<T, tSize> v0, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend<tBls...>(v0.backend(), v1.backend())};
}

template<Vectorizable T, UnsignedIntVectorizable TIdx, std::size_t tTableSize, std::size_t tIdxSize>
GREX_ALWAYS_INLINE inline Vector<T, tIdxSize> shuffle(Vector<T, tTableSize> table,
                                                      Vector<TIdx, tIdxSize> idxs) {
  return Vector<T, tIdxSize>{backend::shuffle(table.backend(), idxs.backend())};
}
template<ShuffleIndex... tIdxs, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> shuffle(Vector<T, tSize> table) {
  return Vector<T, tSize>{backend::shuffle<tIdxs...>(table.backend())};
}

template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_add(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                    Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_add(mask.backend(), a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_subtract(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                         Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_subtract(mask.backend(), a.backend(), b.backend())};
}
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_multiply(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                         Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_multiply(mask.backend(), a.backend(), b.backend())};
}
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_divide(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                       Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_divide(mask.backend(), a.backend(), b.backend())};
}

template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<TValue, tSize> gather(std::span<const TValue, tExtent> data,
                                                       Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::gather(data, indices.backend())};
}
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<TValue, tSize> mask_gather(std::span<const TValue, tExtent> data,
                                                            Mask<TValue, tSize> mask,
                                                            Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::mask_gather(data, mask.backend(), indices.backend())};
}
} // namespace grex

// implement the tuple-like interface for structured bindings and automatic {fmt} formatting
template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Vector<T, tSize>> : public std::integral_constant<std::size_t, tSize> {
};
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Vector<T, tSize>> {
  using type = const T; // NOLINT
};

template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Mask<T, tSize>> : public std::integral_constant<std::size_t, tSize> {};
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Mask<T, tSize>> {
  using type = const bool; // NOLINT
};
#endif

#endif // INCLUDE_GREX_TYPES_HPP
