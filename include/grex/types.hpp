// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_TYPES_HPP
#define INCLUDE_GREX_TYPES_HPP

#include "grex/backend.hpp"

namespace grex {
/** Maximum native lane count for a given scalar type, or 1 with the scalar backend. */
template<Vectorizable T>
static constexpr std::size_t max_native_size = backend::max_native_size<T>;

/** Minimum native lane count for a given scalar type, or 1 with the scalar backend. */
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
/** The native lane counts for a given scalar type. */
template<Vectorizable T>
static constexpr std::array native_sizes = backend::native_sizes<T>;

/** The number of bits in each kind of native vector register. */
static constexpr std::array register_bits = backend::register_bits;

/** The number of bytes in each kind of native vector register. */
static constexpr std::array register_bytes = backend::register_bytes;

/** Boolean mask for `Vector<T, tSize>`. */
template<Vectorizable T, std::size_t tSize>
struct Mask {
  /** %Value type of the mask. */
  using Value = bool;
  /** %Value type of the vector that this mask applies to. */
  using VectorValue = T;
  /** %Backend mask type. */
  using Backend = backend::MaskFor<T, tSize>;
  /** Number of lanes. */
  static constexpr std::size_t size = tSize;

  /** Constructs an all-false mask. */
  GREX_ALWAYS_INLINE Mask() : mask_{backend::zeros(type_tag<Backend>)} {}

  /** Broadcasts a single Boolean value to all lanes. */
  GREX_ALWAYS_INLINE explicit Mask(bool value)
      : mask_{backend::broadcast(value, type_tag<Backend>)} {}

  /** Constructs a mask from per-lane values. */
  template<typename... Ts>
  requires(((sizeof...(Ts) == tSize) && ... && std::same_as<Ts, bool>))
  GREX_ALWAYS_INLINE explicit Mask(Ts... values)
      : mask_{backend::set(type_tag<Backend>, values...)} {}

  /** Constructs a mask from a backend mask. */
  GREX_ALWAYS_INLINE explicit Mask(Backend v) : mask_(v) {}

  /** Returns an all-false mask. */
  GREX_ALWAYS_INLINE static Mask zeros() {
    return Mask{backend::zeros(type_tag<Backend>)};
  }

  /** Returns an all-true mask. */
  GREX_ALWAYS_INLINE static Mask ones() {
    return Mask{backend::ones(type_tag<Backend>)};
  }

  /** Returns a mask with the first `i` lanes set and the rest cleared. */
  GREX_ALWAYS_INLINE static Mask cutoff_mask(std::size_t i) {
    return Mask{backend::cutoff_mask(i, type_tag<Backend>)};
  }

  /** Converts mask to a mask for another type with the same lane count. */
  template<Vectorizable TDst>
  GREX_ALWAYS_INLINE Mask<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Mask<TDst, tSize>{backend::convert(mask_, type_tag<TDst>)};
  }

  /** Lane-wise logical _NOT_. */
  GREX_ALWAYS_INLINE Mask operator!() const {
    return Mask{backend::logical_not(mask_)};
  }

  /** Lane-wise logical _AND_ of two masks. */
  GREX_ALWAYS_INLINE friend Mask operator&&(Mask a, Mask b) {
    return Mask{backend::logical_and(a.mask_, b.mask_)};
  }

  /** Lane-wise logical _OR_ of two masks. */
  GREX_ALWAYS_INLINE friend Mask operator||(Mask a, Mask b) {
    return Mask{backend::logical_or(a.mask_, b.mask_)};
  }

  /** Lane-wise logical _XOR_ of two masks. */
  GREX_ALWAYS_INLINE friend Mask operator!=(Mask a, Mask b) {
    return Mask{backend::logical_xor(a.mask_, b.mask_)};
  }

  /** Lane-wise equality comparison. */
  GREX_ALWAYS_INLINE friend Mask operator==(Mask a, Mask b) {
    return Mask{backend::compare_eq(a.mask_, b.mask_)};
  }

  /** Returns lane `i`. */
  GREX_ALWAYS_INLINE bool operator[](std::size_t i) const {
    return backend::extract(mask_, i);
  }

  /** Returns lane `i` with a compile-time index. */
  GREX_ALWAYS_INLINE bool operator[](AnyIndexTag auto i) const {
    return backend::extract(mask_, i);
  }

  /** Returns lane `tIdx` (for tuple-like access). */
  template<std::size_t tIdx>
  GREX_ALWAYS_INLINE friend bool get(const Mask& m) {
    return backend::extract(m.mask_, index_tag<tIdx>);
  }

  /** Returns a copy with lane `i` replaced by `value`. */
  GREX_ALWAYS_INLINE Mask insert(std::size_t i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }

  /** Returns a copy with lane `i` replaced by `value` with a compile-time index. */
  GREX_ALWAYS_INLINE Mask insert(AnyIndexTag auto i, bool value) const {
    return Mask{backend::insert(mask_, i, value)};
  }

  /** Returns underlying backend mask. */
  GREX_ALWAYS_INLINE Backend backend() const {
    return mask_;
  }

  /** Returns contents as `std::array<bool, size>`. */
  GREX_ALWAYS_INLINE std::array<bool, size> as_array() const {
    return backend::to_array(mask_);
  }

private:
  Backend mask_;
};

template<Vectorizable T, std::size_t tSize>
struct Vector;

/** Base class to implement the variadic constructor with typed values for `Vector`. */
template<Vectorizable T, typename TIdxs>
struct VectorBase;
template<Vectorizable T, std::size_t... tIdxs>
struct VectorBase<T, std::index_sequence<tIdxs...>> {
  static constexpr std::size_t size = sizeof...(tIdxs);
  using Backend = backend::VectorFor<T, size>;
  friend Vector<T, size>;

  /** Constructs a vector from typed per-lane values. */
  GREX_ALWAYS_INLINE explicit VectorBase(IdxType<tIdxs, T>... values)
      : vec_{backend::set(type_tag<Backend>, values...)} {}

  /** Constructs a vector from a backend vector. */
  GREX_ALWAYS_INLINE explicit VectorBase(Backend v) : vec_(v) {}

private:
  Backend vec_;
};

/** Generic SIMD vector type of type `T` with `tSize` lanes. */
template<Vectorizable T, std::size_t tSize>
struct Vector : public VectorBase<T, std::make_index_sequence<tSize>> {
  /** Scalar value type. */
  using Value = T;
  /** Corresponding `Mask` type. */
  using Mask = grex::Mask<T, tSize>;
  /** %Backend vector type. */
  using Backend = backend::VectorFor<T, tSize>;
  /** Number of lanes. */
  static constexpr std::size_t size = tSize;

  /** %Base class providing storage and basic constructors. */
  using Base = VectorBase<T, std::make_index_sequence<tSize>>;
  using Base::Base;

  /** Constructs a zero vector. */
  GREX_ALWAYS_INLINE Vector() : Base{backend::zeros(type_tag<Backend>)} {}

  /** Broadcasts a scalar to all lanes. */
  GREX_ALWAYS_INLINE explicit Vector(T value)
      : Base{backend::broadcast(value, type_tag<Backend>)} {}

  /** Expands scalar `x` into a vector with undefined upper lanes. */
  GREX_ALWAYS_INLINE static Vector expanded_any(T x) {
    return Vector{backend::expand_any(backend::Scalar<T>{x}, index_tag<tSize>)};
  }

  /** Expands scalar `x` into a vector with upper lanes filled with zeros. */
  GREX_ALWAYS_INLINE static Vector expanded_zero(T x) {
    return Vector{backend::expand_zero(backend::Scalar<T>{x}, index_tag<tSize>)};
  }

  /** Loads a vector from unaligned memory. */
  GREX_ALWAYS_INLINE static Vector load(const T* ptr) {
    return Vector{backend::load(ptr, type_tag<Backend>)};
  }

  /** Loads a vector from aligned memory. */
  GREX_ALWAYS_INLINE static Vector load_aligned(const T* ptr) {
    return Vector{backend::load_aligned(ptr, type_tag<Backend>)};
  }

  /** Loads `num` (up to `size`) elements from memory with undefined upper lanes. */
  GREX_ALWAYS_INLINE static Vector load_part(const T* ptr, std::size_t num) {
    return Vector{backend::load_part(ptr, num, type_tag<Backend>)};
  }

  /**
   * Loads `size` unsigned integers stored using `tSrcBytes` bytes each
   * and converts each to `Value`.
   */
  template<std::size_t tSrcBytes>
  GREX_ALWAYS_INLINE static Vector load_multibyte(const std::byte* data,
                                                  IndexTag<tSrcBytes> src_bytes)
  requires(UnsignedIntVectorizable<T> && tSrcBytes <= sizeof(Value))
  {
    const auto* raw = reinterpret_cast<const u8*>(data);
    return Vector{backend::load_multibyte(raw, src_bytes, type_tag<Backend>)};
  }

  /**
   * Loads `size` unsigned integers starting at `it` and converts each to `Value`.
   *
   * This convenience overload is intended to be used with iterators to a data structure that stores
   * unsigned integers with `TIt::Container::element_bytes` bytes each and provides a pointer to
   * the underlying bytes (represented using `std::byte`) through `it.raw()`.
   */
  template<MultiByteIterator TIt>
  GREX_ALWAYS_INLINE static Vector load_multibyte(TIt it)
  requires(UnsignedIntVectorizable<T>)
  {
    return load_multibyte(it.raw(), index_tag<TIt::Container::element_bytes>);
  }

  /** Returns an undefined vector. */
  GREX_ALWAYS_INLINE static Vector undefined() {
    return Vector{backend::undefined(type_tag<Backend>)};
  }

  /** Returns a zero vector. */
  GREX_ALWAYS_INLINE static Vector zeros() {
    return Vector{backend::zeros(type_tag<Backend>)};
  }

  /** Returns a vector of lane indices `[0, 1, ..., size - 1]`. */
  GREX_ALWAYS_INLINE static Vector indices() {
    return Vector{backend::indices(type_tag<Backend>)};
  }

  /** Returns a vector of offset lane indices `[start, start + 1, ..., start + size - 1]`. */
  GREX_ALWAYS_INLINE static Vector indices(T start) {
    return indices() + Vector{start};
  }

  /** Lane-wise unary minus. */
  GREX_ALWAYS_INLINE Vector operator-() const {
    return Vector{backend::negate(vec_)};
  }

  /** Lane-wise bitwise negation (integer vectors only). */
  GREX_ALWAYS_INLINE Vector operator~() const
  requires(IntVectorizable<T>)
  {
    return Vector{backend::bitwise_not(vec_)};
  }

#define GREX_VECTOR_BINOP(OP, REQ, NAME, COMMENT_NAME) \
  /** Lane-wise COMMENT_NAME between vectors. */ \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, Vector b) REQ { \
    return Vector{backend::NAME(a.vec_, b.vec_)}; \
  } \
  /** Lane-wise COMMENT_NAME with a scalar on the right. */ \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  /** Lane-wise COMMENT_NAME with a scalar on the left. */ \
  GREX_ALWAYS_INLINE friend Vector operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  } \
  /** Compound lane-wise COMMENT_NAME assignment with vector. */ \
  GREX_ALWAYS_INLINE Vector& operator OP##=(Vector b) REQ { \
    return *this = *this OP b; \
  } \
  /** Compound lane-wise COMMENT_NAME assignment with scalar. */ \
  GREX_ALWAYS_INLINE Vector& operator OP##=(Value b) REQ { \
    return *this = *this OP b; \
  }

  GREX_VECTOR_BINOP(+, , add, addition)
  GREX_VECTOR_BINOP(-, , subtract, subtraction)
  GREX_VECTOR_BINOP(*, , multiply, multiplication)
  GREX_VECTOR_BINOP(/, requires(FloatVectorizable<T>), divide, division)
  GREX_VECTOR_BINOP(&, requires(IntVectorizable<T>), bitwise_and, bitwise _AND_)
  GREX_VECTOR_BINOP(|, requires(IntVectorizable<T>), bitwise_or, bitwise _OR_)
  GREX_VECTOR_BINOP(^, requires(IntVectorizable<T>), bitwise_xor, bitwise _XOR_)
#undef GREX_VECTOR_BINOP

#define GREX_VECTOR_SHOP(OP, REQ, NAME, COMMENT_NAME) \
  /** Lane-wise COMMENT_NAME shift operator by `offset` bits. */ \
  GREX_ALWAYS_INLINE friend Vector operator OP(Vector a, AnyIndexTag auto offset) REQ { \
    return Vector{backend::NAME(a.vec_, offset)}; \
  } \
  /** Compound lane-wise COMMENT_NAME shift operator by `offset` bits. */ \
  GREX_ALWAYS_INLINE Vector& operator OP##=(AnyIndexTag auto offset) REQ { \
    return *this = *this OP offset; \
  }

  GREX_VECTOR_SHOP(<<, , shift_left, left)
  GREX_VECTOR_SHOP(>>, , shift_right, right)
#undef GREX_VECTOR_SHOP

  /** Zeroes out lanes starting at `i`. */
  GREX_ALWAYS_INLINE Vector cutoff(std::size_t i) const {
    return Vector{backend::cutoff(i, vec_)};
  }

  /** Converts to another vectorizable type with the same lane count. */
  template<Vectorizable TDst>
  GREX_ALWAYS_INLINE Vector<TDst, tSize> convert(TypeTag<TDst> /*tag*/ = {}) const {
    return Vector<TDst, tSize>{backend::convert(vec_, type_tag<TDst>)};
  }

  /** Returns lane `i`. */
  GREX_ALWAYS_INLINE T operator[](std::size_t i) const {
    return backend::extract(vec_, i);
  }

  /** Returns lane `i` with a compile-time index. */
  GREX_ALWAYS_INLINE T operator[](AnyIndexTag auto i) const {
    return backend::extract(vec_, i);
  }

  /** Returns lane `tIdx` for tuple-like access. */
  template<std::size_t tIdx>
  GREX_ALWAYS_INLINE friend T get(const Vector& v) {
    return backend::extract(v.vec_, index_tag<tIdx>);
  }

  /** Returns a copy with lane `i` replaced by `value`. */
  GREX_ALWAYS_INLINE Vector insert(std::size_t i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }

  /** Returns a copy with lane `i` replaced by `value` with a compile-time index. */
  GREX_ALWAYS_INLINE Vector insert(AnyIndexTag auto i, T value) const {
    return Vector{backend::insert(vec_, i, value)};
  }

  /** Stores all lanes to unaligned memory. */
  GREX_ALWAYS_INLINE void store(T* value) const {
    backend::store(value, vec_);
  }

  /** Stores all lanes to aligned memory. */
  GREX_ALWAYS_INLINE void store_aligned(T* value) const {
    backend::store_aligned(value, vec_);
  }

  /** Stores the first `num` elements to unaligned memory. */
  GREX_ALWAYS_INLINE void store_part(T* value, std::size_t num) const {
    backend::store_part(value, vec_, num);
  }

#define GREX_VECTOR_CMP_BINOP(OP, REQ, BACKEND, COMMENT_NAME) \
  /** Lane-wise COMMENT_NAME comparison between vectors. */ \
  GREX_ALWAYS_INLINE friend Mask operator OP(Vector a, Vector b) REQ { \
    return Mask{BACKEND}; \
  } \
  /** Lane-wise COMMENT_NAME comparison with a scalar on the right. */ \
  GREX_ALWAYS_INLINE friend Mask operator OP(Vector a, Value b) REQ { \
    return a OP Vector{b}; \
  } \
  /** Lane-wise COMMENT_NAME comparison with a scalar on the left. */ \
  GREX_ALWAYS_INLINE friend Mask operator OP(Value a, Vector b) REQ { \
    return Vector{a} OP b; \
  }

  GREX_VECTOR_CMP_BINOP(==, , backend::compare_eq(a.vec_, b.vec_), _equal to_)
  GREX_VECTOR_CMP_BINOP(!=, , backend::compare_neq(a.vec_, b.vec_), _not equal to_)
  GREX_VECTOR_CMP_BINOP(<, , backend::compare_lt(a.vec_, b.vec_), _less than_)
  GREX_VECTOR_CMP_BINOP(>, , backend::compare_lt(b.vec_, a.vec_), _greater than_)
  GREX_VECTOR_CMP_BINOP(<=, , backend::compare_ge(b.vec_, a.vec_), _less than or equal to_)
  GREX_VECTOR_CMP_BINOP(>=, , backend::compare_ge(a.vec_, b.vec_), _greater than or equal to_)
#undef GREX_VECTOR_CMP_BINOP

  /** Expands this vector to size `tDstSize` with undefined upper lanes. */
  template<std::size_t tDstSize>
  GREX_ALWAYS_INLINE Vector<T, tDstSize> expand_any(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_any(vec_, index_tag<tDstSize>)};
  }

  /** Expands this vector to size `tDstSize` with upper lanes filled with zeros. */
  template<std::size_t tDstSize>
  GREX_ALWAYS_INLINE Vector<T, tDstSize> expand_zero(IndexTag<tDstSize> /*size*/) const {
    return Vector<T, tDstSize>{backend::expand_zero(vec_, index_tag<tDstSize>)};
  }

  /**
   * Shifts values up by one lane and inserts a zero into the first lane:
   * `result[i] = (i > 0) ? (*this)[i - 1] : 0`.
   */
  GREX_ALWAYS_INLINE Vector shingle_up() const {
    return Vector{backend::shingle_up(vec_)};
  }

  /**
   * Shifts values up by one lane and inserts `front` into the first lane:
   * `result[i] = (i > 0) ? (*this)[i - 1] : front`.
   */
  GREX_ALWAYS_INLINE Vector shingle_up(Value front) const {
    return Vector{backend::shingle_up(backend::Scalar<T>{front}, vec_)};
  }

  /**
   * Shifts values down by one lane and inserts a zero into the last lane:
   * `result[i] = (i + 1 < size) ? (*this)[i + 1] : 0`.
   */
  GREX_ALWAYS_INLINE Vector shingle_down() const {
    return Vector{backend::shingle_down(vec_)};
  }

  /**
   * Shifts values down by one lane and inserts `back` into the last lane:
   * `result[i] = (i + 1 < size) ? (*this)[i + 1] : back`.
   */
  GREX_ALWAYS_INLINE Vector shingle_down(Value back) const {
    return Vector{backend::shingle_down(vec_, backend::Scalar<T>{back})};
  }

  /** Returns underlying backend vector. */
  GREX_ALWAYS_INLINE Backend backend() const {
    return vec_;
  }

  /** Returns contents as `std::array<Value, size>`. */
  GREX_ALWAYS_INLINE std::array<Value, size> as_array() const {
    return backend::to_array(vec_);
  }

private:
  using Base::vec_;
};

/** Trait indicating whether `T` is a `Mask` type. */
template<typename T>
struct MaskTrait : public std::false_type {};

/** `MaskTrait` specialization for `Mask`. */
template<Vectorizable T, std::size_t tSize>
struct MaskTrait<Mask<T, tSize>> : public std::true_type {
  /** Corresponding vector type. */
  using VectorFor = Vector<T, tSize>;
};

/** Trait indicating whether `T` is a `Vector` type. */
template<typename T>
struct VectorTrait : public std::false_type {};

/** `VectorTrait` specialization for `Vector`. */
template<Vectorizable T, std::size_t tSize>
struct VectorTrait<Vector<T, tSize>> : public std::true_type {
  /** Corresponding mask type. */
  using MaskFor = Mask<T, tSize>;
};

/** Concept for any `Mask` type. */
template<typename T>
concept AnyMask = MaskTrait<T>::value;

/** Concept for a mask with a specific lane count. */
template<typename T, std::size_t tSize>
concept SizedMask = AnyMask<T> && T::size == tSize;

/** Concept for any `Vector` type. */
template<typename TVec>
concept AnyVector = VectorTrait<TVec>::value;

/** Concept for a `Vector` whose value type is `TVal`. */
template<typename TVec, typename TVal>
concept ValuedVector = AnyVector<TVec> && std::same_as<typename TVec::Value, TVal>;

/** Concept for a `Vector` with a specific lane count. */
template<typename TVec, std::size_t tSize>
concept SizedVector = AnyVector<TVec> && TVec::size == tSize;

/** Concept for a `Vector` with integer elements. */
template<typename TVec>
concept IntVector = AnyVector<TVec> && IntVectorizable<typename TVec::Value>;

/** Concept for a `Vector` with floating-point elements. */
template<typename TVec>
concept FpVector = AnyVector<TVec> && FloatVectorizable<typename TVec::Value>;

/** Convenience alias for the `Mask` type corresponding to a `Vector` type. */
template<AnyVector TVec>
using MaskFor = VectorTrait<TVec>::MaskFor;

/** Convenience alias for the `Vector` type corresponding to a `Mask` type. */
template<AnyMask TMask>
using VectorFor = MaskTrait<TMask>::VectorFor;

/** Lane-wise logical _AND NOT_ between two masks. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Mask<T, tSize> andnot(Mask<T, tSize> a, Mask<T, tSize> b) {
  return Mask<T, tSize>{backend::logical_andnot(a.backend(), b.backend())};
}

/** Lane-wise absolute value. */
template<SignedVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> abs(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::abs(v.backend())};
}

/** Lane-wise square root. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> sqrt(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::sqrt(v.backend())};
}

/** Lane-wise minimum. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> min(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::min(a.backend(), b.backend())};
}

/** Lane-wise maximum. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> max(Vector<T, tSize> a, Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::max(a.backend(), b.backend())};
}

/** Returns mask of lanes with finite values. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Mask<T, tSize> is_finite(Vector<T, tSize> v) {
  return Mask<T, tSize>{backend::is_finite(v.backend())};
}

/** Replaces non-finite lanes (not-a-number/infinities) with finite values. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> make_finite(Vector<T, tSize> v) {
  return Vector<T, tSize>{backend::make_finite(v.backend())};
}

/** Horizontal sum of all lanes. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_add(Vector<T, tSize> v) {
  return backend::horizontal_add(v.backend());
}

/** Horizontal minimum across all lanes. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_min(Vector<T, tSize> v) {
  return backend::horizontal_min(v.backend());
}

/** Horizontal maximum across all lanes. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T horizontal_max(Vector<T, tSize> v) {
  return backend::horizontal_max(v.backend());
}

/** Horizontal logical _AND_ over all mask lanes. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline bool horizontal_and(Mask<T, tSize> m) {
  return backend::horizontal_and(m.backend());
}

/** Lane-wise fused multiply-add: @f$ a \cdot b + c @f$. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fmadd(Vector<T, tSize> a, Vector<T, tSize> b,
                                                 Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmadd(a.backend(), b.backend(), c.backend())};
}

/** Lane-wise fused multiply-subtract: @f$ a \cdot b - c @f$. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fmsub(Vector<T, tSize> a, Vector<T, tSize> b,
                                                 Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fmsub(a.backend(), b.backend(), c.backend())};
}

/** Lane-wise negative fused multiply-add: @f$ -(a \cdot b) + c @f$. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fnmadd(Vector<T, tSize> a, Vector<T, tSize> b,
                                                  Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmadd(a.backend(), b.backend(), c.backend())};
}

/** Lane-wise negative fused multiply-subtract: @f$ -(a \cdot b) - c @f$. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> fnmsub(Vector<T, tSize> a, Vector<T, tSize> b,
                                                  Vector<T, tSize> c) {
  return Vector<T, tSize>{backend::fnmsub(a.backend(), b.backend(), c.backend())};
}

/** Extracts `v[0]`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline T extract_single(Vector<T, tSize> v) {
  return backend::extract_single(v.backend()).value;
}

/** Blends `v1` with zeros: `result[i] = mask[i] ? v1[i] : 0`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend_zero(Mask<T, tSize> mask, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero(mask.backend(), v1.backend())};
}

/** Blends `v1` with zeros using compile-time selectors. */
template<BlendZeroSelector... tBzs, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend_zero(Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend_zero<tBzs...>(v1.backend())};
}

/** Blends between `v0` and `v1`: `result[i] = mask[i] ? v1[i] : v0[i]`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend(Mask<T, tSize> mask, Vector<T, tSize> v0,
                                                 Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend(mask.backend(), v0.backend(), v1.backend())};
}

/** Blends between `v0` and `v1` using compile-time selectors. */
template<BlendSelector... tBls, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> blend(Vector<T, tSize> v0, Vector<T, tSize> v1) {
  return Vector<T, tSize>{backend::blend<tBls...>(v0.backend(), v1.backend())};
}

/**
 * Shuffles `table` using the indices in `idxs`: `result[i] = table[idxs[i]]`.
 *
 * The value in `result[i]` is undefined if `idxs[i] >= size`.
 */
template<Vectorizable T, UnsignedIntVectorizable TIdx, std::size_t tTableSize, std::size_t tIdxSize>
GREX_ALWAYS_INLINE inline Vector<T, tIdxSize> shuffle(Vector<T, tTableSize> table,
                                                      Vector<TIdx, tIdxSize> idxs) {
  return Vector<T, tIdxSize>{backend::shuffle(table.backend(), idxs.backend())};
}

/** Shuffles `table` using compile-time indices. */
template<ShuffleIndex... tIdxs, Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> shuffle(Vector<T, tSize> table) {
  return Vector<T, tSize>{backend::shuffle<tIdxs...>(table.backend())};
}

/** Masked add: `result[i] = mask[i] ? a[i] + b[i] : a[i]`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_add(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                    Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_add(mask.backend(), a.backend(), b.backend())};
}

/** Masked subtract: `result[i] = mask[i] ? a[i] - b[i] : a[i]`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_subtract(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                         Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_subtract(mask.backend(), a.backend(), b.backend())};
}

/** Masked multiply: `result[i] = mask[i] ? a[i] * b[i] : a[i]`. */
template<Vectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_multiply(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                         Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_multiply(mask.backend(), a.backend(), b.backend())};
}

/** Masked divide: `result[i] = mask[i] ? a[i] / b[i] : a[i]`. */
template<FloatVectorizable T, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<T, tSize> mask_divide(Mask<T, tSize> mask, Vector<T, tSize> a,
                                                       Vector<T, tSize> b) {
  return Vector<T, tSize>{backend::mask_divide(mask.backend(), a.backend(), b.backend())};
}

/** Gathers elements from `data` at `indices` into a vector: `result[i] = data[indices[i]]`. */
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<TValue, tSize> gather(std::span<const TValue, tExtent> data,
                                                       Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::gather(data, indices.backend())};
}

/**
 * Gathers values from `data` at `indices` where `mask` is set:
 * `result[i] = mask[i] ? data[indices[i]] : 0`.
 */
template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize>
GREX_ALWAYS_INLINE inline Vector<TValue, tSize> mask_gather(std::span<const TValue, tExtent> data,
                                                            Mask<TValue, tSize> mask,
                                                            Vector<TIndex, tSize> indices) {
  return Vector<TValue, tSize>{backend::mask_gather(data, mask.backend(), indices.backend())};
}
} // namespace grex

/** `tuple_size` specialization for `grex::Vector` (for tuple-like access). */
template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Vector<T, tSize>> : public std::integral_constant<std::size_t, tSize> {
};

/** `tuple_element` specialization for `grex::Vector` (for tuple-like access). */
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Vector<T, tSize>> {
  /** Element type at index `tIdx`. */
  using type = const T; // NOLINT
};

/** `tuple_size` specialization for `grex::Mask` (for tuple-like access). */
template<grex::Vectorizable T, std::size_t tSize>
struct std::tuple_size<grex::Mask<T, tSize>> : public std::integral_constant<std::size_t, tSize> {};

/** `tuple_element` specialization for `grex::Mask` (for tuple-like access). */
template<std::size_t tIdx, grex::Vectorizable T, std::size_t tSize>
struct std::tuple_element<tIdx, grex::Mask<T, tSize>> {
  /** Element type at index `tIdx`. */
  using type = const bool; // NOLINT
};
#endif

#endif // INCLUDE_GREX_TYPES_HPP
