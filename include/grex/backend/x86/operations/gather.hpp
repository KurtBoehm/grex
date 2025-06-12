// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP

#include <cstddef>
#include <span>

#include "grex/backend/choosers.hpp"
#include "grex/backend/defs.hpp"
#include "grex/backend/x86/helpers.hpp"
#include "grex/backend/x86/instruction-sets.hpp"
#include "grex/backend/x86/operations/extract.hpp"
#include "grex/backend/x86/operations/merge.hpp"
#include "grex/backend/x86/operations/set.hpp"
#include "grex/backend/x86/types.hpp" // IWYU pragma: keep
#include "grex/base/defs.hpp"

#if GREX_X86_64_LEVEL >= 3
#include "grex/backend/x86/operations/convert.hpp"
#endif

namespace grex::backend {
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tSize>
inline VectorFor<TValue, tSize> gather(std::span<const TValue> data, Vector<TIndex, tSize> idxs) {
  return static_apply<tSize>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tSize>>, data[std::size_t(extract(idxs, tIdxs))]...);
  });
}
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tPart, std::size_t tSize>
inline VectorFor<TValue, tPart> gather(std::span<const TValue> data,
                                       SubVector<TIndex, tPart, tSize> idxs) {
  return static_apply<tPart>([&]<std::size_t... tIdxs> {
    return set(type_tag<VectorFor<TValue, tPart>>, data[std::size_t(extract(idxs, tIdxs))]...);
  });
}
template<Vectorizable TValue, typename THalf>
inline VectorFor<TValue, 2 * THalf::size> gather(std::span<const TValue> data,
                                                 SuperVector<THalf> idxs) {
  return merge(gather(data, idxs.lower), gather(data, idxs.upper));
}

#define GREX_GATHER_PREFIX(VALUE, INDEX, SIZE) \
  inline VectorFor<VALUE, SIZE> gather(std::span<const VALUE> data, VectorFor<INDEX, SIZE> idxs)

#define GREX_GATHER_CAST_F32 data.data()
#define GREX_GATHER_CAST_F64 data.data()
#define GREX_GATHER_CAST_I32 reinterpret_cast<const int*>(data.data())
#define GREX_GATHER_CAST_I64 reinterpret_cast<const long long*>(data.data())
#define GREX_GATHER_CAST_f(BITS) GREX_GATHER_CAST_F##BITS
#define GREX_GATHER_CAST_i(BITS) GREX_GATHER_CAST_I##BITS
#define GREX_GATHER_CAST_u(BITS) GREX_GATHER_CAST_I##BITS
#define GREX_GATHER_CAST(KIND, BITS) GREX_GATHER_CAST_##KIND(BITS)

#define GREX_GATHER_INSTRINSIC(VALKIND, VALBITS, IDXKIND, IDXBITS, SIZE, REGISTERBITS) \
  GREX_GATHER_PREFIX(VALKIND##VALBITS, IDXKIND##IDXBITS, SIZE) { \
    using Out = VectorFor<VALKIND##VALBITS, SIZE>; \
    return Out{GREX_CAT(GREX_BITPREFIX(REGISTERBITS), _i##IDXBITS##gather_, \
                        GREX_EPI_SUFFIX(VALKIND, VALBITS))( \
      GREX_GATHER_CAST(VALKIND, VALBITS), idxs.registr(), GREX_BIT2BYTE(VALBITS))}; \
  }

// TODO Check in practice whether all of this is useful at all!
#if GREX_X86_64_LEVEL >= 3
// i32 indices
// up to 128 bits
GREX_GATHER_INSTRINSIC(f, 64, i, 32, 2, 128)
GREX_GATHER_INSTRINSIC(i, 64, i, 32, 2, 128)
GREX_GATHER_INSTRINSIC(u, 64, i, 32, 2, 128)
// TODO Add sub-native variants!
GREX_GATHER_INSTRINSIC(f, 32, i, 32, 4, 128)
GREX_GATHER_INSTRINSIC(i, 32, i, 32, 4, 128)
GREX_GATHER_INSTRINSIC(u, 32, i, 32, 4, 128)
// up to 256 bits
GREX_GATHER_INSTRINSIC(f, 64, i, 32, 4, 256)
GREX_GATHER_INSTRINSIC(i, 64, i, 32, 4, 256)
GREX_GATHER_INSTRINSIC(u, 64, i, 32, 4, 256)
GREX_GATHER_INSTRINSIC(f, 32, i, 32, 8, 256)
GREX_GATHER_INSTRINSIC(i, 32, i, 32, 8, 256)
GREX_GATHER_INSTRINSIC(u, 32, i, 32, 8, 256)

// i64 indices
// up to 128 bits
GREX_GATHER_INSTRINSIC(f, 64, i, 64, 2, 128)
GREX_GATHER_INSTRINSIC(i, 64, i, 64, 2, 128)
GREX_GATHER_INSTRINSIC(u, 64, i, 64, 2, 128)
GREX_GATHER_INSTRINSIC(f, 32, i, 64, 2, 128)
GREX_GATHER_INSTRINSIC(i, 32, i, 64, 2, 128)
GREX_GATHER_INSTRINSIC(u, 32, i, 64, 2, 128)
// up to 256 bits
GREX_GATHER_INSTRINSIC(f, 64, i, 64, 4, 256)
GREX_GATHER_INSTRINSIC(i, 64, i, 64, 4, 256)
GREX_GATHER_INSTRINSIC(u, 64, i, 64, 4, 256)
GREX_GATHER_INSTRINSIC(f, 32, i, 64, 4, 256)
GREX_GATHER_INSTRINSIC(i, 32, i, 64, 4, 256)
GREX_GATHER_INSTRINSIC(u, 32, i, 64, 4, 256)

// i64 indices: The virtual address space is far below 63 bits, i.e. we ignore the signedness
// up to 128 bits
GREX_GATHER_INSTRINSIC(f, 64, u, 64, 2, 128)
GREX_GATHER_INSTRINSIC(i, 64, u, 64, 2, 128)
GREX_GATHER_INSTRINSIC(u, 64, u, 64, 2, 128)
GREX_GATHER_INSTRINSIC(f, 32, u, 64, 2, 128)
GREX_GATHER_INSTRINSIC(i, 32, u, 64, 2, 128)
GREX_GATHER_INSTRINSIC(u, 32, u, 64, 2, 128)
// up to 256 bits
GREX_GATHER_INSTRINSIC(f, 64, u, 64, 4, 256)
GREX_GATHER_INSTRINSIC(i, 64, u, 64, 4, 256)
GREX_GATHER_INSTRINSIC(u, 64, u, 64, 4, 256)
GREX_GATHER_INSTRINSIC(f, 32, u, 64, 4, 256)
GREX_GATHER_INSTRINSIC(i, 32, u, 64, 4, 256)
GREX_GATHER_INSTRINSIC(u, 32, u, 64, 4, 256)

// 8- and 16-bit indices: convert to i32
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tSize>
requires(sizeof(TValue) >= 4 && sizeof(TIndex) <= 2)
inline VectorFor<TValue, tSize> gather(std::span<const TValue> data, Vector<TIndex, tSize> idxs) {
  return gather(data, convert(idxs, type_tag<i32>));
}
template<Vectorizable TValue, Vectorizable TIndex, std::size_t tPart, std::size_t tSize>
requires(sizeof(TValue) >= 4 && sizeof(TIndex) <= 2)
inline VectorFor<TValue, tPart> gather(std::span<const TValue> data,
                                       SubVector<TIndex, tPart, tSize> idxs) {
  return gather(data, convert(idxs, type_tag<i32>));
}

// u32:
// - data.size() < 2^31: idxs < 2^31 → cast to i32 is safe
// - data.size() ≥ 2^31: add 2^31 to the base pointer and subtract 2^31 from idxs,
//   which is equivalent to idxs ^ 2^31, which transforms the value range of u32 to i32.
// sadly, the latter cannot be done in general in standard C++ because pointer arithmetic
// past the array leads to undefined behaviour
template<Vectorizable TValue, std::size_t tSize>
requires(sizeof(TValue) >= 4)
inline VectorFor<TValue, tSize> gather(std::span<const TValue> data, Vector<u32, tSize> idxs) {
  constexpr u32 limit = std::size_t{1} << 31;
  if (data.size() >= limit) {
    return gather(
      std::span{data.data() + limit, data.size() - limit},
      convert(bitwise_xor(idxs, broadcast(limit, type_tag<Vector<u32, tSize>>)), type_tag<i32>));
  }
  return gather(data, convert(idxs, type_tag<i32>));
}
#endif
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_GATHER_HPP
