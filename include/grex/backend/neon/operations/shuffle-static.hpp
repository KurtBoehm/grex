// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
#define INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP

#include <cassert>
#include <cstddef>
#include <limits>
#include <optional>

#include "grex/backend/base.hpp"
#include "grex/backend/defs.hpp" // IWYU pragma: keep
#include "grex/backend/macros/for-each.hpp"
#include "grex/backend/macros/types.hpp"
#include "grex/backend/neon/macros/types.hpp"
#include "grex/backend/neon/operations/extract.hpp"
#include "grex/backend/neon/operations/set.hpp"
#include "grex/backend/shared/defs.hpp"
#include "grex/backend/shared/operations/shuffle-static.hpp"
#include "grex/base.hpp"

namespace grex::backend {
struct ShufflerTbl : BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr auto shuf = convert<1>(SI).value();
    return {.r = as<Value>(vqtbl1q_u8(as<u8>(vec.r), shuf.vector(false_tag).r))};
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 1, .latency = 8};
  }
};

/**
 * A shuffle that resolves to a single `EXT`, which concatenates its two operands, shifts the
 * resulting 32 bytes down by an immediate, and extracts the low 16 bytes.
 *
 * This shuffler supports any `EXT` pattern in which the input vector `v` is at least one of the
 * operands and the zero vector may be the other operand:
 * - `[v, 0]`: Components shifted towards the beginning, zeros fill the end.
 * - `[0, v]`: Components shifted towards the end, zeros fill the beginning.
 * - `[v, v]`: Components are rotated.
 */
struct ShufflerExt : BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return args<SI>().has_value();
  }

  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    static constexpr Args a = args<SI>().value();

    const uint8x16_t v = as<u8>(vec.r);
    const auto ext = vextq_u8(operand(v, auto_tag<a.lo>), operand(v, auto_tag<a.hi>), a.offset);
    return {.r = as<ValueOf<Vec>>(ext)};
  }

  /**
   * One `EXT`, which has a latency of 2 and is present on each of the four SIMD execution units on
   * each performance core on Apple Silicon, according to the Apple Silicon CPU Optimization Guide.
   * The zero register, if needed, is usually hoisted out of loops and ignored here.
   */
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 0.25, .latency = 2};
  }

private:
  /** The kind of either operand: either the input vector or zero. */
  enum struct Operand : u8 { vector, zero };

  /** The operands and the byte offset of the `EXT` that realizes a given shuffle. */
  struct Args {
    Operand lo;
    Operand hi;
    u8 offset;
  };

  /** The parameters of the `EXT` that realizes `SI`, if there is one. */
  template<AnyShuffleIndices auto SI>
  static constexpr std::optional<Args> args() {
    // `EXT` operates on bytes; the intrinsics for larger types resolve to the byte-level
    // instruction.
    static constexpr auto idxs = convert<1>(SI).value();
    static constexpr std::size_t size = 16;
    static_assert(idxs.size == size);

    // Determine the offset, check whether the input vector is accessed via either operand, and
    // store the indices to be zeroed in a bit mask.
    std::optional<u8> offset{};
    bool lo_value = false;
    bool hi_value = false;
    u16 zero_mask = 0;
    for (std::size_t i = 0; i < size; ++i) {
      const ShuffleIndex si = idxs[i];
      switch (si) {
        case any_sh: continue;
        case zero_sh: {
          zero_mask |= static_cast<u16>(1U << i);
          continue;
        }
        default: {
          // All other indices reference the input vector.
          const u8 idx = static_cast<u8>(si);
          const u8 offset_i = static_cast<u8>((size + idx - i) % size);

          if (idx == i || (offset.has_value() && offset != offset_i)) {
            // A no-op mapping is left to other shufflers: `EXT` is unnecessary in this case.
            // If a different offset was determined earlier, the pattern is not supported.
            return std::nullopt;
          }
          offset = offset_i;

          // The value is from the first operand if `idx > i` and vice versa.
          lo_value = lo_value || idx > i;
          hi_value = hi_value || idx < i;
        }
      }
    }

    if (!offset.has_value() || (lo_value && hi_value && zero_mask != 0)) {
      // If the input vector was not accessed at all, either the input vector or the zero vector is
      // a more efficient alternative, which are handled by `ShufflerBlendZero`.
      // If both halves access the input vector, they cannot contain zeros.
      return std::nullopt;
    }

    if (zero_mask == 0) {
      // If there are no zeros at all, a rotation always works just fine and does not require a zero
      // register.
      return Args{.lo = Operand::vector, .hi = Operand::vector, .offset = *offset};
    }

    // Determine which operand contains at least one zero. If either contains both zeros
    // and accesses the input vector, the pattern is unsupported; otherwise, resolve to the
    // respective shift pattern.
    const auto lo_mask = static_cast<u16>(std::numeric_limits<u16>::max() >> *offset);
    const bool lo_zero = (zero_mask & lo_mask) != 0;
    const bool hi_zero = (zero_mask & ~lo_mask) != 0;

    if ((lo_zero && lo_value) || (hi_zero && hi_value)) {
      // One operand cannot contain both input values and zeros.
      return std::nullopt;
    }

    return Args{
      .lo = lo_zero ? Operand::zero : Operand::vector,
      .hi = hi_zero ? Operand::zero : Operand::vector,
      .offset = *offset,
    };
  }

  /** Determines one of the two operands for `EXT`, which is the input vector. */
  GREX_ALWAYS_INLINE static uint8x16_t operand(uint8x16_t v, AutoTag<Operand::vector> /*tag*/) {
    return v;
  }
  /** Determines one of the two operands for `EXT`, which is zero. */
  GREX_ALWAYS_INLINE static uint8x16_t operand(uint8x16_t /*v*/, AutoTag<Operand::zero> /*tag*/) {
    return vdupq_n_u8(0);
  }
};

#define GREX_DUP_I(KIND, BITS, SIZE, REGKIND) \
  template<int Lane> \
  inline KIND##BITS##x##SIZE duplicate_lane(KIND##BITS##x##SIZE v) { \
    return {.r = GREX_ISUFFIXED(vdupq_laneq, REGKIND, BITS)(v.r, Lane)}; \
  }
#define GREX_DUP(KIND, BITS, SIZE) GREX_DUP_I(KIND, BITS, SIZE, GREX_REGKIND(KIND, BITS))
GREX_FOREACH_TYPE_EXT(GREX_DUP, 128)

struct ShufflerDup : BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    static constexpr std::optional<ShuffleIndex> constant = SI.constant();
    return constant.has_value() && is_index(*constant);
  }

  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    return duplicate_lane<int(SI.constant().value())>(vec);
  }

  /**
   * One `DUP` (element), which has a latency of 2 and is present on each of the four SIMD execution
   * units on each performance core on Apple Silicon, according to the Apple Silicon CPU
   * Optimization Guide.
   */
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 0.25, .latency = 2};
  }
};

struct ShufflerExtractSet : BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return true;
  }
  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec vec, AutoTag<SI> /*tag*/) {
    using Value = Vec::Value;
    static constexpr std::size_t size = Vec::size;

    auto f = [&](std::size_t i) { return is_index(SI[i]) ? extract(vec, u8(SI[i])) : Value{}; };
    return static_apply<size>([&]<std::size_t... I> { return set(type_tag<Vec>, f(I)...); });
  }
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 1, .latency = 8};
  }
};

template<AnyShuffleIndices auto I>
requires((I.value_size * I.size == 16)) // NOLINT(*-redundant-parentheses)
struct ShufflerTrait<I> {
  using Shuffler =
    CheapestType<I, ShufflerDup, ShufflerExt, ShufflerBlendZero, ShufflerTbl, ShufflerExtractSet>;
};

/**
 * A pair shuffle that resolves to a single `EXT`, which concatenates its two operands, shifts the
 * resulting 32 bytes down by an immediate, and extracts the low 16 bytes.
 *
 * This shuffler supports any pattern achievable using `EXT` with the two arguments passed in either
 * order.
 */
struct PairShufflerExt : BaseExpensiveOp {
  template<AnyShuffleIndices auto SI>
  static constexpr bool is_applicable(AutoTag<SI> /*tag*/) {
    return offset<SI>().has_value();
  }

  template<AnyVector Vec, ShuffleIndicesFor<Vec> SI>
  static Vec apply(Vec lo, Vec hi, AutoTag<SI> /*tag*/) {
    static constexpr u8 off = offset<SI>().value();

    if constexpr (off > 16) {
      return {.r = as<ValueOf<Vec>>(vextq_u8(as<u8>(lo.r), as<u8>(hi.r), off - 16))};
    } else {
      return {.r = as<ValueOf<Vec>>(vextq_u8(as<u8>(hi.r), as<u8>(lo.r), off))};
    }
  }

  /**
   * One `EXT`, which has a latency of 2 and is present on each of the four SIMD execution units on
   * each performance core on Apple Silicon, according to the Apple Silicon CPU Optimization Guide.
   */
  template<AnyShuffleIndices auto SI>
  static constexpr Cost cost(AutoTag<SI> /*idxs*/) {
    return {.inv_throughput = 0.25, .latency = 2};
  }

private:
  /**
   * The byte offset of the `EXT` that realizes a given shuffle, if there is one, with a bias of 16.
   * An `offset ≥ 16` corresponds to `EXT dst, a, b, #(offset - 16)` whereas `offset < 16`
   * corresponds to `EXT dst, b, a, #offset`.
   */
  template<AnyShuffleIndices auto SI>
  static constexpr std::optional<u8> offset() {
    // `EXT` operates on bytes; the intrinsics for larger types resolve to the byte-level
    // instruction.
    static constexpr auto idxs = convert<1>(SI).value();
    static constexpr std::size_t size = 16;
    static constexpr std::size_t index_ub = 32;
    static_assert(idxs.size == size);

    // Determine the biased offset.
    std::optional<u8> offset{};
    for (std::size_t i = 0; i < size; ++i) {
      const ShuffleIndex si = idxs[i];
      switch (si) {
        case any_sh: continue;
        case zero_sh: {
          // Zeroing is not supported.
          return std::nullopt;
        }
        default: {
          const u8 idx = static_cast<u8>(si);
          assert(idx < index_ub);
          const u8 offset_i = static_cast<u8>((size + idx - i) % index_ub);

          if (offset.has_value() && offset != offset_i) {
            // If a different offset was determined earlier, the pattern is not supported.
            return std::nullopt;
          }
          offset = offset_i;
        }
      }
    }

    if (!offset.has_value() || *offset == 0 || *offset == size) {
      // If neither input vector was accessed, simply returning one of the inputs is more efficient,
      // which are handled elsewhere.
      // If the offset sans bias is 0 or -16, the result is just one of the two arguments and there
      // are better solutions.
      return std::nullopt;
    }

    return offset;
  }
};

template<AnyShuffleIndices auto SI>
requires((SI.value_size * SI.size == 16)) // NOLINT(*-redundant-parentheses)
struct PairShufflerTrait<SI> {
  using Shuffler = CheapestType<SI, PairShufflerSingle, PairShufflerBlend, PairShufflerExt>;
};
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_NEON_OPERATIONS_SHUFFLE_STATIC_HPP
