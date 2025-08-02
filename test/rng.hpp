// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "grex/base/defs.hpp"

// Based on https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c
// Originally licensed under the Apache License, Version 2.0.

namespace grex::test {
struct Pcg32 {
  // RNG state. All values are possible.
  u64 state = 0x853c49e6748fea9b;
  // Controls which RNG sequence (stream) is selected. Must be odd.
  u64 inc = 0xda3e39cb94b95bdb;

  constexpr Pcg32() = default;
  constexpr Pcg32(u64 initstate, u64 initseq) : state{}, inc{(initseq << 1U) | 1U} {
    random();
    state += initstate;
    random();
  }

  constexpr u32 random() {
    u64 oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    auto xorshifted = u32(((oldstate >> 18U) ^ oldstate) >> 27U);
    u32 rot = oldstate >> 59U;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  constexpr u32 bounded_random(u32 bound) {
    // To avoid bias, we need to make the range of the RNG a multiple of
    // bound, which we do by dropping output less than a threshold.
    // A naive scheme to calculate the threshold would be to do
    //   u32 threshold = 0x100000000ull % bound;
    // but 64-bit div/mod is slower than 32-bit div/mod (especially on
    // 32-bit platforms). In essence, we do
    //   u32 threshold = (0x100000000ull-bound) % bound;
    // because this version will calculate the same modulus, but the LHS
    // value is less than 2^32.
    u32 threshold = -bound % bound;
    // Uniformity guarantees that this loop will terminate. In practice, it
    // should usually terminate quickly; on average (assuming all bounds are
    // equally likely), 82.25% of the time, we can expect it to require just
    // one iteration. In the worst case, someone passes a bound of 2^31 + 1
    // (i.e., 2147483649), which invalidates almost 50% of the range. In
    // practice, bounds are typically small and only a tiny amount of the range
    // is eliminated.
    while (true) {
      const u32 r = random();
      if (r >= threshold) {
        return r % bound;
      }
    }
  }
};
} // namespace grex::test
