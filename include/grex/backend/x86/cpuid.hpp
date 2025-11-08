// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_CPUID_HPP
#define INCLUDE_GREX_BACKEND_X86_CPUID_HPP

#include <array>
#include <cassert>
#include <stdexcept>

#include <cpuid.h>

namespace grex::backend {
struct CpuId {
  unsigned int eax;
  unsigned int ebx;
  unsigned int ecx;
  unsigned int edx;
};

[[nodiscard]] inline CpuId read_cpuid(unsigned int leaf, unsigned int subleaf) {
  unsigned int eax{};
  unsigned int ebx{};
  unsigned int ecx{};
  unsigned int edx{};
  int ret = __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx);
  if (ret == 0) {
    throw std::runtime_error{"CPUID failed!"};
  }
  return CpuId{.eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx};
}

[[nodiscard]] inline unsigned int runtime_x86_64_level() {
  const CpuId leaf1 = read_cpuid(1, 0);
  const CpuId leaf7_0 = read_cpuid(7, 0);
  const CpuId leaf81 = read_cpuid(0x80000001, 0);

  auto read_bit = [](unsigned int value, unsigned int mask) { return (value & mask) != 0; };

  // level 1
  const bool cmov = read_bit(leaf1.edx, bit_CMOV);
  const bool cx8 = read_bit(leaf1.edx, bit_CMPXCHG8B);
  const bool fpu = read_bit(leaf1.edx, 1U << 1);
  const bool fxsr = read_bit(leaf1.edx, bit_FXSAVE);
  const bool mmx = read_bit(leaf1.edx, bit_MMX);
  // OSFXSR is about OS support, not CPUID, and is not checked
  const bool osfxsr = read_bit(leaf1.edx, bit_FXSAVE);
  const bool sce = read_bit(leaf81.edx, 1U << 11);
  const bool sse = read_bit(leaf1.edx, bit_SSE);
  const bool sse2 = read_bit(leaf1.edx, bit_SSE2);
  // level 2
  const bool cmpxchg16b = read_bit(leaf1.ecx, bit_CMPXCHG16B);
  const bool lahf_sahf = read_bit(leaf81.ecx, bit_LAHF_LM);
  const bool popcnt = read_bit(leaf1.ecx, bit_POPCNT);
  const bool sse3 = read_bit(leaf1.ecx, bit_SSE3);
  const bool sse4_1 = read_bit(leaf1.ecx, bit_SSE4_1);
  const bool sse4_2 = read_bit(leaf1.ecx, bit_SSE4_2);
  const bool ssse3 = read_bit(leaf1.ecx, bit_SSSE3);
  // level 3
  const bool avx = read_bit(leaf1.ecx, bit_AVX);
  const bool avx2 = read_bit(leaf7_0.ebx, bit_AVX2);
  const bool bmi1 = read_bit(leaf7_0.ebx, bit_BMI);
  const bool bmi2 = read_bit(leaf7_0.ebx, bit_BMI2);
  const bool f16c = read_bit(leaf1.ecx, bit_F16C);
  const bool fma = read_bit(leaf1.ecx, bit_FMA);
  const bool lzcnt = read_bit(leaf81.ecx, bit_ABM);
  const bool movbe = read_bit(leaf1.ecx, bit_MOVBE);
  const bool osxsave = read_bit(leaf1.ecx, bit_XSAVE);
  // level 4
  const bool avx512f = read_bit(leaf7_0.ebx, bit_AVX512F);
  const bool avx512bw = read_bit(leaf7_0.ebx, bit_AVX512BW);
  const bool avx512cd = read_bit(leaf7_0.ebx, bit_AVX512CD);
  const bool avx512dq = read_bit(leaf7_0.ebx, bit_AVX512DQ);
  const bool avx512vl = read_bit(leaf7_0.ebx, bit_AVX512VL);

  const std::array<bool, 4> level_conditions{
    /*level 1*/ cmov && cx8 && fpu && fxsr && mmx && osfxsr && sce && sse && sse2,
    /*level 2*/ cmpxchg16b && lahf_sahf && popcnt && sse3 && sse4_1 && sse4_2 && ssse3,
    /*level 3*/ avx && avx2 && bmi1 && bmi2 && f16c && fma && lzcnt && movbe && osxsave,
    /*level 4*/ avx512f && avx512bw && avx512cd && avx512dq && avx512vl,
  };
  assert(level_conditions[0]);

  for (unsigned int i = 0; i < 4; ++i) {
    if (!level_conditions[i]) {
      return i;
    }
  }
  return 4;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_CPUID_HPP
