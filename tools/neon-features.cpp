// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <print>

#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <cstddef>
#include <cstdint>

#include <sys/sysctl.h>
#endif

namespace {
/**
 * Whether the CPU supports the ARMv8.2-A FP16 extension, i.e. half-precision arithmetic in both
 * scalar and vector form, which is what `-march=armv8-a+fp16` enables. Systems whose feature
 * detection is not implemented are reported as not supporting it, which merely means that the
 * corresponding tests are not built.
 */
[[nodiscard]] bool has_fp16() {
#if defined(__linux__)
  const unsigned long hwcap = getauxval(AT_HWCAP);
  return (hwcap & HWCAP_FPHP) != 0 && (hwcap & HWCAP_ASIMDHP) != 0;
#elif defined(__APPLE__)
  std::int32_t supported{};
  std::size_t size = sizeof(supported);
  if (sysctlbyname("hw.optional.arm.FEAT_FP16", &supported, &size, nullptr, 0) != 0) {
    return false;
  }
  return supported != 0;
#else
  return false;
#endif
}
} // namespace

int main() {
  // The single line of output holds the marches to build for, joined by ";": the ARMv8-a baseline
  // is always included, the FP16 extension only if the CPU supports it.
  std::print("armv8-a");
  if (has_fp16()) {
    std::print(";armv8-a+fp16");
  }
  std::print("\n");
}
