// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_UTILS_HPP
#define INCLUDE_GREX_BACKEND_X86_UTILS_HPP

#include "grex/base.hpp"

namespace grex::backend {
/**
 * Clang re-optimizes shuffle intrinsics; in one of the two-step sequences below, it would trade the
 * constant-free sequence for one with a constant-pool vector. An empty `asm` on the intermediate
 * value stops that and costs no instruction.
 */
template<typename T>
GREX_ALWAYS_INLINE inline T pin(T x) {
#if GREX_CLANG
  __asm__("" : "+v"(x));
#endif
  return x;
}
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_UTILS_HPP
