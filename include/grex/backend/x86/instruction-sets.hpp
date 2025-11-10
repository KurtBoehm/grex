// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
#define INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP

#ifndef GREX_X86_64_LEVEL
#if defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#define GREX_X86_64_LEVEL 4
#elif defined(__AVX2__)
#define GREX_X86_64_LEVEL 3
#elif defined(__SSE4_2__)
#define GREX_X86_64_LEVEL 2
#elif defined(__SSE2__) || defined(__x86_64__)
#define GREX_X86_64_LEVEL 1
#else
#error "Only x86-64 is supported, which implies SSE2!"
#endif
#endif

#if __AVX512VBMI__
#define GREX_HAS_AVX512VBMI true
#else
#define GREX_HAS_AVX512VBMI false
#endif
#if __AVX512VBMI2__
#define GREX_HAS_AVX512VBMI2 true
#else
#define GREX_HAS_AVX512VBMI2 false
#endif

#endif // INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
