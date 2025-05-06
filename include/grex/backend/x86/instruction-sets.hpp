// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
#define INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP

namespace grex::backend {
// Define the instruction sets as values which are used later
#define GREX_SSE2 2
#define GREX_SSE3 3
#define GREX_SSSE3 4
#define GREX_SSE4_1 5
#define GREX_SSE4_2 6
#define GREX_AVX 7
#define GREX_AVX2 8
#define GREX_AVX512_1 9 // AVX-512 F/CD (de facto always present with AVX-512)
#define GREX_AVX512_2 10 // AVX-512 VL/DQ/BW (de factor all or none are present)

#ifndef GREX_INSTRUCTION_SET
#if defined(__AVX512VL__) && defined(__AVX512DQ__) && defined(__AVX512BW__)
#define GREX_INSTRUCTION_SET GREX_AVX512_2
#elif defined(__AVX512F__) || defined(__AVX512__)
#define GREX_INSTRUCTION_SET GREX_AVX512_1
#elif defined(__AVX2__)
#define GREX_INSTRUCTION_SET GREX_AVX2
#elif defined(__AVX__)
#define GREX_INSTRUCTION_SET GREX_AVX
#elif defined(__SSE4_2__)
#define GREX_INSTRUCTION_SET GREX_SSE4_2
#elif defined(__SSE4_1__)
#define GREX_INSTRUCTION_SET GREX_SSE4_1
#elif defined(__SSSE3__)
#define GREX_INSTRUCTION_SET GREX_SSSE3
#elif defined(__SSE3__)
#define GREX_INSTRUCTION_SET GREX_SSE3
#elif defined(__SSE2__) || defined(__x86_64__)
#define GREX_INSTRUCTION_SET GREX_SSE2
#else
#error "At least SSE2 is required!"
#endif
#endif // GREX_INSTRUCTION_SET
} // namespace grex::backend

#endif // INCLUDE_GREX_BACKEND_X86_INSTRUCTION_SETS_HPP
