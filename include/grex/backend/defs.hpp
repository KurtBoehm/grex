#ifndef INCLUDE_GREX_BACKEND_DEFS_HPP
#define INCLUDE_GREX_BACKEND_DEFS_HPP

#if !defined(GREX_BACKEND_X86_64) && !defined(GREX_BACKEND_NEON) && !defined(GREX_BACKEND_SCALAR)
// Auto-detect the most appropriate backend
#if defined(__x86_64__) || defined(_M_X64)
#define GREX_BACKEND_X86_64 true
#elif defined(__aarch64__) || defined(_M_ARM64)
#define GREX_BACKEND_NEON true
#else
#define GREX_BACKEND_SCALAR true
#endif
#endif

// Whether binary16 arithmetic is carried out by dedicated instructions rather than in binary32.
// On ARM64 this requires the FP16 extension (ARMv8.2-A). On x86-64 it requires the AVX512-FP16
// extension, which is not part of any microarchitecture level and is therefore only used if
// explicitly enabled, e.g. via `-mavx512fp16` or `-march=sapphirerapids`.
#ifndef GREX_F16_NATIVE_ARITHMETIC
#if GREX_BACKEND_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define GREX_F16_NATIVE_ARITHMETIC true
#elif GREX_BACKEND_X86_64 && defined(__AVX512FP16__)
#define GREX_F16_NATIVE_ARITHMETIC true
#else
#define GREX_F16_NATIVE_ARITHMETIC false
#endif
#endif

#endif // INCLUDE_GREX_BACKEND_DEFS_HPP
