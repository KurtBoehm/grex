#ifndef INCLUDE_GREX_BACKEND_DEFS_HPP
#define INCLUDE_GREX_BACKEND_DEFS_HPP

#if !defined(GREX_BACKEND_X86_64) && !defined(GREX_BACKEND_NEON) && !defined(GREX_BACKEND_SCALAR)
// Auto-detect the most appropriate backend
#if defined(__x86_64__) || defined(_M_X64)
#define GREX_BACKEND_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define GREX_BACKEND_NEON 1
#else
#define GREX_BACKEND_SCALAR 1
#endif
#endif

#endif // INCLUDE_GREX_BACKEND_DEFS_HPP
