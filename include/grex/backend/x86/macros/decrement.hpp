#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_DECREMENT_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_DECREMENT_HPP

// The only thing we need to decrement is the number of bits in an integer
#define GREX_DECR_8 7
#define GREX_DECR_16 15
#define GREX_DECR_32 31
#define GREX_DECR_64 63
#define GREX_DECR(X) GREX_DECR_##X

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_DECREMENT_HPP
