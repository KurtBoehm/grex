#ifndef INCLUDE_GREX_BACKEND_X86_MACROS_MATH_HPP
#define INCLUDE_GREX_BACKEND_X86_MACROS_MATH_HPP

// The only thing we need to decrement is the number of bits in an integer
#define GREX_DECR_2 1
#define GREX_DECR_4 3
#define GREX_DECR_8 7
#define GREX_DECR_16 15
#define GREX_DECR_32 31
#define GREX_DECR_64 63
#define GREX_DECR(X) GREX_DECR_##X

// Multiply the two arguments
// This is used for (and limited to):
// · Multiplication by two to double the size of a vector (where 2 is the second factor)
// · Computing the register size from the value bits and size
// · Computing the number of bits taken up by a given number of elements
#define GREX_MULTIPLY_2_2 4
#define GREX_MULTIPLY_4_2 8
#define GREX_MULTIPLY_8_2 16
#define GREX_MULTIPLY_8_4 32
#define GREX_MULTIPLY_8_8 64
#define GREX_MULTIPLY_8_16 128
#define GREX_MULTIPLY_8_32 256
#define GREX_MULTIPLY_8_64 512
#define GREX_MULTIPLY_16_2 32
#define GREX_MULTIPLY_16_4 64
#define GREX_MULTIPLY_16_8 128
#define GREX_MULTIPLY_16_16 256
#define GREX_MULTIPLY_16_32 512
#define GREX_MULTIPLY_32_2 64
#define GREX_MULTIPLY_32_4 128
#define GREX_MULTIPLY_32_8 256
#define GREX_MULTIPLY_32_16 512
#define GREX_MULTIPLY_64_2 128
#define GREX_MULTIPLY_64_4 256
#define GREX_MULTIPLY_64_8 512
#define GREX_MULTIPLY_64_16 1024
#define GREX_MULTIPLY_128_2 256
#define GREX_MULTIPLY_256_2 512
#define GREX_MULTIPLY(A, B) GREX_MULTIPLY_##A##_##B

// Divide the two arguments in the following cases:
// · Halve various values
// · Divide a number of bits by 8
// · The minimum size for a given number of value bits
#define GREX_DIVIDE_4_2 2
#define GREX_DIVIDE_8_2 4
#define GREX_DIVIDE_8_8 1
#define GREX_DIVIDE_16_2 8
#define GREX_DIVIDE_16_8 2
#define GREX_DIVIDE_32_2 16
#define GREX_DIVIDE_32_8 4
#define GREX_DIVIDE_64_2 32
#define GREX_DIVIDE_64_8 8
#define GREX_DIVIDE_128_8 16
#define GREX_DIVIDE_128_16 8
#define GREX_DIVIDE_128_32 4
#define GREX_DIVIDE_128_64 2
#define GREX_DIVIDE_256_2 128
#define GREX_DIVIDE_256_8 32
#define GREX_DIVIDE_256_16 16
#define GREX_DIVIDE_256_32 8
#define GREX_DIVIDE_256_64 4
#define GREX_DIVIDE_512_2 256
#define GREX_DIVIDE_512_8 64
#define GREX_DIVIDE_512_16 32
#define GREX_DIVIDE_512_32 16
#define GREX_DIVIDE_512_64 8
#define GREX_DIVIDE_I(A, B) GREX_DIVIDE_##A##_##B
#define GREX_DIVIDE(A, B) GREX_DIVIDE_I(A, B)

// Compute the maximum of the two arguments in the following cases:
// · Maximum of sizes up to 16 and 4
// · Maximum of sizes up to 64 and 8
// · Maximum of a number of bits starting at 64 and 128
#define GREX_MAX_2_4 4
#define GREX_MAX_2_8 8
#define GREX_MAX_4_4 4
#define GREX_MAX_4_8 8
#define GREX_MAX_8_4 8
#define GREX_MAX_8_8 8
#define GREX_MAX_8_32 32
#define GREX_MAX_16_4 16
#define GREX_MAX_16_8 16
#define GREX_MAX_16_32 32
#define GREX_MAX_32_8 32
#define GREX_MAX_32_32 32
#define GREX_MAX_64_8 64
#define GREX_MAX_64_32 64
#define GREX_MAX_64_128 128
#define GREX_MAX_128_128 128
#define GREX_MAX_256_128 256
#define GREX_MAX_I(A, B) GREX_MAX_##A##_##B
#define GREX_MAX(A, B) GREX_MAX_I(A, B)

#endif // INCLUDE_GREX_BACKEND_X86_MACROS_MATH_HPP
