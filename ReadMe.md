# Grex 🐦: Extremely Generic Header-Only SIMD Library for C++23

**Grex** (Latin “float, pack, swarm”) is a header-only C++23 library without external dependencies developed by Kurt Böhm to implement generic SIMD vectorization in a portable manner:
All of Grex’s functionality is available on x86-64 and ARM64 CPUs and Linux, macOS, and Windows are supported (details under [Platform Support](#platform-support)).

Main features:

- **Generic SIMD vector/mask types** that make it easy to write a function that can be applied using SIMD vectors with an arbitrary (supported) value type and size.
- **User-friendly interface** using operator overloading, template functions, and function overloading to enable intuitive code without run-time cost.
  - This includes **a large assortment of useful operations** that are implemented with close to ideal efficiency (in most cases), among others:
    - **A full complement of conversions** between value types, including ones with differently-sized value types and ones not provided by the instruction set (e.g. `u64` to `f64` on x86-64 without AVX-512).
    - **Shuffle and blend operations whose structure is specified at compile time**, with a fancy mechanism to choose the most efficient implementation for the desired behaviour.
- **Operations that can be applied using SIMD vectorization or using scalar values only**, allowing functions built upon these operations to achieve the same. These are **supported on any platform**, where only the scalar variants are available on platforms without vectorization support.
  - These are implemented using function overloading and, if required, use an additional argument of an empty _tag_ type which encodes the desired behaviour at compile-time.
  - Tag types such as `grex::ScalarTag`, `grex::FullTag<size>`, and `grex::PartTag<size>` cover many common use cases and are supported by the tagged operations.
- **Multiple backends** which provide the underlying low-level platform-specific definitions:
  - **x86-64**: Optimized support for each [Microarchitecture level](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels):
    - x86-64-v1: Baseline x86-64 using 128 bits per register. Very limited instruction set, some operations (e.g. shuffling) are not fully optimized.
    - x86-64-v2: All SSE extensions that are shared between Intel and AMD, still using 128 bits per register.
    - x86-64-v3: Up to AVX2 using up 256 bits per register.
    - x86-64-v4: The AVX-512 subset supported by all AVX-512-enabled Intel CPUs since Skylake and all AXV-512-enabled AMD CPUs (i.e. starting with Zen 4) using 512 bits per register.
  - **ARM64**: Uses ARM64 Advanced SIMD/Neon instructions (including some not present on 32-bit Neon) with 128-bit registers.
  - **Scalar**: Fallback for other platforms, only provides scalar implementations of the generic operations.

## Platform Support

Grex fully supports vectorization on x86-64 and (little-endian) ARM64 processors.
Support for vectorization on 32-bit x86 processors (too old to be relevant in HPC applications), 32-bit ARM (not relevant in HPC), or big-endian ARM64 (not used in any relevant system) is not planned.
On other platforms, the scalar backend can be used, which provides scalar definitions of the generic operations but does not provide vector/mask types.

Grex has been tested with x86-64 and ARM64 on Linux, ARM64 on macOS Tahoe, and x86-64 on Windows 11 (Windows 10 should also work) using both GCC 14+ and Clang 17+.
Support for other operating systems and compilers (including MSVC) is not planned.

**Warning: The code emitted by GCC 14 on Apple Silicon intermittently omits SIMD loads/stores and should be avoided. GCC 15 does not exhibit these issues.**

## Building

Grex uses the Meson build system and includes a very extensive set of tests.
These can be run by executing `meson setup -C <build directory>` followed by `meson test -C <build directory>`.
`Makefile` contains targets for calling `meson setup` with different optimization and debug settings.

Grex additionally provides fully-featured CMake build files, including the tests (which use CTest in the CMake version).

## Dependencies

Grex itself does not have any dependencies!
The dependencies of the tests are handled using Meson subprojects (or `FetchContent` in the CMake version), which make it possible to use the tests without installing any packages.
The Meson subprojects are managed by [Tlaxcaltin](https://github.com/KurtBoehm/tlaxcaltin) and are:

- [Thesauros](https://github.com/KurtBoehm/thesauros): Many basic components, from data structures to threading utilities to static ranges and many other things.
- [`{fmt}`](https://github.com/fmtlib/fmt): String formatting and printing.
- [`pcg-cpp`](https://github.com/imneme/pcg-cpp.git): Efficient and high-quality pseudo-random number generation.
- [`options`](https://github.com/KurtBoehm/tlaxcaltin/blob/main/options/meson.build): Compiler options to enable more warnings and optimization settings.

## Licences

Grex is licensed under the terms of the Mozilla Public Licence 2.0, which is provided in [`License`](License).
The file [`test/rng.hpp`](test/rng.hpp) is based on [https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c](https://github.com/imneme/pcg-c-basic/blob/master/pcg_basic.c), which is licensed under the Apache Licence, Version 2.0, as provided in [`test/pcg-license`](test/pcg-license).
