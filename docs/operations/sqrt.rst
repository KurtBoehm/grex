.. cpp:namespace:: grex

###########
Square Root
###########

Element-wise square root on vectors, plus scalar counterparts.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

Only floating-point element types are supported.

.. _operations-sqrt:

***********
Square Root
***********

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Vector<T, N> backend::sqrt(Vector<T, N> v)

   Element-wise square root :math:`\sqrt{v_i}`.

   Shared
   ======

   - **Binary16 without hardware support**: widen to binary32, take the square root there, and round back, which is still correctly rounded (see :ref:`f16-accuracy`).

   x86-64
   ======

   - ``sqrt`` intrinsics.

   Neon
   ====

   - ``vsqrtq`` intrinsics.

.. _operations-sqrt-scalar:

***************
Scalar Variants
***************

.. cpp:function:: template<FloatVectorizable T> \
                  T backend::sqrt(Scalar<T> v)

   Scalar square root :math:`\sqrt{v}`.

   x86-64
   ======

   - Scalar ``sqrt`` intrinsics on a temporary SIMD vector; binary16 without AVX512-FP16 goes through binary32.

   Neon
   ====

   - **GCC**: inline ``fsqrt`` assembly, bypassed for arguments the compiler knows to be constant so that the ``__builtin_sqrt`` family can still fold them.
   - **Clang**: the ``__builtin_sqrt`` family throughout.
   - **Binary16 without the FP16 extension**: neither form is usable, and neither fails loudly — the half-precision ``fsqrt`` assembles to an instruction the target does not have, and ``__builtin_sqrtf16`` turns into a call to ``sqrtf16``, which the target libm need not provide.
     The square root is therefore taken in binary32 and rounded back, as for vectors.
