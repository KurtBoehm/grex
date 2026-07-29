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

.. cpp:function:: f32 backend::sqrt(Scalar<f32> v)
                  f64 backend::sqrt(Scalar<f64> v)

   Scalar square root :math:`\sqrt{v}`.

   x86-64
   ======

   - 128-bit ``sqrt`` intrinsics on a temporary SIMD vector.

   Neon
   ====

   - Inline assembly (GCC) or built-in (Clang) which emits ``fsqrt``.
