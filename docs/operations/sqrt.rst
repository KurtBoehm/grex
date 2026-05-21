.. cpp:namespace:: grex

.. _operations-sqrt:

###########
Square Root
###########

.. cpp:function:: Vector<T, N> backend::sqrt(Vector<T, N> v)

   Element-wise square root :math:`\sqrt{v_i}` for floating-point element types only.

   - **x86-64**: ``sqrt`` intrinsics.
   - **Neon**: ``vsqrtq`` intrinsics.

.. cpp:function:: Scalar<f32> backend::sqrt(Scalar<f32> v)

   Scalar square root :math:`\sqrt{v}` for ``f32``.

   - **x86-64**: 128-bit ``sqrt`` intrinsics on a temporary SIMD vector.
   - **Neon**: inline assembly (GCC) or built-in (Clang) which emits ``fsqrt``.

.. cpp:function:: Scalar<f64> backend::sqrt(Scalar<f64> v)

   Scalar square root :math:`\sqrt{v}` for ``f64``.

   - **x86-64**: 128-bit ``sqrt`` intrinsics on a temporary SIMD vector.
   - **Neon**: inline assembly (GCC) or built-in (Clang) which emits ``fsqrt``.
