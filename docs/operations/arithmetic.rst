.. cpp:namespace:: grex

#####################
Arithmetic Operations
#####################

Element-wise arithmetic operations on vectors.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

Binary16 follows the general rule described in :ref:`f16-implementation`: where the hardware provides binary16 instructions, they are used exactly like their binary32 counterparts, and otherwise both operands are widened to binary32, the operation is carried out there, and the result is rounded back, which still yields the correctly rounded binary16 result (see :ref:`f16-accuracy`).
Negation is the exception, as it is a pure bit operation.

.. _operations-addition:

********
Addition
********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::add(Vector<T, N> a, Vector<T, N> b)

   Element-wise addition :math:`a_i + b_i`.

   x86-64
   ======

   - ``add`` intrinsics.

   Neon
   ====

   - ``vaddq`` intrinsics.

.. _operations-subtraction:

***********
Subtraction
***********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::subtract(Vector<T, N> a, Vector<T, N> b)

   Element-wise subtraction :math:`a_i - b_i`.

   x86-64
   ======

   - ``sub`` intrinsics.

   Neon
   ====

   - ``vsubq`` intrinsics.

.. _operations-negation:

********
Negation
********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::negate(Vector<T, N> v)

   Element-wise arithmetic negation :math:`-v_i`.

   x86-64
   ======

   - **Integers**: :math:`0 - v`.
   - **Floating point**: flip the sign bit, which is what both compilers emit for the binary16 intrinsic as well, so it is used unconditionally there.

   Neon
   ====

   - **Integers and floating point**: ``vnegq`` intrinsics; binary16 flips the sign bit where ``vnegq_f16`` is unavailable.

.. _operations-multiplication:

**************
Multiplication
**************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::multiply(Vector<T, N> a, Vector<T, N> b)

   Element-wise multiplication :math:`a_i \cdot b_i`.

   x86-64
   ======

   - **Floating point**: corresponding intrinsic.
   - **8-bit integers**: emulated via two 16-bit products of even/odd indices, then shifting and blending (based on VCL).
   - **16-bit integers**: ``mullo_epi16``.
   - **32-bit integers**:

     - **x86-64-v2+**: ``mullo_epi32``.
     - **Earlier**: emulated via two 32×32→64-bit multiplies of the even and the odd elements, shifts, and shuffles (from Clang-generated assembly using GCC vector extensions).

   - **64-bit integers**:

     - **x86-64-v4**: ``mullo_epi64``.
     - **Earlier**: emulated via three 32×32→64-bit multiplies, shifts, and adds (from Clang-generated assembly using GCC vector extensions).

   Neon
   ====

   - **Floating point and integers ≤ 32 bits**: corresponding intrinsic.
   - **64-bit integers**: emulated via a 32-bit multiply for the two cross terms, a pairwise widening addition and a shift to combine them, and a widening 32×32→64-bit multiply-accumulate for the product of the low halves, plus a 32-bit reversal to line the operands up.

.. _operations-division:

********
Division
********

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Vector<T, N> backend::divide(Vector<T, N> a, Vector<T, N> b)

   Element-wise division :math:`a_i / b_i` for floating-point element types only.

   Integer division is intentionally not provided due to poor performance.

   x86-64
   ======

   - ``div`` intrinsics.

   Neon
   ====

   - ``vdivq`` intrinsics.
