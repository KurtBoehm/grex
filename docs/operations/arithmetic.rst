.. cpp:namespace:: grex

#####################
Arithmetic Operations
#####################

Element-wise arithmetic operations on vectors.
Sub-native vectors are embedded in a full native vector; each native lane of a super-native vector is processed independently.

.. _operations-addition:

********
Addition
********

.. cpp:function:: Vector<T, N> backend::add(Vector<T, N> a, Vector<T, N> b)

   Element-wise addition :math:`a_i + b_i`.

   - **x86-64/Neon**: uses ``add``/``vaddq`` intrinsics.

.. _operations-subtraction:

***********
Subtraction
***********

.. cpp:function:: Vector<T, N> backend::subtract(Vector<T, N> a, Vector<T, N> b)

   Element-wise subtraction :math:`a_i - b_i`.

   - **x86-64/Neon**: uses ``sub``/``vsubq`` intrinsics.

.. _operations-negation:

********
Negation
********

.. cpp:function:: Vector<T, N> backend::negate(Vector<T, N> v)

   Element-wise arithmetic negation :math:`-v_i`.

   .. list-table::
      :header-rows: 1
      :widths: 1 1 1

      * - Arguments
        - x86-64
        - Neon
      * - Integers
        - :math:`0 - v`
        - ``vnegq``
      * - Floating point
        - Flip sign bit
        - ``vnegq``

.. _operations-multiplication:

**************
Multiplication
**************

.. cpp:function:: Vector<T, N> backend::multiply(Vector<T, N> a, Vector<T, N> b)

   Element-wise multiplication :math:`a_i \cdot b_i`.

   x86-64
   ======

   - **Floating point**: corresponding intrinsic.
   - **8-bit integers**: emulated via two 16-bit products of even/odd indices, then shifting and blending (based on VCL).
   - **16-bit integers**: ``mullo_epi16``.
   - **32-bit integers**:

     - **x86-64-v2+**: ``mullo_epi32``.
     - **Earlier**: emulated via two 32×32→64-bit multiplies, additions, and shuffles (from Clang-generated assembly using GCC vector extensions).

   - **64-bit integers**:

     - **x86-64-v4**: ``mullo_epi64``.
     - **Earlier**: emulated via three 32×32→64-bit multiplies, shifts, and adds (from Clang-generated assembly using GCC vector extensions).

   Neon
   ====

   - **Floating point and integers ≤ 32 bits**: corresponding intrinsic.
   - **64-bit integers**: emulated via a 32-bit multiply, 32-bit fused multiply-add, additions, and shuffling.

.. _operations-division:

********
Division
********

.. cpp:function:: Vector<T, N> backend::divide(Vector<T, N> a, Vector<T, N> b)

   Element-wise division :math:`a_i / b_i` for floating-point element types only.

   Integer division is intentionally not provided due to poor performance.
