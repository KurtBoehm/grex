.. cpp:namespace:: grex

#####################
Arithmetic Operations
#####################

Most arithmetic operations are implemented directly with SIMD intrinsics, with special handling for negation and integer multiplication.
Sub-native vectors are embedded in a full native vector; each native lane of a super-native vector is processed independently.

.. _operations-addition:

********
Addition
********

.. cpp:function:: Vector<T, N> backend::add(Vector<T, N> a, Vector<T, N> b)

   Element-wise addition :math:`a_i + b_i`.
   Implemented using ``add``/``vaddq`` intrinsics (x86-64/Neon).

.. _operations-subtraction:

***********
Subtraction
***********

.. cpp:function:: Vector<T, N> backend::subtract(Vector<T, N> a, Vector<T, N> b)

   Element-wise subtraction :math:`a_i - b_i`.
   Implemented using ``sub``/``vsubq`` intrinsics (x86-64/Neon).

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
      * - Floating Point
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

   - **Floating point**: uses the corresponding intrinsic.
   - **8-bit integers**: emulated via 16-bit products, shifting, and blending.
   - **16-bit integers**: uses ``mullo_epi16`` intrinsics.
   - **32-bit integers**:

     - **x86-64-v2 and newer**: uses ``mullo_epi32`` intrinsics.
     - **Otherwise**: emulated via two 32×32→64-bit multiplies, additions, and shuffles.

   - **64-bit integers**:

     - **x86-64-v4**: uses ``mullo_epi64`` intrinsics.
     - **Otherwise**: emulated via three 32×32→64-bit multiplies, shifts, and adds.

   Neon
   ====

   - **Floating point and integers up to 32 bits**: uses the corresponding intrinsic.
   - **64-bit integers**: emulated via a 32-bit multiplication, a 32-bit fused multiply-add, additions, and shuffling.

.. _operations-division:

********
Division
********

.. cpp:function:: Vector<T, N> backend::divide(Vector<T, N> a, Vector<T, N> b)

   Element-wise division :math:`a_i / b_i` for floating-point element types only.

   Integer division is intentionally not provided due to poor performance.
