.. cpp:namespace:: grex

#####################
Arithmetic Operations
#####################

Most of these operations are implemented directly using intrinsics, with some special cases for negation and multiplication.
Sub-native vectors are extended to full native vectors while all vectors making up a super-native vector are computed independently.

.. _operations-addition:

********
Addition
********

.. cpp:function:: backend::Vector<T, N> backend::add(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise addition :math:`a_i + b_i`.
   Implemented using ``add``/``vaddq`` intrinsics (x86-64/NEON).

.. _operations-subtraction:

***********
Subtraction
***********

.. cpp:function:: backend::Vector<T, N> backend::subtract(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise subtraction :math:`a_i - b_i`.
   Implemented using ``sub``/``vsubq`` intrinsics (x86-64/NEON).

.. _operations-negation:

********
Negation
********

.. cpp:function:: backend::Vector<T, N> backend::negate(backend::Vector<T, N> v)

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

.. cpp:function:: backend::Vector<T, N> backend::multiply(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise multiplication :math:`a_i \cdot b_i`.

   x86-64
   ======

   - Floating point: uses the corresponding intrinsic.
   - 8-bit integers: emulated via 16-bit products, shifting, and blending.
   - 16-bit integers: uses ``mullo_epi16`` intrinsics.
   - 32-bit integers:

     - Starting with x86-64-v2: uses ``mullo_epi32`` intrinsics.
     - Otherwise: emulated via two 32×32→64-bit multiplies, additions, and shuffles.

   - 64-bit integers:

     - On x86-64-v4: uses ``mullo_epi64`` intrinsics.
     - Otherwise: emulated via three 32×32→64-bit multiplies, shifts, and adds.

   NEON
   ====

   - Floating point, integers up to 32 bits: uses the corresponding intrinsic.
   - 64-bit integers: emulated via a 32-bit multiplication, a 32-bit fused multiply-add, additions, and shuffling.

.. _operations-division:

********
Division
********

.. cpp:function:: backend::Vector<T, N> backend::divide(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise division :math:`a_i / b_i` for floating-point element types only.

   Integer division is intentionally not provided due to poor performance.
