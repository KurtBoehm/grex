.. cpp:namespace:: grex

###########
Comparisons
###########

Element-wise comparison operations on vectors, producing a Boolean mask.
Sub-native vectors are compared by embedding them into a full native vector; each native lane of a super-native vector is processed independently and the results are reassembled into a super-mask.

.. _operations-compare-eq:

******************
Equality (Vectors)
******************

.. cpp:function:: Mask<T, N> backend::compare_eq(Vector<T, N> a, Vector<T, N> b)

   Element-wise equality comparison :math:`a_i = b_i`.

   x86-64
   ======

   - **x86-64-v4**: uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **64-bit integers**:

       - **x86-64-v2 and later**: uses ``cmpeq_epi64``.
       - **x86-64-v1**: emulated via two 32-bit equality comparisons, shuffles, and an AND to ensure both 32-bit halves are equal.

     - **Other integer and floating-point**: uses ``cmpeq`` intrinsics.

   Neon
   ====

   - Uses ``vceqq`` intrinsics.

.. _operations-compare-eq-mask:

****************
Equality (Masks)
****************

.. cpp:function:: Mask<T, N> backend::compare_eq(Mask<T, N> a, Mask<T, N> b)

   Element-wise equality comparison :math:`a_i = b_i` between masks.

   x86-64
   ======

   - **x86-64-v4**: uses ``kxnor_mask`` intrinsics to detect bitwise equality.
   - **Earlier**: compares the underlying 8-bit chunks with ``cmpeq_epi8``.

   Neon
   ====

   - Uses ``vceqq`` intrinsics on the underlying unsigned integer mask representation.

.. _operations-compare-neq:

********************
Inequality (Vectors)
********************

.. cpp:function:: Mask<T, N> backend::compare_neq(Vector<T, N> a, Vector<T, N> b)

   Element-wise inequality comparison :math:`a_i \ne b_i`.

   x86-64
   ======

   - **x86-64-v4**: uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating-point**: uses ``cmpneq`` intrinsics.
     - **Integer**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

   Neon
   ====

   - :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

.. _operations-compare-lt:

*********
Less Than
*********

.. cpp:function:: Mask<T, N> backend::compare_lt(Vector<T, N> a, Vector<T, N> b)

   Element-wise strict less-than comparison :math:`a_i < b_i`.

   x86-64
   ======

   - **x86-64-v4**: uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating-point**: uses ``cmpgt`` intrinsics with operands swapped.
     - **Signed integers**:

       - **8/16/32-bit**: use ``cmpgt`` intrinsics with operands swapped.
       - **64-bit**:

         - **x86-64-v2 and later**: uses ``cmpgt_epi64``.
         - **x86-64-v1**: emulated via two 32-bit comparisons, bit manipulations, and shuffles to reconstruct the 64-bit ordering from 32-bit pieces.

     - **Unsigned integers**:

       - **8/16/32-bit starting on x86-64-v2**: inequality with the unsigned maximum, i.e. :math:`a < b \iff a \ne \max\{a, b\}`.
       - **8/16-bit on x86-64-v1**: compares the saturated difference with zero, i.e. :math:`a < b \iff \max\{b - a, 0\} \ne 0`.
       - **32-bit on x86-64-v1, 64-bit starting on x86-64-v2**: flips sign bits and performs a signed comparison.
       - **64-bit on x86-64-v1**: emulated by flipping both sign bits, performing 32-bit “less than” and “equals” comparisons, shuffling to extend the result to 64 bits, and combining the intermediate results.

   Neon
   ====

   - Uses ``vcltq`` intrinsics.

.. _operations-compare-ge:

****************
Greater or Equal
****************

.. cpp:function:: Mask<T, N> backend::compare_ge(Vector<T, N> a, Vector<T, N> b)

   Element-wise greater-or-equal comparison :math:`a_i \ge b_i`.

   x86-64
   ======

   - **x86-64-v4**: uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating-point**: uses ``cmpge`` intrinsics.
     - **Signed integers**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.
     - **Unsigned integers**:

       - **8/16/32-bit starting on x86-64-v2**: equality with the unsigned maximum, i.e. :math:`a \ge b \iff a = \max\{a, b\}`.
       - **8/16-bit on x86-64-v1**: compares the saturated difference with zero, i.e. :math:`a \ge b \iff \max\{b - a, 0\} = 0`.
       - **32-bit on x86-64-v1, 64-bit**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.

   Neon
   ====

   - Uses ``vcgeq`` intrinsics.
