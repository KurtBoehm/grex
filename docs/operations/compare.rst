.. cpp:namespace:: grex

###########
Comparisons
###########

Element-wise comparisons on vectors, producing a Boolean mask.
Sub-native vectors are processed by applying the given operation to their underlying native vectors, while each native lane of a super-native vector is processed independently and results are reassembled into a super-native mask.

.. _operations-compare-eq:

******************
Equality (Vectors)
******************

.. cpp:function:: Mask<T, N> backend::compare_eq(Vector<T, N> a, Vector<T, N> b)

   Element-wise equality :math:`a_i = b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **64-bit integers**:

       - **x86-64-v2+**: ``cmpeq_epi64``.
       - **x86-64-v1**: emulated via two 32-bit equality comparisons, shuffles, and AND to ensure both 32-bit halves match.

     - **Other integer and floating-point**: ``cmpeq`` intrinsics.

   Neon
   ====

   - ``vceqq`` intrinsics.

.. _operations-compare-eq-mask:

****************
Equality (Masks)
****************

.. cpp:function:: Mask<T, N> backend::compare_eq(Mask<T, N> a, Mask<T, N> b)

   Element-wise mask equality :math:`a_i = b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``kxnor_mask`` on compressed masks.
   - **Earlier**: ``cmpeq_epi8`` on the underlying 8-bit mask representation.

   Neon
   ====

   - ``vceqq`` on the underlying unsigned mask vector.

.. _operations-compare-neq:

********************
Inequality (Vectors)
********************

.. cpp:function:: Mask<T, N> backend::compare_neq(Vector<T, N> a, Vector<T, N> b)

   Element-wise inequality :math:`a_i \ne b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point**: ``cmpneq`` intrinsics.
     - **Integer**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

   Neon
   ====

   - :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

.. _operations-compare-lt:

*********
Less Than
*********

.. cpp:function:: Mask<T, N> backend::compare_lt(Vector<T, N> a, Vector<T, N> b)

   Element-wise strict less-than :math:`a_i < b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point**: ``cmpgt`` intrinsics with operands swapped.
     - **Signed integers**:

       - **8/16/32-bit**: ``cmpgt`` intrinsics with operands swapped.
       - **64-bit**:

         - **x86-64-v2+**: ``cmpgt_epi64``.
         - **x86-64-v1**: emulated via two 32-bit comparisons, bit manipulations, and shuffles to reconstruct 64-bit ordering.

     - **Unsigned integers**:

       - **8/16/32-bit, x86-64-v2+**: inequality with unsigned maximum, :math:`a < b \iff a \ne \max\{a, b\}`.
       - **8/16-bit, x86-64-v1**: compares saturated difference with zero, :math:`a < b \iff \max\{b - a, 0\} \ne 0`.
       - **32-bit, x86-64-v1; 64-bit, x86-64-v2+**: flip sign bits and perform signed comparison.
       - **64-bit, x86-64-v1**: flip sign bits, perform 32-bit :cpp:func:`~backend::compare_lt` and :cpp:func:`~backend::compare_eq`, shuffle to extend to 64-bit, and combine.

   Neon
   ====

   - ``vcltq`` intrinsics.

.. _operations-compare-ge:

****************
Greater or Equal
****************

.. cpp:function:: Mask<T, N> backend::compare_ge(Vector<T, N> a, Vector<T, N> b)

   Element-wise greater-or-equal :math:`a_i \ge b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point**: ``cmpge`` intrinsics.
     - **Signed integers**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.
     - **Unsigned integers**:

       - **8/16/32-bit, x86-64-v2+**: equality with unsigned maximum, :math:`a \ge b \iff a = \max\{a, b\}`.
       - **8/16-bit, x86-64-v1**: compares saturated difference with zero, :math:`a \ge b \iff \max\{b - a, 0\} = 0`.
       - **32-bit, x86-64-v1; 64-bit**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.

   Neon
   ====

   - ``vcgeq`` intrinsics.
