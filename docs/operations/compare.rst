.. cpp:namespace:: grex

###########
Comparisons
###########

Element-wise comparison operations on vectors, producing Boolean masks per lane.
Sub-native vectors are compared via their embedding in a full native vector; each native lane of a super-native vector is processed independently and the results reassembled into a super-mask.

.. _operations-compare-eq:

******************
Equality (Vectors)
******************

.. cpp:function:: Mask<T, N> backend::compare_eq(Vector<T, N> a, Vector<T, N> b)

   Element-wise equality comparison :math:`a_i = b_i`.

   x86-64
   ======

   - **x86-64-v4**: Uses ``cmp_*_mask`` intrinsics.
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

   Element-wise inequality comparison :math:`(a_i \ne b_i)`.

   x86-64
   ======

   - **x86-64-v4**: Uses ``cmp_*_mask`` intrinsics.
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

   - **x86-64-v4**: Uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating-point**: uses ``cmpgt`` intrinsics with operands swapped.
     - **Signed integers**:

       - **8/16/32-bit**: use ``cmpgt`` intrinsics with operands swapped.
       - **64-bit**:

         - **x86-64-v2** and later: uses ``cmpgt_epi64``.
         - **x86-64-v1**: emulated via two 32-bit comparisons, bit manipulations, and shuffles to determine 64-bit ordering from 32-bit pieces.

     - **Unsigned integers**:

       - **8/16/32-bit starting on x86-64-v2**: Inequality with unsigned maximum, i.e. :math:`a < b \iff a \neq \max\{a, b\}`.
       - **8/16-bit on x86-64-v1**: Saturated subtraction and a non-zero test.
       - **32-bit on x86-64-v1, 64-bit starting on x86-64-v2**: Use sign-bit flipping to reinterpret as signed and then compare.
       - **64-bit on x86-64-v1**: emulated via two sign-bit flips, two 32-bit signed comparisons, shuffles, and logic.

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

   - **x86-64-v4**: Uses ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating-point**: use ``cmpge`` intrinsics.
     - **Signed integers**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.
     - **Unsigned integers**:

       - **8/16/32-bit starting on x86-64-v2**: Equality with unsigned maximum, i.e. :math:`a \ge b \iff a = \max\{a, b\}`.
       - **8/16-bit on x86-64-v1**: Saturated subtraction and comparison with zero.
       - **32-bit on x86-64-v1, 64-bit**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.

   Neon
   ====

   - Uses ``vcgeq`` intrinsics.
