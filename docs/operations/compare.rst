.. cpp:namespace:: grex

###########
Comparisons
###########

Element-wise comparisons on vectors, producing a Boolean mask.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently and results are reassembled into a super-native mask.

Binary16 uses the corresponding intrinsic where the hardware provides one and otherwise compares in binary32 (see :ref:`f16-implementation`), narrowing the resulting binary32 mask back to binary16 lane width.
Mask equality is a pure bit operation and hence always shares the ``u16`` implementation.

.. _operations-compare-eq:

******************
Equality (Vectors)
******************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::compare_eq(Vector<T, N> a, Vector<T, N> b)

   Element-wise equality :math:`a_i = b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point, 256-bit (x86-64-v3)**: ``cmp_ps``/``cmp_pd`` with the equality predicate as an immediate, exactly as on x86-64-v4.
     - **Floating point, 128-bit**: ``cmpeq`` intrinsics.
     - **64-bit integers**:

       - **x86-64-v2+**: ``cmpeq_epi64``.
       - **x86-64-v1**: emulated via two 32-bit equality comparisons, shuffles, and AND to ensure both 32-bit halves match.

     - **Other integers**: ``cmpeq`` intrinsics.

   Neon
   ====

   - ``vceqq`` intrinsics.

.. _operations-compare-eq-mask:

****************
Equality (Masks)
****************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::compare_eq(Mask<T, N> a, Mask<T, N> b)

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

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::compare_neq(Vector<T, N> a, Vector<T, N> b)

   Element-wise inequality :math:`a_i \ne b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point, 256-bit (x86-64-v3)**: ``cmp_ps``/``cmp_pd`` with the inequality predicate as an immediate, exactly as on x86-64-v4.
     - **Floating point, 128-bit**: ``cmpneq`` intrinsics.
     - **Integer**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

   Neon
   ====

   - :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_eq`.

.. _operations-compare-lt:

*********
Less Than
*********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::compare_lt(Vector<T, N> a, Vector<T, N> b)

   Element-wise strict less-than :math:`a_i < b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point, 256-bit (x86-64-v3)**: ``cmp_ps``/``cmp_pd`` with the less-than predicate as an immediate, exactly as on x86-64-v4, so the operands stay in order.
     - **Floating point, 128-bit**: ``cmpgt`` intrinsics with operands swapped.
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

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::compare_ge(Vector<T, N> a, Vector<T, N> b)

   Element-wise greater-or-equal :math:`a_i \ge b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``cmp_*_mask`` intrinsics.
   - **Earlier**:

     - **Floating point, 256-bit (x86-64-v3)**: ``cmp_ps``/``cmp_pd`` with the greater-or-equal predicate as an immediate, exactly as on x86-64-v4.
     - **Floating point, 128-bit**: ``cmpge`` intrinsics.
     - **Signed integers**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.
     - **Unsigned integers**:

       - **8/16/32-bit, x86-64-v2+**: equality with unsigned maximum, :math:`a \ge b \iff a = \max\{a, b\}`.
       - **8/16-bit, x86-64-v1**: compares saturated difference with zero, :math:`a \ge b \iff \max\{b - a, 0\} = 0`.
       - **32-bit, x86-64-v1; 64-bit**: :cpp:func:`~backend::logical_not` of :cpp:func:`~backend::compare_lt`.

   Neon
   ====

   - ``vcgeq`` intrinsics.
