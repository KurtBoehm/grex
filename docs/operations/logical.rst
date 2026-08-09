.. cpp:namespace:: grex

##################
Logical Operations
##################

Element-wise logical operations on masks.
Sub-native masks are processed via their backing native masks, while each native lane of a super-native mask is processed independently.
Mask operations only move bits, so binary16 masks share the ``u16`` implementation (see :ref:`f16-implementation`).

.. _operations-logical-not:

***********
Logical NOT
***********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::logical_not(Mask<T, N> m)

   Element-wise logical NOT :math:`\neg m_i`.

   x86-64
   ======

   - **x86-64-v4**: mask negation (bitwise complement of the mask register).
   - **Earlier**: XOR with an all-ones mask.

   Neon
   ====

   - Bitwise NOT (``vmvnq``) on the underlying integer mask.

.. _operations-logical-and:

***********
Logical AND
***********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::logical_and(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical AND :math:`a_i \land b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``kand_mask`` intrinsics.
   - **Earlier**: ``and`` intrinsics on the underlying integer vector.

   Neon
   ====

   - ``vandq`` on the underlying integer mask.

.. _operations-logical-andnot:

**************
Logical ANDNOT
**************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::logical_andnot(Mask<T, N> a, Mask<T, N> b)

   Element-wise :math:`\neg a_i \land b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``kandn_mask`` intrinsics.
   - **Earlier**: ``andn`` intrinsics on the underlying integer vector.

   Neon
   ====

   - ``vbicq(b, a)``.

.. _operations-logical-or:

**********
Logical OR
**********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::logical_or(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical OR :math:`a_i \lor b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``kor_mask`` intrinsics.
   - **Earlier**: ``or`` intrinsics on the underlying integer vector.

   Neon
   ====

   - ``vorrq`` on the underlying integer mask.

.. _operations-logical-xor:

***********
Logical XOR
***********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::logical_xor(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical XOR :math:`a_i \oplus b_i`.

   x86-64
   ======

   - **x86-64-v4**: ``kxor_mask`` intrinsics.
   - **Earlier**: ``xor`` intrinsics on the underlying integer vector.

   Neon
   ====

   - ``veorq`` on the underlying integer mask.
