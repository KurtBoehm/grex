.. cpp:namespace:: grex

##################
Logical Operations
##################

Element-wise logical operations on masks.
Sub-native masks are embedded in a full native mask; each native lane of a super-native mask is processed independently.

.. _operations-logical-not:

***********
Logical NOT
***********

.. cpp:function:: Mask<T, N> backend::logical_not(Mask<T, N> m)

   Element-wise logical NOT :math:`\neg m` of mask lanes.

   - **x86-64**:

     - **x86-64-v4**: uses mask negation (bitwise complement of the mask register).
     - **Otherwise**: implemented via XOR with an all-ones mask.

   - **Neon**: implemented via bitwise NOT (``vmvnq``) on the underlying integer mask type.

.. _operations-logical-and:

***********
Logical AND
***********

.. cpp:function:: Mask<T, N> backend::logical_and(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical AND :math:`a \land b` of mask lanes.

   - **x86-64**:

     - **x86-64-v4**: uses ``kand_mask`` intrinsics.
     - **Otherwise**: uses ``and`` intrinsics on the underlying integer vector.

   - **Neon**: uses ``vandq`` on the underlying integer mask type.

.. _operations-logical-andnot:

**************
Logical ANDNOT
**************

.. cpp:function:: Mask<T, N> backend::logical_andnot(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical :math:`\neg a \land b` of mask lanes.

   - **x86-64**:

     - **x86-64-v4**: uses ``kandn_mask`` intrinsics.
     - **Otherwise**: uses ``andn`` intrinsics on the underlying integer vector.

   - **Neon**: uses ``vbicq(b, a)``.

.. _operations-logical-or:

**********
Logical OR
**********

.. cpp:function:: Mask<T, N> backend::logical_or(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical OR :math:`a \lor b` of mask lanes.

   - **x86-64**:

     - **x86-64-v4**: uses ``kor_mask`` intrinsics.
     - **Otherwise**: uses ``or`` intrinsics on the underlying integer vector.

   - **Neon**: uses ``vorrq`` on the underlying integer mask type.

.. _operations-logical-xor:

***********
Logical XOR
***********

.. cpp:function:: Mask<T, N> backend::logical_xor(Mask<T, N> a, Mask<T, N> b)

   Element-wise logical XOR :math:`a \oplus b` of mask lanes.

   - **x86-64**:

     - **x86-64-v4**: uses ``_kxor_mask*`` intrinsics.
     - **Otherwise**: uses ``xor`` intrinsics on the underlying integer vector.

   - **Neon**: uses ``veorq`` on the underlying integer mask type.
