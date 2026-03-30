.. cpp:namespace:: grex

##################
Logical Operations
##################

Element-wise logical operations on masks.
Sub-native masks are extended to full native masks, and all masks in a super-native mask are processed independently.

.. _operations-logical-not:

***********
Logical NOT
***********

.. cpp:function:: backend::Mask<T, N> backend::logical_not(backend::Mask<T, N> m)

   Element-wise logical NOT :math:`\neg m` of mask lanes.

   - x86-64:

     - x86-64-v4: uses mask negation (bitwise complement of the mask register).
     - Otherwise: implemented via XOR with all-ones.

   - NEON: implemented via bitwise NOT (``vmvnq``) on the underlying integer mask type.

.. _operations-logical-and:

***********
Logical AND
***********

.. cpp:function:: backend::Mask<T, N> backend::logical_and(backend::Mask<T, N> a, backend::Mask<T, N> b)

   Element-wise logical AND :math:`a \land b` of mask lanes.

   - x86-64:

     - x86-64-v4: uses ``kand_mask`` intrinsics.
     - Otherwise: uses ``and`` intrinsics on the underlying integer vector.

   - NEON: uses ``vandq`` on the underlying integer mask type.

.. _operations-logical-andnot:

**************
Logical ANDNOT
**************

.. cpp:function:: backend::Mask<T, N> backend::logical_andnot(backend::Mask<T, N> a, backend::Mask<T, N> b)

   Element-wise logical :math:`\neg a \land b` on mask lanes.

   - x86-64:

     - x86-64-v4: uses ``kandn_mask`` intrinsics.
     - Otherwise: uses ``andn`` intrinsics on the underlying integer vector.

   - NEON: uses ``vbicq(b, a)``.

.. _operations-logical-or:

**********
Logical OR
**********

.. cpp:function:: backend::Mask<T, N> backend::logical_or(backend::Mask<T, N> a, backend::Mask<T, N> b)

   Element-wise logical OR :math:`a \lor b` of mask lanes.

   - x86-64:

     - x86-64-v4: uses ``kor_mask`` intrinsics.
     - Otherwise: uses ``or`` intrinsics on the underlying integer vector.

   - NEON: uses ``vorrq`` on the underlying integer mask type.

.. _operations-logical-xor:

***********
Logical XOR
***********

.. cpp:function:: backend::Mask<T, N> backend::logical_xor(backend::Mask<T, N> a, backend::Mask<T, N> b)

   Element-wise logical XOR of mask lanes.

   - x86-64:

     - x86-64-v4: uses ``_kxor_mask*`` intrinsics.
     - Otherwise: uses ``xor`` intrinsics on the underlying integer vector.

   - NEON: uses ``veorq`` on the underlying integer mask type.
