.. cpp:namespace:: grex

##################
Bitwise Operations
##################

Element-wise bitwise operations on vectors.
Sub-native vectors are embedded in a full native vector; each native lane of a super-native vector is processed independently.

.. _operations-bitwise-not:

***********
Bitwise NOT
***********

.. cpp:function:: Vector<T, N> backend::bitwise_not(Vector<T, N> v)

   Element-wise bitwise complement :math:`\neg v` of integer elements.

   - x86-64: implemented as XOR with an all-ones vector.
   - Neon:

     - 8–32-bit: ``vmvnq``.
     - 64-bit: 32-bit ``vmvnq`` followed by reinterpretation.

.. _operations-bitwise-and:

***********
Bitwise AND
***********

.. cpp:function:: Vector<T, N> backend::bitwise_and(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise AND :math:`a_i \land b_i` for integer element types.

   Implemented using ``and``/``vandq`` intrinsics (x86-64/Neon).

.. _operations-bitwise-or:

**********
Bitwise OR
**********

.. cpp:function:: Vector<T, N> backend::bitwise_or(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise OR :math:`a_i \lor b_i` for integer element types.

   Implemented using ``or``/``vorrq`` intrinsics (x86-64/Neon).

.. _operations-bitwise-xor:

***********
Bitwise XOR
***********

.. cpp:function:: Vector<T, N> backend::bitwise_xor(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise XOR :math:`a_i \oplus b_i` for integer element types.

   Implemented using ``xor``/``veorq`` intrinsics (x86-64/Neon).
