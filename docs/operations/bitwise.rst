.. cpp:namespace:: grex

##################
Bitwise Operations
##################

Element-wise bitwise operations on vectors.
Sub-native vectors are extended to full native vectors, and all vectors in a super-native vector are processed independently.

.. _operations-bitwise-not:

***********
Bitwise NOT
***********

.. cpp:function:: backend::Vector<T, N> backend::bitwise_not(backend::Vector<T, N> v)

   Element-wise bitwise complement :math:`\neg v` of integer elements.

   - x86-64: implemented as XOR with all-ones.
   - NEON:

     - 8–32-bit: ``vmvnq``.
     - 64-bit: 32-bit ``vmvnq`` and reinterpretation.

.. _operations-bitwise-and:

***********
Bitwise AND
***********

.. cpp:function:: backend::Vector<T, N> backend::bitwise_and(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise bitwise AND :math:`a_i \land b_i` for integer element types.

   Implemented using ``and``/``vandq`` intrinsics (x86-64/NEON).

.. _operations-bitwise-or:

**********
Bitwise OR
**********

.. cpp:function:: backend::Vector<T, N> backend::bitwise_or(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise bitwise OR :math:`a_i \lor b_i` for integer element types.

   Implemented using ``or``/``vorrq`` intrinsics (x86-64/NEON).

.. _operations-bitwise-xor:

***********
Bitwise XOR
***********

.. cpp:function:: backend::Vector<T, N> backend::bitwise_xor(backend::Vector<T, N> a, backend::Vector<T, N> b)

   Element-wise bitwise XOR :math:`a_i \oplus b_i` for integer element types.

   Implemented using ``xor``/``veorq`` intrinsics (x86-64/NEON).
