.. cpp:namespace:: grex

##################
Bitwise Operations
##################

Element-wise bitwise operations on integer vectors.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

.. _operations-bitwise-not:

***********
Bitwise NOT
***********

.. cpp:function:: template<IntVectorizable T, std::size_t N> \
                  Vector<T, N> backend::bitwise_not(Vector<T, N> v)

   Element-wise bitwise complement :math:`\neg v_i`.

   x86-64
   ======

   - XOR with an all-ones vector.

   Neon
   ====

   - **8/16/32-bit**: ``vmvnq`` intrinsics.
   - **64-bit**: 32-bit ``vmvnq`` with reinterpretation.

.. _operations-bitwise-and:

***********
Bitwise AND
***********

.. cpp:function:: template<IntVectorizable T, std::size_t N> \
                  Vector<T, N> backend::bitwise_and(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise AND :math:`a_i \land b_i`.

   x86-64
   ======

   - ``and`` intrinsics.

   Neon
   ====

   - ``vandq`` intrinsics.

.. _operations-bitwise-or:

**********
Bitwise OR
**********

.. cpp:function:: template<IntVectorizable T, std::size_t N> \
                  Vector<T, N> backend::bitwise_or(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise OR :math:`a_i \lor b_i`.

   x86-64
   ======

   - ``or`` intrinsics.

   Neon
   ====

   - ``vorrq`` intrinsics.

.. _operations-bitwise-xor:

***********
Bitwise XOR
***********

.. cpp:function:: template<IntVectorizable T, std::size_t N> \
                  Vector<T, N> backend::bitwise_xor(Vector<T, N> a, Vector<T, N> b)

   Element-wise bitwise XOR :math:`a_i \oplus b_i`.

   x86-64
   ======

   - ``xor`` intrinsics.

   Neon
   ====

   - ``veorq`` intrinsics.
