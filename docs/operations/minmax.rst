.. cpp:namespace:: grex

###############
Minimum/Maximum
###############

Element-wise minimum and maximum on vectors.
Sub-native vectors are processed by applying the given operation to their underlying native vectors, while each native lane of a super-native vector is processed independently.

.. _operations-min:

*******
Minimum
*******

.. cpp:function:: Vector<T, N> backend::min(Vector<T, N> a, Vector<T, N> b)

   Element-wise minimum :math:`\min(a_i, b_i)`.

   x86-64
   ======

   - **Floating point**: ``min`` intrinsics.
   - **Integers**:

     - **8-bit signed/16-bit unsigned (x86-64-v1)**: flip the sign bit to reuse unsigned/signed minimum, then flip back.
     - **32-bit integers (x86-64-v1), 64-bit integers (before x86-64-v4)**: compare via :cpp:func:`~backend::compare_lt` and select with :cpp:func:`~backend::blend`.
     - **Otherwise**: integer ``min`` intrinsics.

   Neon
   ====

   - **Floating point**: ``vminnmq`` intrinsics.
   - **8/16/32-bit integers**: ``vminq`` intrinsics.
   - **64-bit integers**: compare with ``vcltq`` and select with ``vbslq``.

.. _operations-max:

*******
Maximum
*******

.. cpp:function:: Vector<T, N> backend::max(Vector<T, N> a, Vector<T, N> b)

   Element-wise maximum :math:`\max(a_i, b_i)`.

   x86-64
   ======

   - **Floating point**: ``max`` intrinsics.
   - **Integers**:

     - **8-bit signed/16-bit unsigned (x86-64-v1)**: flip the sign bit to reuse unsigned/signed maximum, then flip back.
     - **32-bit integers (x86-64-v1), 64-bit integers (before x86-64-v4)**: compare via :cpp:func:`~backend::compare_lt` and select with :cpp:func:`~backend::blend`.
     - **Otherwise**: integer ``max`` intrinsics.

   Neon
   ====

   - **Floating point**: ``vmaxnmq`` intrinsics.
   - **8/16/32-bit integers**: ``vmaxq`` intrinsics.
   - **64-bit integers**: compare with ``vcgtq`` and select with ``vbslq``.
