.. cpp:namespace:: grex

################
Shift Operations
################

Element-wise bitwise left and right shifts of integer vectors by a compile-time constant number of bits.
Sub-native vectors are embedded in a full native vector; each native lane of a super-native vector is processed independently.

.. _operations-shift-left:

**********
Left Shift
**********

.. cpp:function:: Vector<T, N> backend::shift_left(Vector<T, N> v, AnyIndexTag auto offset)

   Element-wise logical left shift :math:`v_i \ll \text{offset}` by a constant number of bits for integer element types only.

   x86-64
   ======

   - **8-bit integers**: emulated via 16-bit left shifts plus masking.
   - **16/32/64-bit integers**: uses ``slli_epi{16,32,64}`` intrinsics.

   Neon
   ====

   - Implemented using ``vshlq_n`` intrinsics.

.. _operations-shift-right:

***********
Right Shift
***********

.. cpp:function:: Vector<T, N> backend::shift_right(Vector<T, N> v, AnyIndexTag auto offset)

   Element-wise right shift :math:`v_i \gg \text{offset}` by a constant number of bits for integer element types only.

   As for primitive types, unsigned integers use logical shift; signed integers use arithmetic shift.

   x86-64
   ======

   - **Unsigned (logical shift)**:

     - **8-bit**: emulated via 16-bit logical shifts plus masking.
     - **16/32/64-bit**: uses ``srli_epi{16,32,64}`` intrinsics.

   - **Signed (arithmetic shift)**:

     - **8-bit**: emulated via a 16-bit logical shift, masking, sign-bit reconstruction, and subtraction.
     - **16/32-bit**: uses ``srai_epi{16,32}`` intrinsics.
     - **64-bit**:

       - **x86-64-v4**: uses ``srai_epi64`` intrinsics.
       - **Otherwise**: emulated via 32-bit arithmetic/logical shifts, shuffles, and unpacking.

   Neon
   ====

   Implemented using ``vshrq_n`` intrinsics; an offset of zero returns the input unchanged.
