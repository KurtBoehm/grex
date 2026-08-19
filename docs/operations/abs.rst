.. cpp:namespace:: grex

##############
Absolute Value
##############

Element-wise absolute value on vectors.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

.. _operations-abs:

**************
Absolute Value
**************

.. cpp:function:: template<SignedVectorizable T, std::size_t N> \
                  Vector<T, N> backend::abs(Vector<T, N> v)

   Element-wise absolute value :math:`|v_i|` for floating-point and signed-integer element types.

   x86-64
   ======

   - **Floating point**:

     - **x86-64-v4**: ``range`` intrinsics with the control that computes :math:`|\min(v, v)|`.
     - **Earlier**: clear the sign bit via bitwise AND with a constant sign-mask vector (based on VCL).
     - **Binary16**: always clear the sign bit, since the compilers translate ``_mm512_abs_ph`` to the same thing.

   - **Signed integers**:

     - **x86-64-v4**: ``abs`` intrinsics for all widths (8/16/32/64-bit).
     - **x86-64-v2/v3**:

       - **64-bit**: compare with zero to form a sign mask, then conditionally negate via XOR and subtract (based on VCL).
       - **8/16/32-bit**: ``abs`` intrinsics.

     - **x86-64-v1**:

       - **64-bit**: sign extraction with 32-bit shifts, XOR, and subtract (based on VCL).
       - **32-bit**: sign extraction with arithmetic shifts, XOR, and subtract (based on VCL).
       - **16-bit**: per-lane :math:`\max(c, -c)` (based on VCL).
       - **8-bit**: per-lane unsigned :math:`\min(c, -c)` exploiting the sign bit (based on VCL).

   Neon
   ====

   - **Floating point and signed integers**: ``vabsq`` intrinsics; binary16 clears the sign bit where ``vabsq_f16`` is unavailable.
