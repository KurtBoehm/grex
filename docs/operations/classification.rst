.. cpp:namespace:: grex

##############
Classification
##############

Element-wise floating-point classification and finite-value filtering.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

.. _operations-is-finite:

*****************
Finite-Value Test
*****************

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Mask<T, N> backend::is_finite(Vector<T, N> v)

   Element-wise test for finite floating-point values:

   .. math::

      m_i = \operatorname{isfinite}(v_i).

   A value is non-finite exactly if all of its exponent bits are set, so masking out sign and mantissa and comparing the remainder against the bit pattern of infinity decides finiteness; both backends build on this where no classification instruction applies.

   x86-64
   ======

   - **x86-64-v4**: ``fpclass_*_mask`` intrinsics with classification mask ``0x99`` (NaN and infinities), complemented with ``knot_mask`` to obtain finiteness; binary16 only has such an instruction with AVX512-FP16.
   - **Earlier**: mask the reinterpreted value with the bit pattern of infinity and compare the result against that same pattern with :cpp:func:`~backend::compare_lt`.
     The masking can only ever produce a subset of the exponent bits, so the operand is a non-negative integer bounded by the pattern and a single signed comparison suffices.
   - **Binary64 on x86-64-v1**, which lacks ``pcmpgtq``: masking clears the lower half of every value, so the upper halves are compared as ``i32`` and the outcome is broadcast down.

   Neon
   ====

   - ``vcagtq``, an absolute-value comparison, against a broadcast infinity: :math:`|v| < \infty`.
   - **Binary16 without the FP16 extension**: test the exponent bits against all-ones directly, as above.

.. _operations-make-finite-vector:

*************************
Finite-Value Vector Clamp
*************************

.. cpp:function:: template<AnyVector V> \
                  V backend::make_finite(V v)

   Replaces non-finite lanes (NaN or infinities) by zero, leaving finite lanes unchanged:

   .. math::

      r_i =
      \begin{cases}
        v_i & \operatorname{isfinite}(v_i) \\
        0   & \text{otherwise}
      \end{cases}

   Shared
   ======

   - Implemented as :cpp:func:`~backend::blend_zero` with the mask from :cpp:func:`~backend::is_finite`, so it inherits the backend-specific paths of both.

.. _operations-make-finite-scalar:

*************************
Finite-Value Scalar Clamp
*************************

.. cpp:function:: template<FloatVectorizable T> \
                  T backend::make_finite(T v)

   Scalar version of :cpp:func:`make_finite() <template\<AnyVector V\> V backend::make_finite(V v)>`.

   Shared
   ======

   - Expands ``v`` to a minimal native vector.
   - Applies :cpp:func:`~backend::make_finite` on that vector.
   - Extracts the lowest lane as the result.
