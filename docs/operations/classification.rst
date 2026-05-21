.. cpp:namespace:: grex

##############
Classification
##############

Element-wise floating-point classification and finite-value filtering.
Sub-native vectors are processed via their backing native vectors; each native lane of a super-native vector is processed independently.

.. _operations-is-finite:

*****************
Finite-Value Test
*****************

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Mask<T, N> backend::is_finite(Vector<T, N> v)

   Element-wise test for finite floating-point values:

   .. math::

      m_i = \operatorname{isfinite}(v_i).

   x86-64
   ======

   - **x86-64-v4**: ``fpclass_*_mask`` intrinsics with classification mask ``0x99`` (NaN and infinities), complemented with ``knot_mask`` to obtain finiteness.
   - **Earlier**:

     - Compute :math:`|v|` via :cpp:func:`~backend::abs` and reinterpret as unsigned integers.
     - Broadcast the bit pattern of positive infinity as an unsigned integer.
     - Compare :math:`|v| < +\infty` with :cpp:func:`~backend::compare_lt`; finite values compare less.

   Neon
   ====

   - Broadcast the largest finite value ``std::numeric_limits<T>::max()``.
   - Compare :math:`|v| \le \text{max}` via ``vabsq`` and ``vcleq``.

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

   Implemented as :cpp:func:`~backend::blend_zero` with the mask from :cpp:func:`~backend::is_finite`.

.. _operations-make-finite-scalar:

*************************
Finite-Value Scalar Clamp
*************************

.. cpp:function:: template<FloatVectorizable T> \
                  Scalar<T> backend::make_finite(Scalar<T> v)

   Scalar version of :cpp:func:`make_finite() <template\<AnyVector V\> V backend::make_finite(V v)>`:

   - Expands ``v`` to a minimal native vector.
   - Applies :cpp:func:`~backend::make_finite` on that vector.
   - Extracts the lowest lane as the result.
