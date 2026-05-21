.. cpp:namespace:: grex

############
Masked Blend
############

.. _operations-blend:

*****
Blend
*****

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::blend(Mask<T, N> m, Vector<T, N> v0, Vector<T, N> v1)

   Element-wise masked blend:

   .. math::

      r_i =
      \begin{cases}
        v_{1,i} & m_i = \text{true} \\
        v_{0,i} & m_i = \text{false}
      \end{cases}

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: ``mask_mov`` intrinsics on the appropriate vector type:

     .. math::

        r = (\neg m) \land v_0 \;\lor\; m \land v_1.

   - **x86-64-v2+ (broad masks)**: ``blendv`` intrinsics with the mask interpreted as an integer/float vector as required by the intrinsic.
   - **x86-64-v1 (broad masks)**: synthesized from bitwise operations on the underlying integer vectors:

     .. math::

        r = \operatorname{andnot}(m, v_0) \lor (m \land v_1),

     with reinterprets for floating-point element types.

   Neon
   ====

   - ``vbslq`` intrinsics on the underlying mask/vector type to select bits from ``v1`` where ``m`` is set and from ``v0`` otherwise.

   Shared
   ======

   - **Sub-native**: forwards to native :cpp:func:`~backend::blend` on the backing vector/mask and re-wraps.
   - **Super-native**: applies :cpp:func:`~backend::blend` to lower and upper halves independently and recombines.

.. _operations-blend-zero:

*************
Blend (Zeros)
*************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::blend_zero(Mask<T, N> m, Vector<T, N> v1)

   Element-wise masked zeroing:

   .. math::

      r_i =
      \begin{cases}
        v_{1,i} & m_i = \text{true} \\
        0       & m_i = \text{false}
      \end{cases}

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: ``maskz_mov`` intrinsics, which move elements from ``v1`` where the mask bit is set and zero other lanes.
   - **Earlier (broad masks)**: bitwise AND of the mask with ``v1`` on the underlying integer vectors, with reinterprets for floating-point elements:

     .. math::

        r = m \land v_1.

   Neon
   ====

   - ``vandq`` intrinsics between the integer representation of ``m`` and ``v1`` (with reinterprets for floating-point element types).

   Shared
   ======

   - **Sub-native**: forwards to native :cpp:func:`~backend::blend_zero` on the backing vector/mask and re-wraps.
   - **Super-native**: applies :cpp:func:`~backend::blend_zero` to lower and upper halves independently and recombines.
