.. cpp:namespace:: grex

.. _operations-mask-arithmetic:

#################
Masked Arithmetic
#################

Element-wise arithmetic operations applied conditionally under a mask.
For lanes where the mask is false, the value from ``a`` is preserved, including the sign of a zero.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.
Binary16 uses the masked intrinsics where AVX512-FP16 provides them and otherwise takes the same blend-based fallback as every other type, which inherits the binary32 emulation from the unmasked operation (see :ref:`f16-implementation`).

.. _operations-mask-addition:

***************
Masked Addition
***************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::mask_add(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

   Element-wise masked addition:

   .. math::

      r_i =
      \begin{cases}
        a_i + b_i & m_i \\
        a_i       & \neg m_i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4**: uses masked-add ``mask_add`` intrinsics with ``a`` as pass-through.
   - **Earlier, integers**: zeroes lanes of ``b`` where :math:`\neg m_i` and adds the result to ``a``, which is exact because there is no signed zero.
   - **Earlier, floating point**: computes :cpp:func:`~backend::add` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.
     The zeroing shortcut would be wrong here, since :math:`(-0) + (+0)` is :math:`+0`, so a masked-off lane holding a negative zero would not be preserved.
     This is the one operation that pays for the blend: it costs three instructions rather than one on x86-64-v1, which has no blend instruction.

   Neon
   ====

   - Computes :cpp:func:`~backend::add` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

.. _operations-mask-subtraction:

******************
Masked Subtraction
******************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::mask_subtract(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

   Element-wise masked subtraction:

   .. math::

      r_i =
      \begin{cases}
        a_i - b_i & m_i \\
        a_i       & \neg m_i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4**: uses masked-subtraction ``mask_sub`` intrinsics with ``a`` as pass-through.
   - **Earlier**: zeroes lanes of ``b`` where :math:`\neg m_i` and subtracts the result from ``a``.
     Unlike for floating-point addition, this preserves every masked-off lane, since :math:`a - (+0)` reproduces ``a`` for every value, negative zero included; it is one instruction rather than three on x86-64-v1 and is therefore kept for every element type.
     The sole difference from a blend is that a masked-off lane holding a signalling not-a-number comes back quieted, since it still passes through the subtraction.

   Neon
   ====

   - Computes :cpp:func:`~backend::subtract` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

.. _operations-mask-multiplication:

*********************
Masked Multiplication
*********************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::mask_multiply(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

   Element-wise masked multiplication:

   .. math::

      r_i =
      \begin{cases}
        a_i \cdot b_i & m_i \\
        a_i           & \neg m_i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4 (apart from 8-bit integers)**: uses masked-multiply ``mask_mul``/``mask_mullo`` intrinsics with ``a`` as pass-through.
   - **Earlier/8-bit integers**: computes :cpp:func:`~backend::multiply` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

   Neon
   ====

   - Computes :cpp:func:`~backend::multiply` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

.. _operations-mask-division:

***************
Masked Division
***************

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Vector<T, N> backend::mask_divide(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

   Element-wise masked division for floating-point vectors only:

   .. math::

      r_i =
      \begin{cases}
        a_i / b_i & m_i \\
        a_i       & \neg m_i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4**: uses masked-divide ``mask_div`` intrinsics with ``a`` as pass-through.
   - **Earlier**: computes :cpp:func:`~backend::divide` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

   Neon
   ====

   - Computes :cpp:func:`~backend::divide` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.
