.. cpp:namespace:: grex

.. _operations-mask-arithmetic:

#################
Masked Arithmetic
#################

Element-wise arithmetic operations applied conditionally under a mask.
For lanes where the mask is false, the value from ``a`` is preserved.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

.. _operations-mask-addition:

***************
Masked Addition
***************

.. cpp:function:: Vector<T, N> backend::mask_add(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

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
   - **Earlier**: zeroes lanes of ``b`` where :math:`\neg m_i` and adds the result to ``a``.

   Neon
   ====

   - Computes :cpp:func:`~backend::add` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

.. _operations-mask-subtraction:

******************
Masked Subtraction
******************

.. cpp:function:: Vector<T, N> backend::mask_subtract(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

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

   Neon
   ====

   - Computes :cpp:func:`~backend::subtract` and blends with ``a`` under ``m`` using :cpp:func:`~backend::blend`.

.. _operations-mask-multiplication:

*********************
Masked Multiplication
*********************

.. cpp:function:: Vector<T, N> backend::mask_multiply(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

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

.. cpp:function:: Vector<T, N> backend::mask_divide(Mask<T, N> m, Vector<T, N> a, Vector<T, N> b)

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
