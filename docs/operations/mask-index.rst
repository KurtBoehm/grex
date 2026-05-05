.. cpp:namespace:: grex

#################
Index-Based Masks
#################

Index-based mask operations construct masks and masked vectors from lane indices.

.. _operations-cutoff-mask:

***********
Cutoff Mask
***********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::cutoff_mask(std::size_t i, TypeTag<Mask<T, N>>)

   Constructs a mask whose first :math:`i \le N` lanes are set and whose remaining lanes are cleared:

   .. math::

      m_j =
      \begin{cases}
        \text{true} & j < i \\
        \text{false} & j \ge i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**:

     - For native masks, builds a bitmask with the lowest :math:`i` bits set via shifts and complement, using the smallest appropriate ``__mmask`` type.

   - **Earlier (broad masks)**:

     - Constructs an index vector via :cpp:func:`~backend::indices`, broadcasts ``i``, and performs :cpp:func:`~backend::compare_lt` to obtain the prefix mask.
     - 64-bit elements use 32-bit index patterns and comparisons.

   Neon
   ====

   - As on earlier x86-64: :cpp:func:`~backend::indices` with :cpp:func:`~backend::compare_lt`.

   Shared
   ======

   - **Sub-native**: delegate to native :cpp:func:`~backend::cutoff_mask` on the backing mask and wrap.
   - **Super-native**:

     - If :math:`i \le N / 2`:

       - Lower half: :cpp:func:`~backend::cutoff_mask`.
       - Upper half: :cpp:func:`~backend::zeros`.

     - Otherwise:

       - Lower half: :cpp:func:`~backend::ones`.
       - Upper half: :cpp:func:`~backend::cutoff_mask` with :math:`i - N / 2`.

.. _operations-single-mask:

****************
Single-Lane Mask
****************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::single_mask(std::size_t i, TypeTag<Mask<T, N>>)

   Constructs a mask with lane :math:`i < N` set and all other lanes cleared.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: bitfield with a single bit set at position ``i`` in the appropriate ``__mmask``.
   - **Earlier (broad masks)**: :cpp:func:`~backend::indices` and :cpp:func:`~backend::compare_eq` with broadcast ``i``.

   Neon
   ====

   - Same pattern as earlier x86-64: :cpp:func:`~backend::indices` and :cpp:func:`~backend::compare_eq` on the unsigned mask representation.

   Shared
   ======

   - **Sub-native**: delegate to the native single-lane mask and wrap.
   - **Super-native**:

     - If :math:`i < N / 2`: lower half has a single lane set at ``i``, upper half is cleared.
     - Otherwise: lower half cleared, upper half has a single lane set at :math:`i - N / 2`.

.. _operations-cutoff:

*******
Cut-Off
*******

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::cutoff(std::size_t i, Vector<T, N> v)

   Zeroes all lanes at indices :math:`j \ge i`, leaving the first :math:`i` lanes unchanged:

   .. math::

      r_j =
      \begin{cases}
        v_j & j < i \\
        0   & j \ge i
      \end{cases}

   Valid for :math:`i \le N` only.

   x86-64
   ======

   - **Native**:

     - **x86-64-v4**: ``maskz_mov`` intrinsics with a compressed cut-off mask.
     - **Earlier**: ``and`` intrinsics with a broad cut-off mask.

   Neon
   ====

   - **Native**: :cpp:func:`~backend::blend_zero` with a cut-off mask from :cpp:func:`~backend::cutoff_mask` and a zero vector.

   Shared
   ======

   - **Sub-native**: forward to native :cpp:func:`~backend::cutoff` on the backing vector and wrap.
   - **Super-native**:

     - If :math:`i \le N / 2`: lower half cut off at :math:`i`, upper half set to :cpp:func:`~backend::zeros`.
     - Otherwise: lower half unchanged, upper half cut off at :math:`i - N / 2`.
