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

   Constructs a mask whose first ``i \le N`` lanes are set and whose remaining lanes are cleared:

   .. math::

      m_j =
      \begin{cases}
        \text{true} & j < i \\
        \text{false} & j \ge i
      \end{cases}

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**:

     - For native masks, builds a bitmask with the lowest ``i`` bits set via shifts and bitwise complement.
       The implementation is lane-width aware and uses the smallest appropriate ``__mmask`` type.

   - **Earlier (broad masks)**:

     - Constructs an index vector via :cpp:func:`~backend::indices`, broadcasts ``i``, and performs a strict less-than comparison via :cpp:func:`~backend::compare_lt` to obtain the prefix mask.
     - 64-bit elements are handled via 32-bit index patterns and comparisons.

   Neon
   ====

   - Constructs an index vector via :cpp:func:`~backend::indices`, broadcasts ``i``, and compares them with :cpp:func:`~backend::compare_lt` to obtain the prefix mask.

   Shared
   ======

   - **Sub-native**: delegates to the corresponding native :cpp:func:`~backend::cutoff_mask` on the backing mask, then wraps the result.
   - **Super-native**:

     - If :math:`i \le N / 2`:

       - Lower half uses :cpp:func:`~backend::cutoff_mask`.
       - Upper half is cleared with :cpp:func:`~backend::zeros`.

     - Otherwise:

       - Lower half is fully set with :cpp:func:`~backend::ones`.
       - Upper half uses :cpp:func:`~backend::cutoff_mask`.

.. _operations-single-mask:

****************
Single-Lane Mask
****************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::single_mask(std::size_t i, TypeTag<Mask<T, N>>)

   Constructs a mask with lane ``i < N`` set and all other lanes cleared.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**:

     - Forms a bitmask with a single bit set at position ``i`` in the appropriate ``__mmask`` type.

   - **Earlier (broad masks)**:

     - Constructs an index vector via :cpp:func:`~backend::indices`, broadcasts ``i``, and compares them via :cpp:func:`~backend::compare_eq` to select only lane ``i``.

   Neon
   ====

   - Implemented analogously to the earlier x86-64 path: uses :cpp:func:`~backend::indices` and :cpp:func:`~backend::compare_eq` on the unsigned mask representation.

   Shared
   ======

   - **Sub-native**: delegates to the corresponding native single-lane mask and wrapping.
   - **Super-native**:

     - If :math:`i < N / 2`: lower half has a single lane set at ``i``, upper half is cleared.
     - Otherwise: lower half is cleared, upper half has a single lane set at ``i - N / 2``.

.. _operations-cutoff:

*******
Cut-Off
*******

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::cutoff(std::size_t i, Vector<T, N> v)

   Zeroes all lanes at indices :math:`j \ge i`, leaving the first ``i`` lanes unchanged:

   .. math::

      r_j =
      \begin{cases}
        v_j & j < i \\
        0   & j \ge i
      \end{cases}

   Only valid for :math:`i \le N`.

   x86-64
   ======

   - **Native vectors**:

     - **x86-64-v4**: uses ``maskz_mov`` intrinsics driven by a compressed cut-off mask.
     - **Earlier**: uses ``and`` intrinsics a broad cut-off mask.

   Neon
   ====

   - **Native vectors**: uses :cpp:func:`~backend::blend_zero` with a cut-off mask produced by :cpp:func:`~backend::cutoff_mask` and a zero vector.

   Shared
   ======

   - **Sub-native**: forwards to the native :cpp:func:`~backend::cutoff` on the backing vector and wraps the result.
   - **Super-native**:

     - If :math:`i \le N / 2`: Lower half is cut off at ``i``, upper half is set to :cpp:func:`~backend::zeros`.
     - Otherwise: Lower half is kept unchanged, upper half is cut off at ``i - N / 2``.
