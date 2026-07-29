.. cpp:namespace:: grex

######
Gather
######

Random-access element loading from scalar arrays into SIMD vectors.
Sub-native index vectors are processed via their backing native vectors, while each native lane of a super-native index vector is processed independently and results are merged.

.. _operations-gather:

******
Gather
******

.. cpp:function:: template<Vectorizable TValue, std::size_t Extent, IntVectorizable TIndex, std::size_t N> \
                  VectorFor<TValue, N> backend::gather(std::span<const TValue, Extent> data, Vector<TIndex, N> idxs)

   Loads elements from ``data`` at positions given by ``idxs``:

   .. math::

      r_i = \mathit{data}[\mathit{idxs}_i]

   for integral index element types only.

   - ``data`` must contain at least :math:`1 + \max_i \mathit{idxs}_i` elements; out-of-range indices yield undefined behaviour.
   - Index values are interpreted as element indices (not bytes).

   x86-64
   ======

   - **x86-64-v3+ (native vectors, 32/64-bit elements)**:

     - ``i32``/``i64`` indices: passed directly to AVX2/AVX-512 ``gather`` intrinsics as signed offsets in elements.
     - ``u32`` indices: two cases are distinguished based on :math:`M = \mathtt{data.size()}`:

       - If :math:`M \le 2^{31}`, indices are simply reinterpreted as ``i32`` and used unchanged; the value range is identical for valid indices.
       - If :math:`M > 2^{31}`, the implementation offsets the base pointer and the indices by :math:`L = 2^{31}`:

         - The base pointer is advanced by :math:`L` elements.
         - Indices are XORed with :math:`L`, which is equivalent to subtracting :math:`L` under two’s complement: indices :math:`\ge L` are mapped to values below :math:`L`, while indices below :math:`L` become negative after reinterpretation to ``i32``.
         - Perform the native gather operation with ``i32`` indices.

     - ``u64`` indices:

       - On current and foreseeable x86-64 implementations, user-space virtual addresses occupy strictly fewer than 63 bits.
         For any valid address, the high (sign) bit of the 64-bit index is therefore zero, so reinterpreting ``u64`` as ``i64`` does not change the effective address.
       - The implementation thus calls the ``i64`` gather intrinsics with ``u64`` indices, ignoring signedness.

     - **8/16-bit indices**: indices are widened to ``i32`` with :cpp:func:`~backend::convert` to re-use the ``i32`` path.

   - **Other cases**: implemented via per-lane scalar loads using :cpp:func:`~backend::extract` and :cpp:func:`~backend::set`.

   Neon
   ====

   - No dedicated hardware gather; implemented purely via per-lane scalar loads using :cpp:func:`~backend::extract` and :cpp:func:`~backend::set`.

.. _operations-mask-gather:

*************
Masked Gather
*************

.. cpp:function:: template<Vectorizable TValue, std::size_t Extent, Vectorizable TIndex, std::size_t N> \
                  VectorFor<TValue, N> backend::mask_gather(std::span<const TValue, Extent> data, MaskFor<TValue, N> m, Vector<TIndex, N> idxs)

   Masked variant of :cpp:func:`~backend::gather`:

   .. math::

      r_i =
      \begin{cases}
        \mathit{data}[\mathit{idxs}_i] & m_i \\
        0                              & \neg m_i
      \end{cases}

   - Masked-off lanes are set to :math:`0` and do not access memory.

   x86-64
   ======

   - **x86-64-v3+ (native vectors, 32/64-bit elements)**:

     - ``i32``/``i64`` indices: passed directly to ``mask_i{32,64}gather`` (broad mask)/``mmask_i{32,64}gather`` (compact mask) intrinsics as signed offsets in elements.
     - ``u32``/``u64`` indices: handled identically to the unmasked :cpp:func:`~backend::gather` rules above.
     - **8/16-bit indices**: index vector and mask are converted to ``i32`` with :cpp:func:`~backend::convert` to re-use the ``i32`` path.

   - **Other cases**: implemented via per-lane conditionals:

     - For each lane, read the mask entry :math:`m_i` and index :math:`\mathit{idxs}_i` with :cpp:func:`~backend::extract`.
     - If :math:`m_i` is true, load :math:`\mathit{data}[\mathit{idxs}_i]`, otherwise use :math:`0`.
     - Reconstruct the result vector with :cpp:func:`~backend::set`.

   Neon
   ====

   - Identical to the x86-64 fallback.
