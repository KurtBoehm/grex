.. cpp:namespace:: grex

##############
Table Shuffles
##############

Element-wise table lookups driven by per-lane indices.

.. _operations-shuffle-indices-x64:

*********************************
Shuffle Index Conversion (x86-64)
*********************************

.. cpp:function:: template<UnsignedIntVectorizable T, std::size_t N, std::size_t D, std::size_t V> \
                  Vector<UnsignedInt<D>, N * V / D> \
                  backend::shuffle_indices(Vector<T, N> idxs, IndexTag<D>, IndexTag<V>)

   Converts the unsigned lane indices in ``idxs`` to indices appropriate for a given shuffle instruction, taking into account the width :math:`V` of the values in the lookup table and the width :math:`D` of the chunks that are shuffled by the instruction the output indices are to be used for.
   If :math:`V = D`, ``idxs`` is simply converted to ``UnsignedInt<D>``; if :math:`V > D`, the indices are mapped as follows, with :math:`F = V / D` as the number of :math:`D`-sized chunks which one :math:`V`-sized value occupies:

   .. math::

      \mathit{result}_i = F \cdot idxs_{\lfloor i / F \rfloor} + i \bmod F

   :math:`V < D` cannot occur, as a shuffle operation operating on :math:`D`-sized chunks cannot shuffle a vector with narrower values.

   - **x86-64-v1**: no implementation, as index conversions are not necessary for the fallback shuffle implementation.
   - **x86-64-v2+**:

     - :math:`V = D`: convert ``idxs`` to ``UnsignedInt<D>`` using :cpp:func:`~backend::convert`.
       This applies to many cases, including all cross-lane shuffle operations in AVX-512.
     - Index elements are wider than :math:`V`: first narrow to :math:`V` bytes using :cpp:func:`~backend::convert`, then go on with one of the other cases.
     - Index elements are at most as wide as :math:`V`, byte-level shuffle (i.e. :math:`F = V`; ``pshufb`` is the only runtime shuffling instruction on x86-64-v2) with larger values:

       - Scale each index by :math:`V` (using a right shift) so that each index selects the starting byte of the corresponding value in the lookup table.
       - Shuffle each scaled index using a constant table so that it is repeated in each byte of the corresponding :math:`V`-sized chunk.
       - Offset each byte using a constant table so that byte :math:`i` within each :math:`V`-sized block is offset by :math:`i`.

   - **x86-64-v3 (AVX2)**:

     - The only cross-lane shuffle acts on 32-bit values, requiring an implementation for tables with 64-bit values starting from four-lane indices with different width:

       - Specialized paths using ``pshufd``/``vpshufd``, ``pshuflw``, ``vpshufb``, and some miscellaneous broadcasting, unpacking, shifting, and addition operations.

   - **x86-64-v4 (AVX-512)**:

     - All new cross-lane permutation instructions use values and indices of the same width, i.e. :math:`V = D`, and do not require additional special cases.

.. _operations-shuffle-indices-neon:

*******************************
Shuffle Index Conversion (Neon)
*******************************

.. cpp:function:: template<UnsignedIntVectorizable T, std::size_t N, std::size_t V> \
                  Vector<u8, N * V> backend::shuffle_indices(Vector<T, N> idxs, IndexTag<V>)

   Converts the unsigned lane indices in ``idxs`` to indices appropriate for the ``tbl`` instruction, which permutes on a byte level, taking into account the width :math:`V` of the values in the lookup table.
   If :math:`V = 1`, ``idxs`` is simply converted to ``u8``; if :math:`V > 1`, the indices are mapped as follows:

   .. math::

      \mathit{result}_i = V \cdot idxs_{\lfloor i / V \rfloor} + i \bmod V

   - Uses a mix of lane zip/unzip, shifts, multiplications, and ``vqtbl1q_u8``/``vqtbl2q_u8`` lookups with small constant tables to expand element indices to byte indices.
   - 64-bit element indices are first compressed to 32-bit via lane unpacking, then treated as 32-bit indices.

.. _operations-shuffle-dynamic:

*************
Table Shuffle
*************

.. cpp:function:: template<AnyVectorizable TTbl, std::size_t NTbl, UnsignedIntVectorizable TIdx, std::size_t NIdx, std::size_t UpperBound, std::size_t Offset> \
                  Vector<TTbl, NIdx> backend::shuffle(Vector<TTbl, NTbl> table, Vector<TIdx, NIdx> idxs, IndexTag<UpperBound> index_ub, IndexTag<Offset> index_offset)

   Element-wise table lookup.

   Each lane :math:`i` of the result reads element :math:`\mathit{idxs}_i` from ``table``, treating indices as element indices relative to :math:`\mathit{Offset}`; out-of-range indices are masked or wrapped according to backend-specific masking, but behaviour is only specified when :math:`0 \le \mathit{idxs}_i - \mathit{Offset} < \mathit{UpperBound}`.

   .. note::

         This overload is used internally to implement large-table shuffles by splitting tables into chunks.
         Higher-level code always calls the simpler two-argument form.

.. cpp:function:: template<AnyVectorizable TTbl, std::size_t NTbl, UnsignedIntVectorizable TIdx, std::size_t NIdx> \
                  Vector<TTbl, NIdx> backend::shuffle(Vector<TTbl, NTbl> table, Vector<TIdx, NIdx> idxs)

   Convenience overload equivalent to:

   .. code-block:: cpp

      return shuffle(table, idxs, index_tag<NTbl>, index_tag<0>);

   Common Behaviour
   ================

   - **Index range smaller than table size**: if the index type cannot address the full table (:math:`\mathtt{NTbl} > 2^{\mathrm{bits}(\mathtt{TIndex})}`), the table is first shrunk to the largest addressable prefix.
   - **Super-native output**: split ``idxs`` into low/high halves, shuffle each half, then merge.
   - **Super-native table**:

     - Split ``table`` into low/high halves with an implied midpoint offset :math:`M`.
     - Build a mask :math:`\mathit{idxs}_i < M` and blend results from lower/upper tables.

   - **Mismatched element/index sizes**:

     - **x86-64**: starting on level 3, there are various shuffle instructions that operate on differently-sized chunks; ``idxs`` is transformed via :cpp:func:`~backend::shuffle_indices` to be appropriate for the respective chunk size.
     - **Neon**: the table is reinterpreted as ``u8``, the indices are transformed into byte indices using :cpp:func:`~backend::shuffle_indices`.

   - **Sub-/super-native combinations**:

     - If both table and indices are sub-native, they are expanded so that at least one becomes native; the result is then shrunk.
     - If only one of table/indices is sub-native, the sub-native operand is expanded to its corresponding native representation and re-wrapped afterwards.

x86-64
======

- **x86-64-v1**: repeated broadcast-and-blend trees:

  - Broadcasts candidate table lanes using ``unpck``/``shuffle`` intrinsics.
  - Derives lane-selection masks from leading index bits via shifts and comparisons.
  - Selects the requested element per lane with :cpp:func:`~backend::blend`.

- **x86-64-v2 (SSSE3)**: uses packed byte shuffles (``pshufb``):

  - The table is reinterpreted as ``u8``, the indices are transformed into byte indices using :cpp:func:`~backend::shuffle_indices`.
  - ``_mm_shuffle_epi8`` permutes bytes; results are then reinterpreted back to the original element type.

- **x86-64-v3 (AVX2)**:

  - **128-bit tables, 32/64-bit values**: ``_mm_permutevar_ps``/``_mm_permutevar_pd`` intrinsics.
  - **256-bit tables, 32/64-bit values**: ``_mm256_permutevar8x32_epi32`` intrinsics; 64-bit indices are transformed using :cpp:func:`~backend::shuffle_indices`.
  - **256-bit tables, 16/8-bit values:** two ``_mm256_shuffle_epi8`` (which permute locally to each 128-bit lane), one with the original table and one with the 128-bit halves swapped, and blending; 16-bit indices are transformed to byte indices using :cpp:func:`~backend::shuffle_indices`.

- **x86-64-v4 (AVX-512)**:

  - Uses native permute intrinsics where available:

    - **Single-table shuffles**: ``permutexvar`` intrinsics; the 128/256-bit versions are used where no earlier instructions exist.
    - **Shuffles across two 512-bit tables (8-bit only with AVX-512VBMI)**: ``_mm512_permutex2var`` intrinsics.

      - **8-bit fallback**: broadcast each 128-bit lane, shuffle using the 128-bit-lane-local ``_mm512_shuffle_epi8``, and blend.

Neon
====

- Uses table-lookup intrinsics on byte vectors:

  - ``vqtbl1q_u8``/``vqtbl2q_u8``/``vqtbl4q_u8`` for 16/32/64-byte tables respectively.
  - Indices are masked to the table size when ``index_ub`` exceeds the native table width, i.e. AND with ``0x0F``/``0x1F``/``0x3F``.

- Non-byte tables are reinterpreted as ``u8`` while non-byte index vectors are converted to byte indices via :cpp:func:`~backend::shuffle_indices`, shuffled with the corresponding byte-table intrinsic, and reinterpreted back to the destination element type.
- Super-native tables or results are implemented by splitting tables/indices into native halves and reassembling the shuffled halves.
