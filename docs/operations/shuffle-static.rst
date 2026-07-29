.. cpp:namespace:: grex

######################
Compile-Time Shuffles
######################

Table lookups whose per-lane indices are fixed at compile time rather than supplied in an index vector.
As with :doc:`blend-static`, a constant index pattern lets the backend select an immediate-controlled permute — or fold the operation into a cheaper one — instead of building an index vector at run time.

The run-time, vector-driven counterpart is described in :doc:`shuffle`.

This is an :doc:`expensive operation <../expensive-operations>`: it is implemented by a list of candidate instruction sequences, and the cheapest applicable candidate is selected at instantiation time.
Candidates are listed below in their declaration order, which is also the order in which ties are broken.
Costs are quoted as :math:`(\text{inverse throughput}, \text{latency})`.

Two candidate families are involved: ``Shuffler`` permutes a single vector, and ``PairShuffler`` permutes across two vectors, which is how super-native shuffles are built.

The index enumerators, and the queries the applicability conditions are phrased in, are described under :ref:`operations-shuffle-static-indices`.

.. _operations-shuffle-static:

****************************
Table Shuffle (Compile-Time)
****************************

.. cpp:function:: template<ShuffleIndex... Idxs, AnyVector V> \
                  V backend::shuffle(V vec)

   Element-wise table lookup with compile-time indices, requiring exactly ``V::size`` indices:

   .. math::

      r_i =
      \begin{cases}
        \mathit{vec}_{\mathit{Idxs}_i} & \mathit{Idxs}_i \text{ is an index} \\
        0                              & \mathit{Idxs}_i = \mathtt{zero\_sh} \\
        \text{unspecified}             & \mathit{Idxs}_i = \mathtt{any\_sh}
      \end{cases}

   A recurring theme below is that byte-granular shuffles get zeroing for free — an out-of-range byte index makes the instruction write a zero — whereas immediate-controlled permutes cannot zero at all and must be followed by a :ref:`compile-time blend_zero <operations-blend-zero-static>`, whose cost is added to theirs.

   Shared
   ======

   - ``ShufflerBlendZero`` — cost of the underlying zero blender

     - **Applicable when** every index either is not an index (``zero_sh``/``any_sh``) or equals its own position, i.e. no element actually moves.
     - Delegates to :ref:`compile-time blend_zero <operations-blend-zero-static>` on ``blend_zeros()``, and reports that candidate's cost as its own.
       Together with ``ZeroBlenderNoop`` this makes an identity shuffle free.

   - ``SubShuffler`` and ``SuperShuffler`` are selected by pattern width rather than by cost.

     - ``SubShuffler`` pads the indices to the full backing register with ``sub_extended()`` and runs the native shuffler on ``vec.full``.
     - ``SuperShuffler`` computes each output half with a ``PairShuffler`` over ``lower`` and ``upper``, using ``half_raw(0)``/``half_raw(1)``, and sums both halves' costs.

   - ``PairShufflerSingle`` — cost of the single shuffle it reduces to

     - **Applicable when** ``indices_in_vector(0)`` or ``indices_in_vector(1)`` succeeds, i.e. every real index falls within one of the two sources.
     - Discards the other source and performs an ordinary one-vector shuffle.

   - ``PairShufflerBlend`` — sum of two shuffles and one blend, universal ``PairShuffler`` fallback

     - Shuffles each source separately, using ``indices_in_vector_fallback(k, any_sh)`` so that indices belonging to the other source become ``any_sh`` and place no constraint on that half's decision.
     - Combines the two results with a :ref:`compile-time blend <operations-blend-static>` on ``blend_vectors()``.

   x86-64
   ======

   **128-bit** — ``ShufflerBlendZero``, ``ShufflerShuffle8x16``, ``ShufflerShuffle32x4``, ``ShufflerExtractSet``:

   - ``ShufflerShuffle8x16`` — :math:`(0.5, 4)`, x86-64-v2+

     - **Applicable whenever SSSE3 is available**, for any pattern: ``pshufb`` permutes bytes arbitrarily within 128 bits, and the conversion to byte indices always succeeds.
     - Reinterprets to ``i8x16`` and issues one ``_mm_shuffle_epi8`` with a constant index vector.
       ``zero_sh``/``any_sh`` lanes become byte index :math:`-1`, which ``pshufb`` zeroes, so zeroing costs nothing extra.

   - ``ShufflerShuffle32x4`` — :math:`(0.5 + z, \max(1, z_\text{lat}))` where :math:`z` is the zero-blend cost

     - **Applicable when** ``convert<4>()`` succeeds, i.e. the pattern is a permutation of whole 32-bit lanes.
     - One ``_mm_shuffle_epi32`` with the immediate from ``imm8()``, followed by a compile-time ``blend_zero`` if any lane must be zeroed, since ``pshufd`` cannot produce zeros.
     - Beats ``ShufflerShuffle8x16`` on latency when nothing needs zeroing, and is the only option below x86-64-v2.

   - ``ShufflerExtractSet`` — :math:`(2N, 1)`, universal fallback

     - Extracts each required lane with :cpp:func:`~backend::extract` and rebuilds the vector with :cpp:func:`~backend::set`.
     - Its inverse throughput is deliberately proportional to the lane count, so it only ever wins when nothing else applies.

   **256-bit (x86-64-v3)** — ``ShufflerBlendZero``, ``ShufflerShuffle8x32``, ``ShufflerShuffle32x8``, ``ShufflerPermute64x4``, ``ShufflerShuffle8x32Ext``:

   - ``ShufflerShuffle8x32`` — :math:`(0.5, 4)`

     - **Applicable when** ``is_lane_local()``, i.e. every index stays inside its own 128-bit lane, because ``_mm256_shuffle_epi8`` shuffles each lane independently.
     - One ``_mm256_shuffle_epi8`` with a constant index vector from ``laned_indices()``; zeroing is again free.

   - ``ShufflerShuffle32x8`` — :math:`(0.5 + z, \max(1, z_\text{lat}))`

     - **Applicable when** ``convert<4>()`` succeeds **and** its ``single_lane()`` succeeds, since ``_mm256_shuffle_epi32`` applies one immediate to both 128-bit lanes.
     - ``_mm256_shuffle_epi32`` plus a zero blend where needed.

   - ``ShufflerPermute64x4`` — :math:`(1 + z, \max(4, z_\text{lat}))`

     - **Applicable when** ``convert<8>()`` succeeds, i.e. the pattern is a permutation of 64-bit elements.
     - ``_mm256_permute4x64_epi64`` — the one immediate-controlled instruction here that crosses the 128-bit lane boundary — plus a zero blend where needed.

   - ``ShufflerShuffle8x32Ext`` — :math:`(2, 4)`, universal fallback at this width

     - **Always applicable**, including arbitrary cross-lane byte patterns.
     - Produces a copy of the vector with its two 128-bit halves swapped via ``_mm256_permute4x64_epi64``, shuffles the original with ``intralane_indices()`` and the swapped copy with ``extralane_indices()``, and ORs the two results.
       Each index array carries :math:`-1` wherever the other one is responsible, so the two shuffles contribute disjoint lanes and the OR simply merges them.

   Note that ``ShufflerExtractSet`` does not appear at this width — ``ShufflerShuffle8x32Ext`` covers every remaining pattern more cheaply.

   **512-bit (x86-64-v4)** — ``ShufflerBlendZero``, ``ShufflerShuffle128x4``, ``ShufflerShuffle8x64``, ``ShufflerShuffle32x16``, ``ShufflerPermutex64x8``, and the ``ShufflerPermutexVar`` family:

   - ``ShufflerShuffle128x4`` — :math:`(0.5, 3)`

     - **Applicable when** ``convert<16>()`` succeeds — the pattern permutes whole 128-bit lanes — and the 32-bit projection has no ``subzero``, i.e. zeroing is never needed *within* a lane.
     - ``_mm512_shuffle_i32x4``, or ``_mm512_maskz_shuffle_i32x4`` when whole lanes must be zeroed, so zeroing stays free.

   - ``ShufflerShuffle8x64`` — :math:`(0.5, 4)`; **applicable when** ``is_lane_local()``; one ``_mm512_shuffle_epi8`` with a constant index vector.
   - ``ShufflerShuffle32x16`` — :math:`(0.5 + z, \max(1, z_\text{lat}))`; **applicable when** ``convert<4>()`` and its ``single_lane()`` succeed; ``_mm512_shuffle_epi32`` plus a zero blend.
   - ``ShufflerPermutex64x8`` — :math:`(1 + z, \max(4, z_\text{lat}))`; **applicable when** ``convert<8>()`` succeeds and its ``double_lane()`` succeeds, since ``_mm512_permutex_epi64`` reuses its immediate across each 256-bit half; plus a zero blend.
   - ``ShufflerPermutexVar64x8``, ``ShufflerPermutexVar32x16``, ``ShufflerPermutexVar16x32``, ``ShufflerPermutexVar8x64`` — :math:`(1, 5)`, :math:`(1, 5)`, :math:`(1, 7)`, :math:`(1, 7)`

     - **Applicable when** ``convert<…>()`` to the corresponding element width succeeds and the result has no ``subzero``.
     - A single fully general cross-lane ``_mm512_permutexvar_epi*`` with a constant index vector, or ``_mm512_permutex2var_epi*`` against a zeroed second operand when lanes must be zeroed.
     - The 8-bit variant requires AVX-512VBMI.

   - Without AVX-512VBMI, a differently built ``ShufflerPermutexVar8x64`` — :math:`(4, 7)` — takes its place as the universal fallback: two cross-lane ``_mm512_permutexvar_epi16`` passes gather the even and odd bytes of each target position, two lane-local ``_mm512_shuffle_epi8`` passes select within them, and the results are ORed.

   The matching ``PairShuffler`` list is ``PairShufflerSingle``, the ``PairShufflerPermutexVar`` counterparts — which use ``_mm512_permutex2var_epi*`` (or its ``maskz`` form) to read from both sources in one instruction — and ``PairShufflerBlend``.

   Neon
   ====

   - ``ShufflerBlendZero`` — as above.
   - ``ShufflerTbl`` — :math:`(1, 8)`

     - **Always applicable**, for any pattern: ``vqtbl1q_u8`` permutes bytes arbitrarily across the whole 128-bit register.
     - Reinterprets the vector as ``u8``, converts the element indices to unsigned byte indices, and issues one ``vqtbl1q_u8``.
       ``zero_sh``/``any_sh`` lanes become out-of-range indices, which ``tbl`` defines to produce zero, so zeroing is free.

   - ``ShufflerExtractSet`` — :math:`(1, 8)`, universal fallback

     - Same extract-and-rebuild strategy as on x86-64, but quoted at a flat cost here rather than one proportional to the lane count.
     - It carries the same cost as ``ShufflerTbl``, so ``ShufflerTbl`` wins every tie purely by coming first in the list.

.. _operations-shuffle-static-indices:

*******
Indices
*******

.. note::

   Shuffle indices are a backend-internal representation.
   Code outside the backend passes the enumerators directly as template arguments to ``grex::shuffle`` and never constructs or inspects the wrapper type described here.

Each lane is described by one ``ShuffleIndex``, so a shuffle over an :math:`N`-lane vector takes :math:`N` of them:

- A plain index, written with the ``_sh`` literal (``0_sh``, ``1_sh``, …), selects that lane of the source.
- ``zero_sh`` forces the lane to zero.
- ``any_sh`` marks the lane as never read, leaving its value unspecified.

``any_sh`` plays the same role as ``any_bl`` for blends: it widens the set of applicable instructions, since an unconstrained lane never contradicts a candidate's granularity.

Internally, the indices are wrapped in ``ShuffleIndices<ValueBytes, N>``, which also carries a ``subzero`` flag recording that zeroing is required at a finer granularity than the pattern is being expressed at — several candidates reject a pattern precisely because ``subzero`` is set.
It exposes the queries the candidates test in their ``is_applicable``:

- ``imm8()`` packs four indices into the immediate of a ``shufps``-style instruction.
- ``convert<DstValueBytes>()`` re-expresses the pattern for a different element width; widening expands each index into consecutive finer indices, narrowing fails when the merged lanes are not contiguous and aligned.
- ``is_lane_local()``/``laned_indices()`` report whether every index stays inside its own 128-bit lane, which is what the lane-local ``pshufb``-style instructions require.
- ``intralane_indices()``/``extralane_indices()`` split a pattern into its in-lane and cross-lane parts, each padded with :math:`-1` where the other applies, so the two can be shuffled separately and merged.
- ``repeated<Segment>()`` checks that the pattern stays within, and agrees across, every ``Segment``-lane block; ``single_lane()`` and ``double_lane()`` are its instantiations for one and two 128-bit lanes, used by instructions that reuse one immediate across lanes.
- ``blend_zeros()`` projects the pattern onto :ref:`blend-zero selectors <operations-blend-selectors>`, both for the case where no lane moves and for the trailing zeroing step of the immediate-controlled permutes.
- ``indices_in_vector(k)`` asks whether all indices of a two-vector shuffle fall within vector ``k``; ``indices_in_vector_fallback(k, any_sh)`` instead replaces the foreign ones with ``any_sh``.
- ``half_raw(k)`` projects onto the lower/upper half of a super-native result.
- ``requires_zeroing()`` reports whether any lane must end up zero, which selects between the plain and ``maskz`` forms of several AVX-512 permutes.
- ``vector()``/``mask()`` materialize the constant index vector and the zeroing mask that the applied instruction consumes.
