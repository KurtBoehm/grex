.. cpp:namespace:: grex

####################
Compile-Time Blends
####################

Blends whose per-lane selection is fixed at compile time rather than supplied as a run-time mask.
Because the selection is a constant, the backend can pick an immediate-controlled instruction — or elide the operation entirely — instead of materializing a mask and issuing a variable blend.

The run-time, mask-driven counterparts are described in :doc:`blend`.

Both operations are :doc:`expensive operations <../expensive-operations>`: rather than a fixed decision tree, each is implemented by a list of candidate instruction sequences, and the cheapest applicable candidate is selected at instantiation time.
Candidates are listed below in their declaration order, which is also the order in which ties are broken.
Costs are quoted as :math:`(\text{inverse throughput}, \text{latency})`.

The selector enumerators, and the queries the applicability conditions are phrased in, are described under :ref:`operations-blend-selectors`.

.. _operations-blend-static:

***********************
Blend (Compile-Time)
***********************

.. cpp:function:: template<BlendSelector... Bls, AnyVector V> \
                  V backend::blend(V a, V b)

   Element-wise blend of ``a`` and ``b`` under compile-time selectors, requiring exactly ``V::size`` selectors:

   .. math::

      r_i =
      \begin{cases}
        a_i                & \mathit{Bls}_i = \mathtt{lhs\_bl} \\
        b_i                & \mathit{Bls}_i = \mathtt{rhs\_bl} \\
        \text{unspecified} & \mathit{Bls}_i = \mathtt{any\_bl}
      \end{cases}

   Shared
   ======

   - ``BlenderConstant`` — :math:`(0, 0)`

     - **Applicable when** ``constant()`` succeeds, i.e. no lane names ``lhs_bl`` and another names ``rhs_bl``.
     - Returns ``a`` or ``b`` unchanged, emitting no instruction at all.
       This is why it heads every candidate list: at zero cost it always wins when it applies.

   - ``SubBlender`` and ``SuperBlender`` are selected by pattern width rather than by cost.

     - ``SubBlender`` pads the selectors to the full backing register with ``sub_extended()`` — the padding lanes become ``any_bl``, so they never constrain the native decision — and runs the native blender on ``a.full``/``b.full``.
     - ``SuperBlender`` blends ``lower()`` and ``upper()`` with independent decisions and recombines; it reports the sum of both halves' costs, so an enclosing candidate sees the true cost of the whole expansion.

   x86-64
   ======

   **128-bit** — ``BlenderConstant``, ``BlenderBlend32x4``, ``BlenderBlend16x8``, ``BlenderVariable``:

   - ``BlenderBlend32x4`` — :math:`(0.5, 1)`, x86-64-v2+

     - **Applicable when** ``convert<4>()`` succeeds, i.e. the selectors agree within each 4-byte group, so the pattern can be expressed over four 32-bit lanes.
     - Reinterprets both inputs as ``f32x4`` and issues one ``_mm_blend_ps`` with the immediate from ``imm8()``.

   - ``BlenderBlend16x8`` — :math:`(0.5, 1)`, x86-64-v2+

     - **Applicable when** ``convert<2>()`` succeeds, i.e. the selectors agree within each 2-byte group.
     - Reinterprets both inputs as ``i16x8`` and issues one ``_mm_blend_epi16``.
     - A pattern uniform over 4-byte groups is also uniform over 2-byte groups, so this candidate is applicable whenever ``BlenderBlend32x4`` is, at the same cost; the tie goes to ``BlenderBlend32x4`` because it is listed first.

   - ``BlenderVariable`` — :math:`(0.5, 4)`, universal fallback

     - Builds a ``NativeMask`` from the selectors with :cpp:func:`~backend::set` and calls the run-time :cpp:func:`~backend::blend`.
     - The latency of 4 accounts for materializing the mask; on x86-64-v1 this is the only non-constant path, since both immediate blends need SSE4.1.

   **256-bit (x86-64-v3)** — ``BlenderConstant``, ``BlenderBlend32x8``, ``BlenderBlend16x16``, ``BlenderVariable``:

   - ``BlenderBlend32x8`` — :math:`(0.5, 1)`

     - **Applicable when** ``convert<4>()`` succeeds.
     - One ``_mm256_blend_ps``; its 8-bit immediate covers all eight 32-bit lanes, so no extra restriction applies.

   - ``BlenderBlend16x16`` — :math:`(0.5, 1)`

     - **Applicable when** ``convert<2>()`` succeeds **and** the result's ``single_lane()`` succeeds.
     - The second condition exists because ``_mm256_blend_epi16`` reuses its 8-bit immediate for both 128-bit lanes, so the pattern must repeat identically across them.

   **512-bit (x86-64-v4)** — ``BlenderConstant`` and ``BlenderVariable`` only.
   AVX-512 expresses blending through mask registers anyway, so ``BlenderVariable`` already produces a single masked move.

   Neon
   ====

   - ``BlenderConstant`` — as above.
   - ``BlenderVariable`` — :math:`(1, 8)`, universal fallback

     - **Always applicable.** Neon has no immediate-controlled blend, so every non-constant pattern goes through it.
     - Materializes the selector pattern as a ``static constexpr`` array, loads it, converts it to a mask with ``vector2mask``, and calls the run-time :cpp:func:`~backend::blend` (a ``vbslq``).
     - The cost is markedly higher than the x86-64 equivalent because the mask comes from a constant-pool load rather than from immediate operands.

.. _operations-blend-zero-static:

****************************
Blend Zeros (Compile-Time)
****************************

.. cpp:function:: template<BlendZeroSelector... Bzs, AnyVector V> \
                  V backend::blend_zero(V v)

   Element-wise zeroing of ``v`` under compile-time selectors, requiring exactly ``V::size`` selectors:

   .. math::

      r_i =
      \begin{cases}
        v_i                & \mathit{Bzs}_i = \mathtt{keep\_bz} \\
        0                  & \mathit{Bzs}_i = \mathtt{zero\_bz} \\
        \text{unspecified} & \mathit{Bzs}_i = \mathtt{any\_bz}
      \end{cases}

   Shared
   ======

   - ``ZeroBlenderNoop`` — :math:`(0, 0)`

     - **Applicable when** every selector is ``keep_bz`` or ``any_bz``, i.e. no lane is actually required to be zero.
     - Returns ``v`` unchanged, emitting nothing.

   - ``ZeroBlenderZero`` — :math:`(0, 1)`

     - **Applicable when** every selector is ``zero_bz`` or ``any_bz``, i.e. no lane is actually required to be kept.
     - Returns :cpp:func:`~backend::zeros`, discarding ``v`` entirely.

   - ``SubZeroBlender`` and ``SuperZeroBlender`` are selected by pattern width, analogously to ``SubBlender``/``SuperBlender`` above.

   x86-64
   ======

   **128-bit** — ``ZeroBlenderNoop``, ``ZeroBlenderZero``, ``ZeroBlenderMovq``, ``ZeroBlenderBlend32x4``, ``ZeroBlenderBlend16x8``, ``ZeroBlenderAnd``:

   - ``ZeroBlenderMovq`` — :math:`(0.5, 1)`

     - **Applicable when** ``convert<8>()`` succeeds and the resulting two-lane pattern keeps the low 64 bits and zeros the high 64 bits.
     - One ``_mm_move_epi64``.
     - Any pattern coarse enough for this is also expressible over 32-bit lanes, so from x86-64-v2 onwards ``ZeroBlenderBlend32x4`` is applicable too and wins on throughput; ``ZeroBlenderMovq`` is what keeps this common case cheap on x86-64-v1.

   - ``ZeroBlenderBlend32x4`` — :math:`(0.25, 2)`, x86-64-v2+

     - **Applicable when** ``convert<4>()`` succeeds.
     - ``_mm_blend_ps`` against a zeroed register, with the immediate from ``imm8()``.

   - ``ZeroBlenderBlend16x8`` — :math:`(0.25, 2)`, x86-64-v2+

     - **Applicable when** ``convert<2>()`` succeeds.
     - ``_mm_blend_epi16`` against a zeroed register.

   - ``ZeroBlenderAnd`` — :math:`(0.5, 4)`, universal fallback

     - Builds an all-ones/all-zeros integer mask from the selectors with :cpp:func:`~backend::set` and applies :cpp:func:`~backend::bitwise_and`.
     - On x86-64-v1 this is the only optimized path beyond the two trivial candidates.

   **256-bit (x86-64-v3)** — ``ZeroBlenderNoop``, ``ZeroBlenderZero``, ``ZeroBlenderBlend32x8``, ``ZeroBlenderBlend16x16``, ``ZeroBlenderAnd``:

   - ``ZeroBlenderBlend32x8`` — :math:`(0.5, 2)`; **applicable when** ``convert<4>()`` succeeds; ``_mm256_blend_ps`` against zero.
   - ``ZeroBlenderBlend16x16`` — :math:`(0.5, 2)`; **applicable when** ``convert<2>()`` succeeds **and** its ``single_lane()`` succeeds, for the same per-128-bit-lane immediate reason as ``BlenderBlend16x16``; ``_mm256_blend_epi16`` against zero.

   **512-bit (x86-64-v4)** — ``ZeroBlenderNoop``, ``ZeroBlenderZero``, and ``ZeroBlenderAnd``.

   Neon
   ====

   - ``ZeroBlenderNoop`` and ``ZeroBlenderZero`` — as above.
   - ``ZeroBlenderAnd`` — :math:`(1, 8)`, universal fallback

     - **Always applicable.** Loads a constant all-ones/all-zeros mask built from the selectors and applies ``vandq``.
     - As with ``BlenderVariable``, the cost reflects a constant-pool load rather than an immediate.

.. _operations-blend-selectors:

*********
Selectors
*********

.. note::

   Selectors are a backend-internal representation.
   Code outside the backend passes the enumerators directly as template arguments to ``grex::blend``/``grex::blend_zero`` and never constructs or inspects the wrapper types described here.

Each lane is described by one enumerator, so a blend over an :math:`N`-lane vector takes :math:`N` selectors:

.. list-table::
   :header-rows: 1

   * - Type
     - Enumerators
     - Meaning
   * - ``BlendSelector``
     - ``lhs_bl``, ``rhs_bl``, ``any_bl``
     - Take the lane from the first vector, from the second vector, or leave it unspecified.
   * - ``BlendZeroSelector``
     - ``keep_bz``, ``zero_bz``, ``any_bz``
     - Keep the lane, replace it with zero, or leave it unspecified.

``any_bl``/``any_bz`` mark lanes whose value is irrelevant, potentially allowing a more efficient operation to be used than would be possible with some arbitrary index.

Internally, the selector arrays are wrapped in ``BlendSelectors<ValueBytes, N>`` and ``BlendZeroSelectors<ValueBytes, N>``, which provide the queries the candidates test in their ``is_applicable``:

- ``imm8()`` packs the selectors into the immediate operand of a ``blend`` instruction (for at most 8 lanes).
- ``convert<DstValueBytes>()`` re-expresses the pattern for a different element width.
  Widening splits each selector across the finer lanes and always succeeds; narrowing merges lanes and fails (returns an empty ``std::optional``) when the merged lanes disagree, ignoring ``any``.
  Most applicability conditions above are a ``convert`` to the element width an instruction operates on.
- ``constant()`` reports whether every lane selects the same input, ignoring ``any``.
- ``lower()``/``upper()`` project onto the halves of a super-native vector.
- ``single_lane()`` checks whether the pattern repeats identically in every 128-bit lane, which is what instructions reusing one immediate across lanes require.
- ``sub_extended()`` pads a sub-native pattern with ``any`` up to the full backing register.
