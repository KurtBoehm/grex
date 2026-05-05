.. cpp:namespace:: grex

#########################
Multibyte Integer Loading
#########################

Loading of packed integers whose logical value spans :math:`M` bytes into SIMD vectors with element width :math:`N = 2^B` bytes, where :math:`N = \text{bitceil}(M)`.
The input memory is padded on both sides by at least one full SIMD register.

Sub-native vectors are embedded in a full native vector; super-native vectors are assembled from native halves.

.. _operations-load-multibyte:

.. cpp:function:: template<std::size_t SrcBytes, AnyVector Dst> \
                  Dst backend::load_multibyte(const u8* ptr, IndexTag<SrcBytes>, TypeTag<Dst>)

   Load ``Dst::size`` integers, each represented by ``SrcBytes`` consecutive bytes, into a SIMD vector with element type ``Dst::Value``.

   - :math:`M = \mathtt{SrcBytes}`: source bytes per logical integer.
   - :math:`N = \mathtt{sizeof(Dst::Value)} \ge M`: destination byte width, a power of two.
   - :math:`O = N - M`: padding bytes.
   - :math:`S = \mathtt{Dst::size}`: number of elements.

   Lane :math:`i` receives the zero-extended little-endian value of bytes :math:`[i \cdot M, (i + 1) \cdot M)` in the low :math:`M` bytes of the lane; the high :math:`O` bytes are zero.

   Diagrams use the index of the logical value in each byte; ``.`` represents an unspecified byte, ``·`` represents a zero byte, ``|`` is for readability only.

   ******
   x86-64
   ******

   - :math:`M = N`: direct :cpp:func:`~backend::load`.
   - **x86-64-v1**:

     - :math:`N = 8`, :math:`M < 8`, 128-bit output: offset load, shift, merge, shift-back; for :math:`M = 5`:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load 16 bytes at offset :math:`O = 8 - M = 3`
            - ``...00000|11111...``
          * - 64-bit left shift by :math:`O` bytes
            - ``···..000|···11111``
          * - Merge so both values are top-aligned
            - ``...00000|···11111``
          * - 64-bit right shift by :math:`O` bytes
            - ``00000···|11111···``

     - :math:`N = 4`, :math:`M = 3`, 128-bit output: offset load, then shifts/merges:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load with offset 2 so each 3-byte value is in its 32-bit lane
            - ``..00|0111|2223|33..``
          * - 64-bit left shifts to top-align each value
            - | ``···.|.000|···1|1122``
              | ``····|·..0|····|·222``
              | ``··..|0001|··22|2333``
          * - Merge halves to top-align per 32-bit lane
            - | ``···.|..00|.000|0111``
              | ``····|··22|·222|2333``
          * - Merge all values top-aligned
            - ``.000|·111|·222|2333``
          * - 32-bit right shift by 1 byte
            - ``000·|111·|222·|333·``

     - :math:`N = 4`, :math:`M = 3`, 64-bit output: simplified variant:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load with offset 1
            - ``.000|111.|....|....``
          * - 32-bit left shift by 1 byte
            - ``·.00|·111|·...|·...``
          * - Merge both values top-aligned
            - ``.000|·111|·...|·...``
          * - 32-bit right shift by 1 byte
            - ``000·|111·|...·|...·``

   - **x86-64-v2+**:

     - 128-bit baseline: load packed bytes without offset and shuffle via ``_mm_shuffle_epi8`` with a compile-time table.
     - :math:`N = 8`, :math:`M = 6`, 128-bit: load with offset 2, ``_mm_shufflelo_epi16`` to reorder the low half, then ``_mm_blend_epi16`` to clear unused high bytes per 64-bit lane.

   - **x86-64-v3+ (native 256-bit output)**:

     - Load 256 bits at offset :math:`(S / 2) \cdot O` so each :math:`M`-byte value resides within its 128-bit half after shuffling.
     - Build a compile-time ``pshufb`` index vector of length :math:`N \cdot S`.
       For each output byte index :math:`i`:

       - If :math:`i \bmod N < M`, map to the corresponding source byte, accounting for element index, padding, and offset.
       - Otherwise, set to :math:`-1` (``pshufb`` zero).

     - Apply ``_mm256_shuffle_epi8`` (within 128-bit lanes) to expand each :math:`M`-byte integer to :math:`N` bytes and zero-fill padding.

   - **x86-64-v4 (native 512-bit output)**:

     - Load 512 bits at offset :math:`(S / 2) \cdot O` so each :math:`M`-byte value resides in its final 256-bit half.
     - ``_mm512_permutexvar_epi32`` permutes 32-bit chunks (compile-time indices) so each :math:`M`-byte value resides in the target 128-bit lane.
       Bytes in first/last 128-bit lanes are shifted by :math:`\pm (S / 2) \cdot O`; middle lanes stay in place.
       After this, lanes 0 and 2 are bottom-aligned; lanes 1 and 3 are top-aligned.
     - Build a compile-time ``pshufb`` index vector of length :math:`N \cdot S`.
       For each output byte index :math:`i`:

       - If :math:`i \bmod N < M`, select the correct source byte based on:

         - position within the element,
         - element index within the 128-bit lane,
         - extra offset for top-aligned lanes.

       - Otherwise, set to :math:`-1`.

     - Apply ``_mm512_shuffle_epi8`` (within 128-bit lanes) to gather the :math:`M` data bytes and zero the :math:`O` padding bytes.
     - Variants using ``vpermb`` or ``vpermi2b``/``vpermt2b`` instead of the first permutation perform worse on Tigerlake and no better on Zen 5, and are not used.

   ****
   Neon
   ****

   - :math:`M = N`: direct :cpp:func:`~backend::load`.
   - :math:`N = 8`, :math:`M < 8`, 128-bit output: offset load, shift, merge, shift-back; for :math:`M = 5`:

     .. list-table::
        :header-rows: 1

        * - Operation
          - Pattern
        * - Load with offset :math:`O = 8 - M = 3`
          - ``...00000|11111...``
        * - 64-bit left shift by :math:`O` bytes
          - ``···..000|···11111``
        * - Copy high part of shifted lane into the other lane
          - ``...00000|···11111``
        * - 64-bit right shift by :math:`O` bytes
          - ``00000···|11111···``

   - :math:`N = 4`, :math:`M = 3`, 128-bit output: unaligned 128-bit load and table lookup via ``vqtbl1q_u8`` with a compile-time index vector.
   - :math:`N = 4`, :math:`M = 3`, 64-bit output: forward to the 128-bit implementation and wrap as a sub-vector.
