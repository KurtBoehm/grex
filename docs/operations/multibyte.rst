.. cpp:namespace:: grex

#########################
Multibyte Integer Loading
#########################

Loading of packed integers whose logical value spans :math:`M` bytes into SIMD vectors with element width :math:`N = 2^B` bytes, where :math:`N = \text{bitceil}(M)`.
The input memory is assumed to be padded on both sides by at least one full SIMD register.

Sub-native vectors are embedded in a full native vector; super-native vectors are assembled from native halves.

.. _operations-load-multibyte:

.. cpp:function:: template<std::size_t SrcBytes, AnyVector Dst> \
                  Dst backend::load_multibyte(const u8* ptr, IndexTag<SrcBytes>, TypeTag<Dst>)

   Load ``Dst::size`` integers, each represented by ``SrcBytes`` consecutive bytes starting at ``ptr``, into a SIMD vector whose element type is ``Dst::Value``.

   - :math:`M = \mathtt{SrcBytes}` is the number of source bytes per logical integer.
   - :math:`N = \mathtt{sizeof(Dst::Value)} \geq M` is the destination byte width, and a power of two.
   - :math:`O = N - M` is the number of zero-padding bytes to be added.
   - :math:`S = \mathtt{Dst::size}` is the number of elements to be loaded.

   Each lane :math:`i` receives the zero-extended little-endian value of bytes :math:`[i \cdot M, (i + 1) \cdot M)` from memory, stored in the low :math:`M` bytes of that lane; the remaining higher bytes are zeroed.

   The visualizations below show the index of the logical value contained in each byte position; ``.`` represents an unspecified byte and ``·`` represents a zero byte.
   A ``|`` may be added for readability and has no semantic meaning.

   ******
   x86-64
   ******

   - :math:`M = N`: loads directly using :cpp:func:`~backend::load`.
   - x86-64-v1:

     - :math:`N = 8`, :math:`M < 8`, 128-bit output: load with an offset, shift, merge lanes, then shift back; for :math:`M = 5`:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load 16 bytes at offset :math:`O = 8 - M = 3` so each 5-byte value is entirely within its lane
            - ``...00000|11111...``
          * - 64-bit logical left shift by :math:`O` bytes to bring the upper value to the top of its lane
            - ``···..000|···11111``
          * - Merge so that both values are top-aligned
            - ``...00000|···11111``
          * - 64-bit logical right shift by :math:`O` bytes to bottom-align both values and zero the high bytes
            - ``00000···|11111···``

     - :math:`N = 4`, :math:`M = 3`, 128-bit output: load with an offset, then use shifts and merges:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load with an offset of :math:`2` bytes so that the four values are each contained in their 32-bit lanes
            - ``..00|0111|2223|33..``
          * - 64-bit left shifts to move each 3-byte value to the top of its 32-bit lane
            - | ``···.|.000|···1|1122``
              | ``····|·..0|····|·222``
              | ``··..|0001|··22|2333``
          * - Merge halves so that each 32-bit lane is top-aligned
            - | ``···.|..00|.000|0111``
              | ``····|··22|·222|2333``
          * - Merge so that all values are top-aligned
            - ``.000|·111|·222|2333``
          * - 32-bit right shift by :math:`1` byte to bottom-align and zero-fill the high bytes
            - ``000·|111·|222·|333·``

     - :math:`N = 4`, :math:`M = 3`, 64-bit output: simplified variant of the 128-bit implementation:

       .. list-table::
          :header-rows: 1

          * - Operation
            - Pattern
          * - Load with an offset of :math:`1` byte so that the two values are each in their 32-bit lanes
            - ``.000|111.|....|....``
          * - 32-bit logical left shift by :math:`1` byte so that the upper value is aligned to the top of its 32-bit lane
            - ``·.00|·111|·...|·...``
          * - Merge so that both values are top-aligned
            - ``.000|·111|·...|·...``
          * - 32-bit logical right shift by :math:`1` byte to bottom-align and zero-fill the high bytes
            - ``000·|111·|...·|...·``

   - Starting on x86-64-v2:

     - 128-bit baseline: loads the packed bytes without offset and shuffles with ``_mm_shuffle_epi8`` using a compile-time index table.
     - :math:`N = 8`, :math:`M = 6`, 128-bit output: loads with an offset of :math:`2` bytes, applies ``_mm_shufflelo_epi16`` to reorder the low half, then uses ``_mm_blend_epi16`` to clear the unused high bytes in each 64-bit lane.

   - Starting on x86-64-v3 (native 256-bit output):

     - Loads a full 256-bit vector at an offset of :math:`(S / 2) \cdot O` so that each :math:`M`-byte value is fully resident in the 128-bit half it ends up in after shuffling.
     - Builds a compile-time ``pshufb`` index vector of length :math:`N \cdot S`.
       For each output byte index :math:`i`:

       - If :math:`i \bmod N < M`, it is mapped to the corresponding source byte inside its element, taking into account the inter-element padding and offset.
       - Otherwise, it is set to :math:`-1`, causing ``pshufb`` to zero that byte.

     - Applies ``_mm256_shuffle_epi8`` (operating independently in each 128-bit lane) to expand each :math:`M`-byte integer into a full-width :math:`N`-byte lane and zero-fill the remaining high bytes.

   - Starting on x86-64-v4 (native 512-bit output):

     - Loads a full 512-bit vector at an offset of :math:`(S / 2) \cdot O` so that each :math:`M`-byte value is fully resident in the 256-bit half it ends up in after shuffling.
     - First uses ``_mm512_permutexvar_epi32`` to permute 32-bit chunks (indices computed at compile time) so that each :math:`M`-byte value is fully resident in the 128-bit lane it ends up in after shuffling.
       Bytes in the first/last 128-bit lanes are shifted by :math:`\pm (S / 2) \cdot O` to skip padding, while the two middle 128-bit lanes stay in place.
       After this, lanes 0 and 2 (even indices) are bottom-aligned while lanes 1 and 3 are top-aligned within their 128-bit lanes.
     - Then builds a compile-time ``pshufb`` index vector of length :math:`N \cdot S`.
       For each output byte index :math:`i`:

       - If :math:`i \bmod N < M`, it selects the corresponding source byte inside the permuted data, accounting for

         - the position within the element,
         - the element index within the 128-bit lane, and
         - an additional offset for lanes that are top-aligned after the 32-bit permutation.

       - Otherwise, it is set to :math:`-1` to zero-fill padding bytes.

     - Applies ``_mm512_shuffle_epi8`` (within 128-bit lanes) using this index vector to gather the :math:`M` data bytes per lane and zero the remaining :math:`O` bytes.

   ****
   Neon
   ****

   - :math:`M = N`: loads directly using :cpp:func:`~backend::load`.
   - :math:`N = 8`, :math:`M < 8`, 128-bit output: load with an offset, shift, merge lanes, then shift back; for :math:`M = 5`:

     .. list-table::
        :header-rows: 1

        * - Operation
          - Pattern
        * - Load with an offset of :math:`O = 8 - M = 3` bytes so that the two values are each in their lanes
          - ``...00000|11111...``
        * - 64-bit logical left shift by :math:`O` bytes so that the upper value is aligned to the top
          - ``···..000|···11111``
        * - Copy the high part of the shifted lane into the other lane to top-align both values
          - ``...00000|···11111``
        * - 64-bit logical right shift by :math:`O` bytes to bottom-align and zero-fill
          - ``00000···|11111···``
   - :math:`N = 4`, :math:`M = 3`, 128-bit output: single unaligned 128-bit load followed by a table lookup with ``vqtbl1q_u8`` using a compile-time index vector.
   - :math:`N = 4`, :math:`M = 3`, 64-bit output: forwards to the 128-bit implementation and wraps the result as a sub-vector.
