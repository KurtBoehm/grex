.. cpp:namespace:: grex

##########
Extraction
##########

Extraction operations read individual lanes from vectors and masks.
Sub-native vectors/masks are processed via their backing native registers, while a super-native vector/mask is handled by selecting the half that contains the requested lane.

Binary16 does not share the ``u16`` code path: ``u16`` extraction ends in a general-purpose register (which is where an integer belongs) whereas a binary16 value belongs in the vector register file.
Binary16 therefore has its own paths that never leave it (see :ref:`f16-implementation`).

.. _operations-extract-single:

*****************
Single-Lane Value
*****************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::extract_single(Vector<T, N> v)

   Returns the lowest lane :math:`v_0`.

   Shared
   ======

   - **Sub-native**: forward to the backing native vector.
   - **Super-native**: extract from the lower half.

   x86-64
   ======

   - **128-bit**:

     - **Floating point**: ``_mm_cvtss_f32``/``_mm_cvtsd_f64``, and ``_mm_cvtsh_h`` for binary16 with AVX512-FP16.
     - **Integers**:

       - **32/64-bit**: ``_mm_cvtsi128_si32``/``_mm_cvtsi128_si64`` with appropriate extension.
       - **8/16-bit**: extract via ``_mm_cvtsi128_si32`` and narrow.

     - **Binary16 without AVX512-FP16**: no intrinsic reads a binary16 lane, so the register is reinterpreted by an empty inline-assembly constraint (GCC) or a store-and-read round trip that the compiler folds away (Clang).

   - **256/512-bit**: extract the lowest 128 bits and delegate to the 128-bit implementation.

   Neon
   ====

   - **Native 128-bit**: ``vgetq_lane`` at lane 0, which exists for binary16 irrespective of the FP16 extension.

.. _operations-extract-value-runtime:

*******************************
Element Value by Run-Time Index
*******************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::extract(Vector<T, N> v, std::size_t index)

   Returns lane :math:`v_{\mathit{index}}` for :math:`\mathit{index} < N`.

   Shared
   ======

   - **Sub-native**: forward to the backing native vector.
   - **Super-native**:

     - If :math:`\mathit{index} < N / 2`: extract from ``v.lower`` at that index.
     - Otherwise: extract from ``v.upper`` at index :math:`\mathit{index} - N / 2`.

   x86-64
   ======

   - **x86-64-v2 and later**: permute the requested lane to the front of the register and read it with :cpp:func:`~backend::extract_single`.
     Since a permutation only moves bit patterns, it is performed on the unsigned integer type of the same width.

     - The permutation acts on *parts* of a width chosen per element width and register width: the element width itself where a variable permutation of that width exists, and 32 bits (``vpermd``) otherwise, mirroring the choices :cpp:func:`~backend::shuffle` makes.
       Bytes always use a byte shuffle in 128-bit registers, as it has the lowest latency even where ``vpermb`` exists.
     - Where the parts are wider than the elements, the permuted chunk holds several of them and the requested one is isolated by a shift: within a general-purpose register for integers, which end up there anyway, and within the vector register for binary16, which must not.
     - The control operand packs one part index per part into a single scalar, computed from the lane index by one multiplication and one addition of compile-time constants.

   - **x86-64-v1**: SSE2 offers no variable shuffle, so the vector is stored to a temporary array and read back at ``index``.

   Neon
   ====

   - **Native**: ``switch (index)`` dispatch to the compile-time-index ``vgetq_lane`` intrinsics.

.. _operations-extract-value-ct:

***********************************
Element Value by Compile-Time Index
***********************************

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  T backend::extract(Vector<T, N> v, I index)

   Returns lane :math:`v_{\mathit{index}}` for :math:`\mathit{index} < N` known at compile time.

   Shared
   ======

   - **Sub-native**: forward to the backing native vector.
   - **Super-native**:

     - If :math:`I < N / 2`: extract from ``v.lower`` at index ``I``.
     - Otherwise: extract from ``v.upper`` at index :math:`I - N / 2`.

   x86-64
   ======

   - **128-bit**:

     - **Integers** (lane-specific intrinsics where available):

       - **8-bit**: ``_mm_extract_epi8`` (or ``_mm_extract_epi16`` and shift on x86-64-v1).
       - **16-bit**: ``_mm_extract_epi16``.
       - **32-bit**: ``_mm_extract_epi32`` (or ``_mm_shuffle_epi32`` and ``_mm_cvtsi128_si32`` on x86-64-v1).
       - **64-bit**: ``_mm_extract_epi64`` (or ``_mm_unpackhi_epi64`` and ``_mm_cvtsi128_si64`` on x86-64-v1).

     - **Floating point**:

       - **16-bit**: shift the lane down to position 0 within the register and use :cpp:func:`~backend::extract_single`.
       - **32-bit**: shuffle the requested lane to position 0 with ``_mm_shuffle_epi32`` and use ``_mm_cvtss_f32``.
       - **64-bit**: select the high element via ``_mm_unpackhi_pd`` when needed, then ``_mm_cvtsd_f64``.

   - **256-bit**:

     - **Integers**: ``_mm256_extract`` intrinsics.
     - **32-bit floating point**:

       - Lower-half indices: from the lower 128 bits.
       - Upper-half indices: from the upper half with index adjusted by 4.

     - **64-bit floating point**:

       - Lower-half indices: from the lower 128 bits.
       - Upper-half indices: permute with ``_mm256_permute4x64_pd`` then ``_mm256_cvtsd_f64``.

   - **512-bit**:

     - **Integers**:

       - Lower-half indices: delegate to 256-bit :cpp:func:`~backend::extract`.
       - Upper-half indices: use ``_mm512_extracti32x4_epi32`` to obtain the 128-bit chunk, then 128-bit :cpp:func:`~backend::extract`.

     - **Floating point**:

       - Lower-half indices: delegate to 256-bit :cpp:func:`~backend::extract`.
       - Upper-half indices: use ``_mm512_alignr`` to move the lane to position 0, then :cpp:func:`~backend::extract_single`.

   - **Wider binary16 vectors**: select the 128-bit lane containing ``I`` and recurse.

   Neon
   ====

   - **Native**: ``vgetq_lane`` with the lane index encoded in ``I``.

.. _operations-extract-mask-runtime:

*******************************
Mask Bit by Run-Time Lane Index
*******************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  bool backend::extract(Mask<T, N> m, std::size_t index)

   Returns the Boolean value of mask lane :math:`\mathit{index} < N`.

   Shared
   ======

   - **Sub-native**: forward to the backing native mask.
   - **Super-native**:

     - If :math:`\mathit{index} < N / 2`: extract from ``m.lower`` at that index.
     - Otherwise: extract from ``m.upper`` at index :math:`\mathit{index} - N / 2`.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: convert the mask register to an integer and test bit ``index``.
   - **Earlier (broad masks)**: reinterpret as an unsigned integer vector and call run-time-index :cpp:func:`~backend::extract` on that vector; compare the result to zero.

   Neon
   ====

   - Reinterpret as an unsigned integer vector and call run-time-index :cpp:func:`~backend::extract`; return non-zero.

.. _operations-extract-mask-ct:

***********************************
Mask Bit by Compile-Time Lane Index
***********************************

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  bool backend::extract(Mask<T, N> m, I index)

   Returns the Boolean value of mask lane :math:`\mathit{index} < N` known at compile time.

   Shared
   ======

   - **Sub-native**: forward to the backing native mask.
   - **Super-native**:

     - If :math:`I < N / 2`: extract from ``m.lower`` at index ``I``.
     - Otherwise: extract from ``m.upper`` at index :math:`I - N / 2`.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: convert the mask register to an integer and bit-test position ``I::value`` (fully constant-foldable).
   - **Earlier (broad masks)**: reinterpret as an unsigned integer vector and call compile-time-index :cpp:func:`~backend::extract`; compare to zero.

   Neon
   ====

   - Reinterpret as an unsigned integer vector and call compile-time-index :cpp:func:`~backend::extract`; test non-zero.
