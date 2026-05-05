.. cpp:namespace:: grex

##########
Extraction
##########

Extraction operations read individual lanes from vectors and masks.
Sub-native vectors/masks are embedded in a full native object; super-native vectors/masks are traversed by selecting the appropriate native half.

*****************
Single-Lane Value
*****************

.. _operations-extract-single:

.. cpp:function:: template<Vectorizable T> \
                  T backend::extract_single(Vector<T, N> v)

   Returns the value of the lowest lane :math:`v_0`.

   x86-64
   ======

   - **128-bit vectors**:

     - **Floating point**: uses ``_mm_cvtss_f32``/``_mm_cvtsd_f64`` on the underlying register.
     - **Integers**:

       - **32/64-bit**: uses ``_mm_cvtsi128_si32``/``_mm_cvtsi128_si64`` with appropriate sign/zero extension.
       - **8/16-bit**: extracts via ``_mm_cvtsi128_si32`` and narrows to the requested width.

   - **256/512-bit vectors**: Extracts the lowest 128 bits and delegates to the 128-bit implementation.

   Neon
   ====

   - **Native 128-bit vectors**: uses ``vgetq_lane`` intrinsics on lane ``0``.

   Shared
   ======

   - **Sub-vector**: forwards to the backing native vector.
   - **Super-vector**: extracts from the lower half.

*******************************
Element Value by Run-Time Index
*******************************

.. _operations-extract-value-runtime:

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::extract(Vector<T, N> v, std::size_t index)

   Returns the scalar value stored in lane :math:`\text{index} < N`, i.e. :math:`v_{\text{index}}`.

   x86-64
   ======

   - **On x86-64-v4 (only with AVX-512VBMI2 for 8/16-bit entries)**:

     - Uses the appropriate ``maskz_compress`` intrinsic driven by :cpp:func:`~backend::single_mask` to compress the selected lane into position 0, then extracts that lane with:

       - Floating point: ``_mm_cvtss_f32`` / ``_mm_cvtsd_f64``.
       - Integers: ``_mm_cvtsi128_si32`` / ``_mm_cvtsi128_si64`` plus narrowing/sign extension to the requested width.

   - **Earlier**: stores ``v`` to a temporary array via :cpp:func:`~backend::store` and returns ``array[index % N]``.

   Neon
   ====

   - **Native vectors**: uses a ``switch (index)`` over all lanes and dispatches to ``vgetq_lane`` intrinsics on the underlying register.

   Shared
   ======

   - **Sub-vector**: forwards to the backing native vector.
   - **Super-vector**: splits the logical index across halves:

     - If :math:`i < N / 2`, returns :cpp:func:`~backend::extract` from ``v.lower`` at index :math:`i`.
     - Otherwise, returns :cpp:func:`~backend::extract` from ``v.upper`` at index :math:`i - N / 2`.

***********************************
Element Value by Compile-Time Index
***********************************

.. _operations-extract-value-ct:

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  T backend::extract(Vector<T, N> v, I index)

   Returns the scalar value stored in lane :math:`\text{index} < N`, i.e. :math:`v_{\text{index}}`.

   x86-64
   ======

   - **128-bit integer vectors**:

     - **Integers**:

       - Uses dedicated lane-extract intrinsics where available:

         - **8-bit**: ``_mm_extract_epi8`` (or ``_mm_extract_epi16`` with shift on x86-64-v1).
         - **16-bit**: ``_mm_extract_epi16``.
         - **32-bit**: ``_mm_extract_epi32`` (or ``_mm_shuffle_epi32`` with ``_mm_cvtsi128_si32`` on x86-64-v1).
         - **64-bit**: ``_mm_extract_epi64`` (or ``_mm_unpackhi_epi64`` with ``_mm_cvtsi128_si64`` on x86-64-v1).

     - **Floating point**:

       - **32-bit**: shuffles the requested lane to position 0 with ``_mm_shuffle_epi32`` and converts with   ``_mm_cvtss_f32``.
       - **64-bit**: selects the high element via ``_mm_unpackhi_pd`` when needed, then converts with ``_mm_cvtsd_f64``.

   - **256-bit vectors**:

     - **Integers**: uses ``_mm256_extract`` intrinsics for all integer lane sizes.
     - **32-bit floating point**:

       - **Indices in the lower half**: extracts from lower 128-bit half.
       - **Indices in the upper half**: extracts from upper 128-bit half with the index adjusted by 4.

     - **64-bit floating point**:

       - **Indices in the lower half**: extracts from lower 128-bit half.
       - **Indices in the upper half**: permutes with ``_mm256_permute4x64_pd`` and converts the selected lane via ``_mm256_cvtsd_f64``.

   - **512-bit vectors**:

     - **Integers**:

       - **Indices in the lower half**: delegates to :cpp:func:`~backend::extract` for lower 256-bit half.
       - **Indices in the upper half**: uses ``_mm512_extracti32x4_epi32`` to obtain the 128-bit lane containing the element and delegates to 128-bit :cpp:func:`~backend::extract`.

     - **Floating point**:

       - **Indices in the lower half**: delegates to :cpp:func:`~backend::extract` for lower 256-bit half.
       - **Indices in the upper half**: uses ``_mm512_alignr`` intrinsics to shift the requested lane into position 0 and calls :cpp:func:`~backend::extract_single` on the resulting vector.

   Neon
   ====

   - **Native vectors**: calls ``vgetq_lane`` directly with the constant lane index encoded in ``I``.

   Shared
   ======

   - **Sub-vector**: forwards to the backing native vector.
   - **Super-vector**: splits the logical index across halves:

     - If :math:`I < N / 2`, returns :cpp:func:`~backend::extract` from ``v.lower`` at index ``I``.
     - Otherwise, returns :cpp:func:`~backend::extract` from ``v.upper`` at index :math:`i - N / 2`.

*******************************
Mask Bit by Run-Time Lane Index
*******************************

.. _operations-extract-mask-runtime:

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  bool backend::extract(Mask<T, N> m, std::size_t index)

   Returns the Boolean value of mask lane :math:`\text{index} < N`, i.e. whether that lane is set (:math:`\text{true}`) or cleared (:math:`\text{false}`).

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: converts the mask register to an integer and tests the bit at position ``index`` via a bit-test operation.
   - **Earlier (broad masks)**: reinterprets the mask as an unsigned integer vector of the same lane width and calls the run-time-index :cpp:func:`~backend::extract` on that vector, comparing the resulting element to zero.

   Neon
   ====

   - Reinterprets the mask as an unsigned integer vector of the same lane width and calls the run-time-index :cpp:func:`~backend::extract` on that vector, returning whether the extracted element is non-zero.

   Shared
   ======

   - **Sub-mask**: forwards to the backing native mask.
   - **Super-mask**: splits the logical index across halves, analogous to super-vectors:

     - If :math:`i < N / 2`, returns :cpp:func:`~backend::extract` from ``m.lower`` at index :math:`i`.
     - Otherwise, returns :cpp:func:`~backend::extract` from ``m.upper`` at index :math:`i - N / 2`.

***********************************
Mask Bit by Compile-Time Lane Index
***********************************

.. _operations-extract-mask-ct:

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  bool backend::extract(Mask<T, N> m, I index)

   Returns the Boolean value of mask lane :math:`\text{index} < N`, i.e. whether that lane is set (:math:`\text{true}`) or cleared (:math:`\text{false}`).

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: as in the run-time case, converts the mask register to an integer and tests the bit at position ``I::value`` via a bit-test operation; the constant index allows the test to be fully constant-folded where possible.
   - **Earlier (broad masks)**: reinterprets the mask as an unsigned integer vector of the same lane width and calls the compile-time-index :cpp:func:`~backend::extract` on that vector, comparing the resulting element to zero.

   Neon
   ====

   - Reinterprets the mask as an unsigned integer vector of the same lane width and calls the compile-time-index :cpp:func:`~backend::extract` on that vector, returning whether the extracted element is non-zero.

   Shared
   ======

   - **Sub-mask**: forwards to the backing native mask.
   - **Super-mask**: splits the logical index across halves, analogous to super-vectors:

     - If :math:`I < N / 2`, returns :cpp:func:`~backend::extract` from ``m.lower`` at index ``I``.
     - Otherwise, returns :cpp:func:`~backend::extract` from ``m.upper`` at index :math:`i - N / 2`.
