.. cpp:namespace:: grex

##########
Extraction
##########

Extraction operations read individual lanes from vectors and masks.

*****************
Single-Lane Value
*****************

.. _operations-extract-single:

.. cpp:function:: template<Vectorizable T> \
                  T backend::extract_single(Vector<T, N> v)

   Returns the lowest lane :math:`v_0`.

   x86-64
   ======

   - **128-bit**:

     - **Floating point**: ``_mm_cvtss_f32``/``_mm_cvtsd_f64``.
     - **Integers**:

       - **32/64-bit**: ``_mm_cvtsi128_si32``/``_mm_cvtsi128_si64`` with appropriate extension.
       - **8/16-bit**: extract via ``_mm_cvtsi128_si32`` and narrow.

   - **256/512-bit**: extract the lowest 128 bits and delegate to the 128-bit implementation.

   Neon
   ====

   - **Native 128-bit**: ``vgetq_lane`` at lane 0.

   Shared
   ======

   - **Sub-vector**: forward to the backing native vector.
   - **Super-vector**: extract from the lower half.

*******************************
Element Value by Run-Time Index
*******************************

.. _operations-extract-value-runtime:

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::extract(Vector<T, N> v, std::size_t index)

   Returns lane :math:`v_{\text{index}}` for :math:`\text{index} < N`.

   x86-64
   ======

   - **x86-64-v4 (with AVX-512VBMI2 for 8/16-bit)**:

     - Use ``maskz_compress`` intrinsics driven by :cpp:func:`~backend::single_mask` to move the selected lane to position 0, then:

       - Floating point: ``_mm_cvtss_f32``/``_mm_cvtsd_f64``.
       - Integers: ``_mm_cvtsi128_si32``/``_mm_cvtsi128_si64`` plus narrowing/extension.

   - **Earlier**: store via :cpp:func:`~backend::store` to a temporary array and return ``array[index % N]``.

   Neon
   ====

   - **Native**: ``switch (index)`` dispatch to ``vgetq_lane`` intrinsics.

   Shared
   ======

   - **Sub-vector**: forward to the backing native vector.
   - **Super-vector**:

     - If :math:`\text{index} < N / 2`: extract from ``v.lower`` at that index.
     - Otherwise: extract from ``v.upper`` at index :math:`\text{index} - N / 2`.

***********************************
Element Value by Compile-Time Index
***********************************

.. _operations-extract-value-ct:

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  T backend::extract(Vector<T, N> v, I index)

   Returns lane :math:`v_{\text{index}}` for :math:`\text{index} < N` known at compile time.

   x86-64
   ======

   - **128-bit**:

     - **Integers** (lane-specific intrinsics where available):

       - **8-bit**: ``_mm_extract_epi8`` (or ``_mm_extract_epi16`` and shift on x86-64-v1).
       - **16-bit**: ``_mm_extract_epi16``.
       - **32-bit**: ``_mm_extract_epi32`` (or ``_mm_shuffle_epi32`` and ``_mm_cvtsi128_si32`` on x86-64-v1).
       - **64-bit**: ``_mm_extract_epi64`` (or ``_mm_unpackhi_epi64`` and ``_mm_cvtsi128_si64`` on x86-64-v1).

     - **Floating point**:

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

   Neon
   ====

   - **Native**: ``vgetq_lane`` with the lane index encoded in ``I``.

   Shared
   ======

   - **Sub-vector**: forward to the backing native vector.
   - **Super-vector**:

     - If :math:`I < N / 2`: extract from ``v.lower`` at index ``I``.
     - Otherwise: extract from ``v.upper`` at index :math:`I - N / 2`.

*******************************
Mask Bit by Run-Time Lane Index
*******************************

.. _operations-extract-mask-runtime:

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  bool backend::extract(Mask<T, N> m, std::size_t index)

   Returns the Boolean value of mask lane :math:`\text{index} < N`.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: convert the mask register to an integer and test bit ``index``.
   - **Earlier (broad masks)**: reinterpret as an unsigned integer vector and call run-time-index :cpp:func:`~backend::extract` on that vector; compare the result to zero.

   Neon
   ====

   - Reinterpret as an unsigned integer vector and call run-time-index :cpp:func:`~backend::extract`; return non-zero.

   Shared
   ======

   - **Sub-mask**: forward to the backing native mask.
   - **Super-mask**:

     - If :math:`\text{index} < N / 2`: extract from ``m.lower`` at that index.
     - Otherwise: extract from ``m.upper`` at index :math:`\text{index} - N / 2`.

***********************************
Mask Bit by Compile-Time Lane Index
***********************************

.. _operations-extract-mask-ct:

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  bool backend::extract(Mask<T, N> m, I index)

   Returns the Boolean value of mask lane :math:`\text{index} < N` known at compile time.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: convert the mask register to an integer and bit-test position ``I::value`` (fully constant-foldable).
   - **Earlier (broad masks)**: reinterpret as an unsigned integer vector and call compile-time-index :cpp:func:`~backend::extract`; compare to zero.

   Neon
   ====

   - Reinterpret as an unsigned integer vector and call compile-time-index :cpp:func:`~backend::extract`; test non-zero.

   Shared
   ======

   - **Sub-mask**: forward to the backing native mask.
   - **Super-mask**:

     - If :math:`I < N / 2`: extract from ``m.lower`` at index ``I``.
     - Otherwise: extract from ``m.upper`` at index :math:`I - N / 2`.
