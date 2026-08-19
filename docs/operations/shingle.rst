.. cpp:namespace:: grex

#########
Shingling
#########

*Shingle* (a portmanteau of *shift single*) operations move elements by one position toward the front or back, inserting a scalar or zero into the vacated lane.
Only bits are moved, so binary16 shares the ``u16`` implementation (see :ref:`f16-implementation`).

.. _operations-shingle-up-zero:

************************
Shingle Up (Insert Zero)
************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_up(Vector<T, N> v)

   Shifts elements toward higher lanes by one lane, inserting zero into the first lane:

   .. math::

      r_j =
      \begin{cases}
        0       & j = 0 \\
        v_{j-1} & 1 \le j < N
      \end{cases}

   x86-64
   ======

   - **Native 128-bit**: byte-wise left shift using ``_mm_bslli_si128``.
   - **Native 256-bit (x86-64-v3+)**:

     - Form a helper with zeros in the lower 128 bits (``_mm256_setzero_si256``) and the original lower half in the upper 128 bits (``_mm256_castsi256_si128`` and ``_mm256_inserti128_si256``).
     - Use ``_mm256_alignr_epi8`` between the original and helper to shift each 128-bit half up by one element, inserting zero into the lowest lane of the lower 128 bits and the last value of the original lower half into the lowest lane of the upper half.

   - **Native 512-bit (x86-64-v4)**:

     - **32/64-bit elements**: ``_mm512_alignr_epi{32,64}``, which shift across lanes, between the input and a zero vector.
     - **8/16-bit elements**: analogous to the 256-bit implementation, using ``_mm512_alignr_epi64`` instead of ``_mm256_inserti128_si256`` to move each 128-bit lane up by one and ``_mm512_alignr_epi8`` for shifting each 128-bit lane.

   - **Sub-native**: Use ``slli`` intrinsics as wide as the active part of the register.

   Neon
   ====

   - **Native 128-bit**: ``vextq`` between a zero vector (``vdupq_n(0)``) and ``v`` with element offset :math:`N - 1`.
   - **Sub-native**: extract the low 64 bits with ``vget_low``, apply ``vext`` on the low 64 bits analogously to the 128-bit case, then re-expand and re-wrap.

.. _operations-shingle-up-front:

**************************
Shingle Up (Insert Scalar)
**************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_up(T front, Vector<T, N> v)

   Analogous to :cpp:func:`backend::shingle_up(Vector\<T, N\>) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> backend::shingle_up(Vector\<T, N\> v)>`, but inserts ``front`` into the first lane:

   .. math::

      r_j =
      \begin{cases}
        \mathit{front} & j = 0 \\
        v_{j-1}        & 1 \le j < N
      \end{cases}

   x86-64
   ======

   - **Native 128-bit**:

     - **64-bit elements**: scalar expansion of ``front``, then ``_mm_unpacklo_epi64``.
     - **Smaller elements**: byte-wise left shift using ``_mm_bslli_si128`` followed by different approaches to copy ``front``:

       - **32/8-bit integers on x86-64-v1**: zero-extend ``front`` to 128 bits and combine with shifted vector using bitwise OR.
       - **Otherwise**: :cpp:func:`backend::insert() <template\<Vectorizable T, std::size_t N, AnyIndexTag I\> Vector\<T, N\> backend::insert(Vector\<T, N\> v, I index, T value)>` into lane 0.

   - **Native 256-bit (x86-64-v2+)**: broadcast ``front`` to 128 bits, combine with the lower half of ``v`` in the upper half, and perform ``_mm256_shuffle_pd`` (64-bit elements) or ``_mm256_alignr_epi8`` (smaller elements).
   - **Native 512-bit (x86-64-v4)**: analogous to the zero-inserting variant with the broadcast ``front`` used as the second argument to the ``alignr`` operations instead of zeros.
   - **Sub-native**: Use ``slli`` intrinsics as wide as the active part of the register, then use the same strategies as the corresponding 128-bit variant to insert ``front``.

   Neon
   ====

   - **Floating-point**: expand ``front`` with :cpp:func:`~backend::expand_any` and use ``vextq`` to move it to the highest lane, then use ``vextq`` for the shift.
   - **Integers**: shift ``v`` using ``vextq``, then insert ``front`` into lane 0 using ``vsetq_lane``.
   - **Sub-native**: extract the low 64 bits with ``vget_low``, apply ``vext`` on the low 64 bits and insert ``front`` into lane 0 with ``vsetq_lane`` analogously to the 128-bit case, then re-expand and re-wrap.

.. _operations-shingle-down-zero:

**************************
Shingle Down (Insert Zero)
**************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_down(Vector<T, N> v)

   Shifts elements toward lower lanes by one lane, inserting zero into the last lane:

   .. math::

      r_j =
      \begin{cases}
        v_{j+1} & 0 \le j < N - 1 \\
        0       & j = N - 1
      \end{cases}

   x86-64
   ======

   - **Native 128-bit**: byte-wise right shift using ``_mm_bsrli_si128``.
   - **Native 256-bit (x86-64-v3+)**:

     - Form a helper with the original upper half in the lower 128 bits and zeros in the upper 128 bits (``_mm256_extracti128_si256`` and ``_mm256_zextsi128_si256``).
     - Use ``_mm256_alignr_epi8`` between the original and helper to shift each 128-bit half down by one element, inserting the first value of the original upper half into the last lane of the lower 128 bits and zero into the last lane of the upper 128 bits.
     - If x86-64-v4 is available, 32/64-bit values use the cross-lane ``_mm256_alignr_epi32``/``_mm256_alignr_epi64`` to avoid the need for the first step.

   - **Native 512-bit (x86-64-v4)**:

     - **32/64-bit elements**: ``_mm512_alignr_epi{32,64}``, which shift across lanes, between the input and a zero vector.
     - **8/16-bit elements**: analogous to the 256-bit implementation, using ``_mm512_alignr_epi64`` instead of ``_mm256_inserti128_si256`` to move each 128-bit lane down by one and ``_mm512_alignr_epi8`` for shifting each 128-bit lane.

   - **Sub-native**: Use ``srli`` intrinsics as wide as the active part of the register.

   Neon
   ====

   - **Native 128-bit**: ``vextq(v, vdupq_n(0), 1)``.
   - **Sub-native**: extract the low 64 bits with ``vget_low``, apply a small ``vext``/zero sequence, then re-expand.

.. _operations-shingle-down-back:

****************************
Shingle Down (Insert Scalar)
****************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_down(Vector<T, N> v, T back)

   Analogous to :cpp:func:`backend::shingle_down(Vector\<T, N\>) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> backend::shingle_down(Vector\<T, N\> v)>`, but inserts ``back`` into the last lane:

   .. math::

      r_j =
      \begin{cases}
        v_{j+1} & 0 \le j < N - 1 \\
        \text{back} & j = N - 1
      \end{cases}

   x86-64
   ======

   - **128-bit**:

     - **32/64-bit floating point**: scalar expansion of ``back`` combined with shuffles/unpacks to place it in the final lane after shifting.
     - **Integers and smaller types**: byte shifts plus ``insert_epi*`` to write ``back`` into the last position.

   - **256/512-bit**: broadcast ``back`` and combine with the shifted vector via ``alignr``-based patterns, analogous to the zero-inserting variant.
   - **Sub-native**: specialized shuffle/shift plus ``insert_epi*`` sequences on 128-bit temporaries.

   Neon
   ====

   - **Native 128-bit**: expand ``back`` with :cpp:func:`~backend::expand_any` and use ``vextq(v, vback, 1)`` so that the last element comes from ``back``.
   - **Sub-native**: operate on the low 64 bits with ``vget_low``, ``vext``, and ``vset_lane`` using an expanded ``back``, then re-expand to sub-native form.
