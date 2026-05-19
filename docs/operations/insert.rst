.. cpp:namespace:: grex

#########
Insertion
#########

Single-lane insertion into vectors and masks.

.. _operations-insert-value-ct:

*************************************
Vector Insertion (Compile-Time Index)
*************************************

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  Vector<T, N> backend::insert(Vector<T, N> v, I index, T value)

   Returns a copy of ``v`` with lane :math:`\text{index} < N` replaced by ``value``; all other lanes are unchanged.

   Shared
   ======

   - **Sub-native**: forwards to the backing native vector.

   - **Super-native**:

     - If :math:`\text{index} < N / 2`: insert into ``lower`` at ``index``; ``upper`` unchanged.
     - Otherwise: insert into ``upper`` at :math:`\text{index} - N / 2`; ``lower`` unchanged.

   x86-64
   ======

   - **128-bit**:

     - ``f64``: expand the scalar with :cpp:func:`~backend::expand_any` and combine via ``_mm_move_sd`` (index 0) or ``_mm_unpacklo_pd`` (index 1).
     - ``f32``:

       - **x86-64-v2+**: expand scalar and use ``_mm_move_ss`` (index 0) or ``_mm_insert_ps`` (other indices).
       - **x86-64-v1**: expand scalar and use ``_mm_move_ss`` (index 0) or shuffles (other indices).

     - **Integers**:

       - **16-bit**: ``_mm_insert_epi16``.
       - **Otherwise**:

         - **x86-64-v2+**: ``_mm_insert`` intrinsics.
         - **Otherwise**: fall back to the :ref:`run-time-index implementation <operations-insert-value-runtime>`.

   - **256-bit (x86-64-v3)**:

     - ``f32``/``f64``:

       - Form an auxiliary vector ``ins`` containing ``value`` in the appropriate lane using :cpp:func:`~backend::expand_any` (index 0), ``_mm_insert_ps``/``_mm_unpacklo_pd`` on the low 128 bits, or ``_mm256_broadcastss_ps``/``_mm256_broadcastsd_pd``.
       - Blend ``v`` with ``ins`` via ``_mm256_blend_ps``/``_mm256_blend_pd``.

     - **8-bit integers**:

       - **Index in lower 128 bits**: ``_mm_insert_epi8`` into lower half and ``_mm256_blend_epi32`` to merge the affected 32 bits.
       - **Otherwise**: Extract upper 128 bits with ``_mm256_extracti128_si256``

     - **Other integers**:

       - Form an auxiliary vector ``ins`` containing ``value`` in the appropriate lane:

         - **32/64-bit, index 0**: ``_mm_cvtsi32_si128``/``_mm_cvtsi64_si128``.
         - **Index in lower 128 bits**: ``_mm_insert`` intrinsics.
         - **Otherwise**: full-register broadcasts.

       - Combine ``v`` with ``ins`` using ``_mm256_blend_epi32``/``_mm256_blend_epi16``.

   - **512-bit (x86-64-v4)**:

     - If the bit position corresponding to ``index`` lies in the low 128 bits, cast to a 128-bit vector, perform a 128-bit compile-time insert, then insert that 128-bit chunk back into the 512-bit register with a 128-bit ``insert`` intrinsic.
     - Otherwise, delegate to the :ref:`run-time-index implementation <operations-insert-value-runtime>`.

   Neon
   ====

   - ``vsetq_lane`` intrinsics.

.. _operations-insert-mask-ct:

***********************************
Mask Insertion (Compile-Time Index)
***********************************

.. cpp:function:: template<Vectorizable T, std::size_t N, AnyIndexTag I> \
                  Mask<T, N> backend::insert(Mask<T, N> m, I index, bool value)

   Returns a copy of ``m`` with mask lane :math:`\text{index} < N` set to ``value``; remaining lanes are unchanged.

   Shared
   ======

   - **Sub-native**: as for vectors, insert into the backing native mask and re-wrap.
   - **Super-native**: select ``lower``/``upper`` half according to ``index`` (compile-time branch) and recurse with adjusted index.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**:

     - Delegated to the :ref:`run-time-index mask insertion <operations-insert-mask-runtime>`.

   - **Earlier (broad masks)**:

     - Convert ``value`` to a signed all-ones/all-zeros integer lane via negation of the Boolean value.
     - Perform compile-time-index vector insertion on the corresponding integer vector type, then reinterpret as a mask.

   Neon
   ====

   - Convert ``value`` to an unsigned all-ones/all-zeros integer lane via negation of the Boolean value.
   - Insert with the appropriate ``vsetq_lane`` intrinsic on the underlying unsigned mask type.

.. _operations-insert-value-runtime:

*********************************
Vector Insertion (Run-Time Index)
*********************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::insert(Vector<T, N> v, std::size_t index, T value)

   Returns a copy of ``v`` with lane :math:`\text{index} < N` replaced by ``value``; all other lanes are unchanged.
   Behaviour is undefined if :math:`\text{index} \ge N`.

   Shared
   ======

   - **Sub-native**: insert into the backing native vector at run-time index and re-wrap.
   - **Super-native**:

     - If :math:`\text{index} < N / 2`: insert into ``lower`` at ``index``.
     - Otherwise: insert into ``upper`` at :math:`\text{index} - N / 2`.

   x86-64
   ======

   - **x86-64-v4**:

     - Uses masked broadcasts of the scalar with a :cpp:func:`~backend::single_mask`:

       - ``f32``: ``mask_broadcastss_ps`` intrinsics.
       - ``f64``: ``_mm_mask_movedup_pd`` (size 2) or ``mask_broadcastsd_pd`` intrinsics.
       - **Integers**: ``mask_set1`` intrinsics.

   - **Earlier**:

     - Build a single-lane mask via :cpp:func:`~backend::single_mask`; only lane ``index`` is set.
     - Broadcast ``value`` to all lanes with :cpp:func:`~backend::broadcast`.
     - Blend old and new vectors with :cpp:func:`~backend::blend`, keeping ``v`` where the mask is false and the broadcast where it is true.

   Neon
   ====

   - Dispatch via ``switch (index)`` to ``vsetq_lane`` intrinsics for each lane position.

.. _operations-insert-mask-runtime:

*************************************
Single-Lane Mask Bit (Run-Time Index)
*************************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::insert(Mask<T, N> m, std::size_t index, bool value)

   Returns a copy of ``m`` with mask lane :math:`\text{index} < N` set to ``value``; remaining lanes are unchanged.
   Behaviour is undefined if :math:`\text{index} \ge N`.

   Shared
   ======

   - **Sub-native**: insert into the backing native mask and re-wrap.
   - **Super-native**:

     - If :math:`\text{index} < N / 2`: insert into ``lower``; ``upper`` unchanged.
     - Otherwise: insert into ``upper`` at :math:`\text{index} - N / 2`; ``lower`` unchanged.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**: treat the mask register as an unsigned integer bitfield:

     .. math::

        r = (m \land \neg(1 \ll \text{index})) \lor (\text{value} \ll \text{index}).

   - **Earlier (broad masks)**:

     - Convert ``value`` to a signed all-ones/all-zeros integer lane via negation of the Boolean value.
     - Perform run-time-index vector insertion on the corresponding integer vector type and reinterpret the result as a mask.

   Neon
   ====

   - Convert ``value`` to an unsigned all-ones/all-zeros integer lane via negation of the Boolean value.
   - Use the vector run-time insert path on the unsigned mask vector (``vsetq_lane`` via ``switch (index)``), then reinterpret as a mask.
