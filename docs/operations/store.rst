.. cpp:namespace:: grex

#######
Storing
#######

Vector storing operations write elements from SIMD vectors to contiguous scalar memory.
Sub-native vectors are embedded in a full native vector; super-native vectors are split into native halves.
Partial stores write a prefix without touching memory beyond the requested number of elements.

.. _operations-store:

**********
Store Full
**********

.. cpp:function:: void backend::store(T* dst, Vector<T, N> v)

   Stores all :math:`N` elements of ``v`` into ``dst[0..N-1]``.

   The pointer must be valid for scalar ``T`` access and may be unaligned.

   x86-64
   ======

   - **Native**: ``storeu``/``store`` intrinsics at the appropriate width.
   - **Sub-native**: minimal-width stores (e.g. ``_mm_storeu_si16``, ``_mm_storeu_si32``) on the backing register so exactly :math:`N` elements are written.

   Neon
   ====

   - **Native**: ``vst1q`` intrinsics.
   - **Sub-native**: implemented via :cpp:func:`~backend::store_part` on the backing native vector.

   Super-native (shared)
   =====================

   - Store lower and upper halves independently.

.. _operations-store-aligned:

*************
Store Aligned
*************

.. cpp:function:: void backend::store_aligned(T* dst, Vector<T, N> v)

   Stores all :math:`N` elements to an address assumed to be aligned for a full SIMD vector of ``T``.

   Behaviour matches :cpp:func:`~backend::store`, but may use alignment-sensitive intrinsics on x86-64.
   On Neon, aligned and unaligned stores are the same.

.. _operations-store-part-runtime:

******************************
Store Partial (Runtime Length)
******************************

.. cpp:function:: void backend::store_part(T* dst, Vector<T, N> v, std::size_t size)

   Stores up to ``size`` elements from ``v`` into ``dst[0..size-1]``, without writing beyond them.

   - If :math:`\text{size} \ge N`, equivalent to :cpp:func:`~backend::store`.
   - If :math:`\text{size} = 0`, stores nothing.

   x86-64
   ======

   Native vectors
   --------------

   - **x86-64-v4**: ``mask_storeu`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
   - **Earlier**:

     - **128-bit, 32/64-bit elements**:

       - **x86-64-v3**: ``maskstore`` intrinsics with integer masks.
       - **Earlier**: small case distinctions with 32/64-bit stores and ``std::memcpy``.

     - **128-bit, 8/16-bit elements**:

       - Store lower 64 bits via ``_mm_storeu_si64`` if :math:`\text{size} \ge N / 2`, then move remaining 64 bits to a GP register and handle via 4/2/1-byte ``std::memcpy``.

     - **256/512-bit**:

       - Split across halves: fully store the lower half if applicable, and partially store the remaining half.

   Sub-native vectors
   ------------------

   - **x86-64-v4**: forward to native :cpp:func:`~backend::store_part` on the backing register.
   - **Earlier**: specialized narrow sequences using ``std::memcpy`` and narrow stores (e.g. ``_mm_storeu_si32``, ``_mm_storeu_si64``).

   Neon
   ====

   Native vectors
   --------------

   - Let :math:`\text{bytes} = \text{size} \cdot \text{sizeof}(T)`.
     Write 8/4/2/1-byte blocks:

     - Use ``std::memcpy`` for leading bytes (typically compiled to ``str``).
     - Use lane-wise stores (``vst1q_lane_u8`` or ``st1`` via inline assembly) for tail bytes.

   Sub-native vectors
   ------------------

   - Dispatch per element count to the native partial store on the backing register.

   Super-native (shared)
   =====================

   - If :math:`\text{size} \le N / 2`, partial store of the lower half.
   - Otherwise, full store of the lower half and partial store of the upper half for the remainder.

.. _operations-store-part-ct:

***********************************
Store Partial (Compile-Time Length)
***********************************

.. cpp:function:: void backend::store_part(T* dst, Vector<T, N> v, AnyIndexTag auto size)

   Stores a compile-time-known number of elements ``size`` into ``dst[0..size-1]`` without writing beyond them.

   - If :math:`\text{size} = N`, equivalent to :cpp:func:`~backend::store`.
   - If :math:`\text{size} = 0`, stores nothing.

   Uses the same mechanisms as the runtime overload, but with ``if constexpr``/template-based specialization, allowing size-specific straight-line code.
