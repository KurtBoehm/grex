.. cpp:namespace:: grex

#######
Storing
#######

Vector storing operations write elements from SIMD vectors to contiguous scalar memory.
Sub-native vectors are embedded in a full native vector; super-native vectors are split into native halves.
Partial stores write a prefix of the vector without touching memory beyond the requested number of elements.

.. _operations-store:

**********
Store Full
**********

.. cpp:function:: void backend::store(T* dst, Vector<T, N> v)

   Stores all :math:`N` elements of ``v`` into ``dst[0..N-1]``.

   The pointer must be valid for scalar ``T`` access and may be unaligned with respect to SIMD alignment.

   x86-64
   ======

   - Native vectors: use ``storeu``/``store`` intrinsics at the appropriate width.
   - Sub-native vectors: use minimal-width stores (e.g. ``_mm_storeu_si16``, ``_mm_storeu_si32``) on the backing register so that exactly :math:`N` elements are written.

   Neon
   ====

   - Native vectors: use ``vst1q`` intrinsics.
   - Sub-native vectors: implemented via :cpp:func:`backend::store_part` on the backing native vector.

   Super-native vectors (shared)
   =============================

   - Store lower and upper halves independently.

.. _operations-store-aligned:

*************
Store Aligned
*************

.. cpp:function:: void backend::store_aligned(T* dst, Vector<T, N> v)

   Stores all :math:`N` elements to an address assumed to be aligned for a full SIMD vector of ``T``.

   Behaviour matches :cpp:func:`backend::store`, but may use alignment-sensitive intrinsics on x86-64.
   On Neon, aligned and unaligned stores are not differentiated.

.. _operations-store-part-runtime:

******************************
Store Partial (Runtime Length)
******************************

.. cpp:function:: void backend::store_part(T* dst, Vector<T, N> v, std::size_t size)

   Stores up to ``size`` elements from ``v`` into ``dst[0..size-1]``, without writing beyond them.

   - If ``size >= N``, equivalent to :cpp:func:`backend::store`.
   - If ``size == 0``, stores nothing.

   x86-64
   ======

   Native vectors
   --------------

   - x86-64-v4: use ``mask_storeu`` intrinsics with a mask produced by :cpp:func:`backend::cutoff_mask`.
   - Earlier:

     - 128-bit, 32/64-bit elements:

       - x86-64-v3: use ``maskstore`` intrinsics with integer masks.
       - Earlier: small case distinctions with 32/64-bit stores and ``std::memcpy``.

     - 128-bit, 8/16-bit elements:

       - Store the lower 64 bits using ``_mm_storeu_si64`` if ``size >= N / 2``, then handle the remaining 64 bits (transferred to a GP register) using ``std::memcpy`` of 4/2/1-byte chunks.

     - 256/512-bit vectors:

       - Split across halves: fully store the lower half if applicable and partially store the remaining half.

   Sub-native vectors
   ------------------

   - x86-64-v4: forward to the corresponding native :cpp:func:`backend::store_part` on the backing register.
   - Earlier: use specialized small-width sequences using ``std::memcpy`` and narrow stores (e.g. ``_mm_storeu_si32``, ``_mm_storeu_si64``).

   Neon
   ====

   Native vectors
   --------------

   - Let ``bytes = size * sizeof(T)``.
     Write 8/4/2/1-byte blocks:

     - Use ``std::memcpy`` for leading bytes (which is translated into ``str`` instructions).
     - Use lane-wise stores (``vst1q_lane_u8`` or ``st1`` via inline assembly) for tail bytes.

   Sub-native vectors
   ------------------

   - Dispatch per element count to the native partial store on the backing register.

   Super-native vectors (shared)
   =============================

   - If ``size <= N / 2``: partial store of the lower half.
   - Otherwise: fully store the lower half and partially store the upper half for the remainder.

.. _operations-store-part-ct:

***********************************
Store Partial (Compile-Time Length)
***********************************

.. cpp:function:: void backend::store_part(T* dst, Vector<T, N> v, AnyIndexTag auto size)

   Stores a compile-time-known number of elements ``size`` into ``dst[0..size-1]`` without writing beyond them.

   - If ``size == N``, equivalent to :cpp:func:`backend::store`.
   - If ``size == 0``, stores nothing.

   Uses the same mechanisms as the runtime overload, but ``size``-dependent branches become ``if constexpr`` or template-based, allowing the compiler to emit size-specialized straight-line code.
