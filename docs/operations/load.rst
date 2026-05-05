.. cpp:namespace:: grex

#######
Loading
#######

Vector loading operations read elements from contiguous scalar memory into SIMD vectors.
Sub-native vectors are embedded in a full native vector; super-native vectors are assembled from native halves.
Partial loads handle a prefix without touching memory beyond the requested number of elements.

.. _operations-load:

**************
Load Unaligned
**************

.. cpp:function:: Vector<T, N> backend::load(const T* ptr, TypeTag<Vector<T, N>>)

   Loads :math:`N` contiguous elements from ``ptr`` into a vector.

   The pointer must be valid for scalar ``T`` access and may be unaligned.

   x86-64
   ======

   - **Native**: ``loadu`` intrinsics at the appropriate width.
   - **Sub-native**: load only the required bytes into a 128-bit integer register via narrow loads (e.g. ``_mm_loadu_si8``) and packing, then reinterpret.

   Neon
   ====

   - **Native**: ``vld1q`` intrinsics.
   - **Sub-native**: implemented via :cpp:func:`~backend::load_part` on the corresponding sub-vector type.

   Super-native (shared)
   =====================

   - Assembled from loads of lower and upper halves.

.. _operations-load-aligned:

************
Load Aligned
************

.. cpp:function:: Vector<T, N> backend::load_aligned(const T* ptr, TypeTag<Vector<T, N>>)

   Loads :math:`N` contiguous elements from an address assumed to be aligned for a full SIMD vector of ``T``.

   Behaviour matches :cpp:func:`~backend::load`, but may use alignment-sensitive intrinsics on x86-64.
   On Neon, aligned and unaligned loads are identical.

.. _operations-load-part-runtime:

*****************************
Load Partial (Runtime Length)
*****************************

.. cpp:function:: Vector<T, N> backend::load_part(const T* ptr, std::size_t size, TypeTag<Vector<T, N>>)

   Loads up to ``size`` elements from ``ptr`` into a vector, without reading beyond them.

   - If :math:`\text{size} \ge N`, equivalent to :cpp:func:`~backend::load`.
   - If :math:`\text{size} = 0`, returns unspecified contents.

   Remaining lanes (if any) are unspecified.

   x86-64
   ======

   Native vectors
   --------------

   - **x86-64-v4**: ``maskz_loadu`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
   - **Earlier**:

     - **128-bit, 32/64-bit elements**:

       - **x86-64-v3**: ``_mm_maskload_epi32``/``_mm_maskload_epi64`` with integer masks.
       - **Earlier**: small case distinctions using 32/64-bit ``std::memcpy`` and packing into a 128-bit register.

     - **128-bit, 8/16-bit elements**:

       - Incrementally copy 1/2/4/8-byte pieces into a 64-bit GP register, then use ``_mm_set_epi64x`` to form a 128-bit integer vector and reinterpret.

     - **256/512-bit**:

       - Split across halves: load one full half and one partial half, then merge.

   Sub-native vectors
   ------------------

   - **x86-64-v4**: forward to native :cpp:func:`~backend::load_part` on the backing register.
   - **Earlier**: specialized narrow sequences using ``std::memcpy`` and narrow loads (e.g. ``_mm_loadu_si32``, ``_mm_loadu_si64``).

   Neon
   ====

   Native vectors
   --------------

   - Let :math:`\text{bytes} = \text{size} \cdot \text{sizeof}(T)`.
     Load 8/4/2/1-byte blocks:

     - Use ``ldr`` via inline assembly for the first block into the low 128 bits.
     - Use lane-wise loads (``vld1q_lane_u8`` or ``ld1`` via inline assembly) for tail bytes.

   Sub-native vectors
   ------------------

   - Delegate to the native partial loader and wrap as a sub-vector.

   Super-native (shared)
   =====================

   - If :math:`\text{size} \le N / 2`: partial load into the lower half; upper half undefined.
   - Otherwise: full lower half, partial upper half for the remainder.

.. _operations-load-part-ct:

**********************************
Load Partial (Compile-Time Length)
**********************************

.. cpp:function:: Vector<T, N> backend::load_part(const T* ptr, AnyIndexTag auto size, TypeTag<Vector<T, N>>)

   Loads a compile-time-known number of elements ``size``, without reading beyond them.

   - If :math:`\text{size} = N`, equivalent to :cpp:func:`~backend::load`.
   - If :math:`\text{size} = 0`, returns unspecified contents.

   Uses the same mechanisms as the runtime overload, but ``size``-dependent branches become ``if constexpr`` or template-based, so the compiler can emit size-specialized straight-line code.
