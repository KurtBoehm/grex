.. cpp:namespace:: grex

#######
Loading
#######

Vector loading operations read elements from contiguous scalar memory into SIMD vectors.
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
   - **Sub-native**: implemented via :cpp:func:`~backend::load_part` on the corresponding sub-native vector type.

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

   Lanes beyond ``size`` (if any) are unspecified.

   x86-64
   ======

   Native vectors
   --------------

   - **x86-64-v4**: ``maskz_loadu`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
   - **x86-64-v3**:

     - **32/64-bit elements (any register width)**: ``maskload`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
     - **128-bit vectors, 8/16-bit elements**: use overlapping scalar-sized loads (8/4/2 bytes), pack into a 128-bit register, then shuffle with ``pshufb`` using precomputed tables.
     - **256-bit vectors, 8/16-bit elements**: two 128-bit partial loads and :cpp:func:`~backend::merge`.
     - **512-bit vectors (no AVX-512BW load)**: recursively split into halves: full/partial lower half plus partial/zero upper half.

   - **x86-64-v2**:

     - **128-bit vectors, 32/16/8-bit elements**: overlapping 64/32/16-bit loads into a 128-bit integer register and ``pshufb`` with precomputed shuffle masks.
     - **Wider vectors**: built from 128-bit partial loads via splitting/merging as above.

   - **x86-64-v1**:

     - **128-bit vectors**: accumulate 8/16/32 bytes into one or two 64-bit temporaries via ``std::memcpy``, then assemble with ``_mm_set_epi64x``.
     - **Wider vectors**: split into halves and combine full/partial/zero halves.

   Sub-native vectors
   ------------------

   - **x86-64-v4**: delegate to the corresponding native vector :cpp:func:`~backend::load_part`.
   - **Earlier**: use narrow scalar loads (``_mm_loadu_si8/16/32/64`` equivalents) plus small, size-specialized paths per sub-vector shape (2/4/8 lanes), reusing the same ``pshufb``/``memcpy`` strategies as native vectors.

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

   - Delegate to the native partial loader and wrap as a sub-native vector.

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
