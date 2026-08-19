.. cpp:namespace:: grex

#######
Loading
#######

Vector loading operations read elements from contiguous scalar memory into SIMD vectors.
Partial loads handle a prefix without accessing memory beyond the requested number of elements.
Loading only moves bits, so binary16 goes through the very same code as ``u16`` (see :ref:`f16-implementation`).

.. _operations-load:

**************
Load Unaligned
**************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::load(const T* ptr, TypeTag<Vector<T, N>>)

   Loads :math:`N` contiguous elements from ``ptr`` into a vector.

   The pointer must be valid for scalar ``T`` access and may be unaligned.

   Shared
   ======

   - **Super-native**: split into halves and load independently.

   x86-64
   ======

   - **Native**: ``loadu`` intrinsics at the appropriate width.
   - **Sub-native**: load exactly the bytes the sub-vector holds with the matching narrow load (``_mm_loadu_si16/32/64``), then reinterpret.

   Neon
   ====

   - **Native**: ``vld1q`` intrinsics.
   - **Sub-native**: implemented via :cpp:func:`~backend::load_part` on the backing native vector, then reinterpret.

.. _operations-load-aligned:

************
Load Aligned
************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::load_aligned(const T* ptr, TypeTag<Vector<T, N>>)

   Loads :math:`N` contiguous elements from an address assumed to be suitably aligned for a full SIMD vector of ``T``.
   Semantics match :cpp:func:`~backend::load`.

   Shared
   ======

   - **Super-native**: as for unaligned loads, but using :cpp:func:`~backend::load_aligned` on each half.

   x86-64
   ======

   - Uses aligned ``load`` intrinsics where available; otherwise identical to unaligned loads.

   Neon
   ====

   - Same code path as :cpp:func:`~backend::load` (Neon loads are alignment-agnostic).

.. _operations-load-part-runtime:

*****************************
Load Partial (Runtime Length)
*****************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::load_part(const T* ptr, std::size_t size, TypeTag<Vector<T, N>>)

   Loads up to ``size`` elements from ``ptr`` into a vector, without reading beyond them.

   - If :math:`\mathit{size} \ge N`, equivalent to :cpp:func:`~backend::load`.
   - If :math:`\mathit{size} = 0`, all lanes are left unspecified.

   Lanes beyond ``size`` (if any) are unspecified.

   Shared
   ======

   - **Super-native**:

     - If :math:`\mathit{size} \le N / 2`: partially load the lower half; upper half undefined.
     - If :math:`\mathit{size} > N / 2`: fully load the lower half and partially load the upper half with :math:`\mathit{size} - N / 2` elements.

   x86-64
   ======

   Native vectors
   --------------

   - **x86-64-v4**: ``maskz_loadu`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
   - **x86-64-v3**:

     - **32/64-bit elements (any register width)**: ``maskload`` intrinsics with a mask from :cpp:func:`~backend::cutoff_mask`.
     - **256-bit vectors, 8/16-bit elements**: fully load the lower half, then load the 16 bytes ending at the last requested element and move them down by the bytes the lower half already covers, using a ``pshufb`` row read at a run-time offset from a 32-byte index table. If the requested size does not reach the upper half, the lower half is loaded partially and the upper half is left undefined.
     - **128-bit vectors, 8/16-bit elements**: the byte-wise prefix load described below.

   - **x86-64-v2** and **x86-64-v1**:

     - **128-bit vectors, 2 elements**: switch statement over the size, as every case is a single narrow load.
     - **128-bit vectors otherwise**: the byte-wise prefix load described below.
     - **Wider vectors**: super-native, hence split into halves.

   The element size and count of the byte-wise prefix load are compile-time constants, which prunes the cases the caller cannot reach; the sequence itself is the same on x86-64-v1 and x86-64-v2 and is also what x86-64-v3 uses for 128-bit vectors of 8/16-bit elements:

   - Two overlapping loads of the largest power-of-two block that fits into the requested byte count cover it entirely, since the second one starts at the last block boundary below the end.
     The bytes the first load does not already provide are the top ones of the second, so it is shifted down by ``_mm_srl_epi64`` and interleaved above the first with an ``unpack``.
     A shift by 64 bits or more yields zero, so a byte count equal to the block size needs no special case.
   - The block size is halved until it fits, down to a single byte, which is loaded on its own; an empty load yields zeros.

   Sub-native vectors
   ------------------

   - **x86-64-v4**: delegate to the corresponding native :cpp:func:`~backend::load_part` and wrap.
   - **Earlier**: the same two paths as native 128-bit vectors, with the byte count bounded by the sub-vector rather than the register.

   Neon
   ====

   Native vectors
   --------------

   - Let :math:`\mathit{bytes} = \mathit{size} \cdot \operatorname{sizeof}(T)`.
   - Decompose into 8/4/2/1-byte blocks:

     - First non-zero block size uses a dedicated ``load_first`` helper:

       - **8/16/32-bit total**: inline-assembly ``ldr`` into a Neon register, possibly preceded by a compile-time constant check that collapses to ``memset``/``memcpy``.
       - **64-bit total**: scalar 64-bit load via ``std::memcpy``, then widening and reinterpretation.

     - Remaining tail bytes are filled with lane-wise loads:

       - **32/16-bit lanes**: ``ld1`` inline assembly with element indices.
       - **Single-byte tails**: ``vld1q_lane_u8`` at the appropriate byte offset.

   Sub-native vectors
   ------------------

   - Delegate to the native 128-bit partial loader and then wrap as a sub-native vector.

.. _operations-load-part-ct:

**********************************
Load Partial (Compile-Time Length)
**********************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::load_part(const T* ptr, AnyIndexTag auto size, TypeTag<Vector<T, N>>)

   Loads a compile-time-known number of elements ``size`` without reading beyond them.

   - If :math:`\mathit{size} = N`, equivalent to :cpp:func:`~backend::load`.
   - If :math:`\mathit{size} = 0`, all lanes are left unspecified.

   Backend behaviour matches the run-time :cpp:func:`~backend::load_part` overload, but:

   - Branches on ``size`` become ``if constexpr`` or template dispatch.
   - On x86-64, this lets the compiler select a single specialized shuffle/mask path.
   - On Neon, the chosen ``load_first`` + lane-insert pattern is fully constant-folded.

   Shared
   ======

   - **Super-native**: the same half-splitting strategy as the run-time overload, with ``size`` tested at compile time.
