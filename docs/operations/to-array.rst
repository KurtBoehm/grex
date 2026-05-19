.. cpp:namespace:: grex

###################
Conversion to Array
###################

Conversion of vectors and masks to contiguous scalar or :cpp:type:`std::array` storage.

.. _operations-to-array-vector-ptr:

*********************
Vector → Scalar Array
*********************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  void backend::to_array(T* dst, Vector<T, N> v)

   Stores all :math:`N` elements of ``v`` into ``dst[0..N-1]``.

   - Implemented as a thin wrapper around :cpp:func:`~backend::store`.
   - The pointer may be unaligned but must be valid for writing :math:`N` scalars of type ``T``.

.. _operations-to-array-mask-ptr:

********************
Mask → Boolean Array
********************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  void backend::to_array(bool* dst, Mask<T, N> m)

   Stores the Boolean value of each mask lane into ``dst[0..N-1]`` as bytes with value ``0`` or ``1``.

   Shared
   ======

   - **Non-byte masks**:

     - Convert ``m`` to a ``u8`` mask with :cpp:func:`convert <template\<AnyMask MSrc, typename Dst\> Mask\<Dst, MSrc::size\> backend::convert(MSrc m, TypeTag\<Dst\>)>`.
     - The resulting ``u8`` mask is then handled as below.

   - **Super-native byte masks**:

     - Process halves independently:

       - Lower half → ``dst[0..N/2-1]``.
       - Upper half → ``dst[N/2..N-1]``.

   - **Native/sub-native byte masks**:

     - Store using backend-specific overloads that materialize one byte per lane with value ``0`` or ``1``.
     - Other combinations are ill-formed and rejected at compile time.

   x86-64
   ======

   - **x86-64-v4 (compressed masks)**:

     - Use ``maskz_mov_epi8`` intrinsics with an all-ones ``u8`` vector to expand the compressed mask into bytes whose low bit equals the corresponding mask bit and whose higher bits are zero.
     - Store the resulting bytes with unaligned integer stores (native) or narrow unaligned stores (8/4/2 bytes, sub-native).

   - **Earlier (broad masks)**:

     - Isolate the low bit in each byte via a bitwise AND with an all-ones ``u8`` vector.
     - Store the resulting bytes with unaligned integer stores (native) or narrow unaligned stores (8/4/2 bytes, sub-native).

   In all x86-64 cases, each output byte contains the mask value in its low bit and zeros elsewhere, so the observable values are exactly ``0`` or ``1``.

   Neon
   ====

   - Isolate the low bit in each byte via a bitwise AND with an all-ones ``u8`` vector, using only the low 64 bits if applicable.
   - Store the appropriate bytes via ``vst1q_u8`` (16 bytes), ``vst1_u8`` (8 bytes), or by delegating to :cpp:func:`~backend::store` with a appropriately-sized sub-native argument (4/2 bytes).

.. _operations-to-array-vector-std-array:

***********************
Vector → ``std::array``
***********************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  std::array<T, N> backend::to_array(Vector<T, N> v)

   Returns a :cpp:type:`std::array` containing all :math:`N` lanes of ``v``.

   - Allocates a ``std::array<T, N>`` and fills it via the pointer-based :cpp:func:`to_array() <template\<Vectorizable T, std::size_t N\> void backend::to_array(T* dst, Vector\<T, N\> v)>`.

.. _operations-to-array-mask-std-array:

***************************
Mask → ``std::array<bool>``
***************************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  std::array<bool, N> backend::to_array(Mask<T, N> m)

   Returns a ``std::array<bool, N>`` holding the Boolean value of each mask lane.

   - Allocates a ``std::array<bool, N>`` and fills it via the pointer-based :cpp:func:`to_array() <template\<Vectorizable T, std::size_t N\> void backend::to_array(bool* dst, Mask\<T, N\> m)>`.
   - Each entry is the Boolean interpretation of the corresponding mask lane as defined above.
