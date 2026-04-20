.. cpp:namespace:: grex

############
Construction
############

Vector and mask construction operations build vectors from scalars, constants, or indices.
Sub-native vectors are embedded in a full native vector; super-native vectors are assembled from native halves.

***************
Vector Creation
***************

.. _operations-zeros-vector:

*****
Zeros
*****

.. cpp:function:: Vector<T, N> backend::zeros(TypeTag<Vector<T, N>>)

   All elements set to zero.

   - x86-64: uses ``setzero`` intrinsics.
   - Neon: uses ``vdupq_n(0)`` intrinsics.

.. _operations-undefined-vector:

*********
Undefined
*********

.. cpp:function:: Vector<T, N> backend::undefined(TypeTag<Vector<T, N>>)

   Returns a vector with undefined contents, suitable only as a destination for later writes.

   - x86-64: uses ``undefined`` intrinsics.
   - Neon: uses inline assembly to obtain an undefined Neon register.

.. _operations-broadcast-vector:

*********
Broadcast
*********

.. cpp:function:: Vector<T, N> backend::broadcast(T value, TypeTag<Vector<T, N>>)

   Broadcasts a single scalar to all lanes.

   - x86-64: uses ``set1`` intrinsics with appropriate casts.
   - Neon: uses ``vdupq_n`` intrinsics.

.. _operations-set-vector:

***
Set
***

.. cpp:function:: Vector<T, N> backend::set(TypeTag<Vector<T, N>>, T... values)

   Constructs a vector from per-lane scalar values.

   - x86-64: uses ``set`` intrinsics with appropriate casts.
   - Neon:

     - 64-bit: expand scalars and combine lanes with ``vzip1q``.
     - 32-bit: pack pairs into 64-bit intermediates (``bfi`` on GCC, shift+OR otherwise), then merge/interleave and reinterpret as ``u32``/``f32`` as needed.
     - 16-bit/8-bit: pack into wider integer temporaries with shifts and ``orr``, then expand and reinterpret to the target element type.
     - Sub-native vectors: use simplified specialized sequences.

.. _operations-zeros-mask:

*****
Zeros
*****

.. cpp:function:: Mask<T, N> backend::zeros(TypeTag<Mask<T, N>>)

   All mask lanes cleared.

   - x86-64: uses zero-valued mask registers (x86-64-v4) or ``setzero`` (earlier).
   - Neon: uses ``vdupq_n(0)`` intrinsics with the unsigned integer mask element type.

.. _operations-ones-mask:

****
Ones
****

.. cpp:function:: Mask<T, N> backend::ones(TypeTag<Mask<T, N>>)

   All mask lanes set.

   - x86-64: fills compressed or broad masks with all-ones.
   - Neon: uses ``vdupq_n`` intrinsics with an all-ones unsigned value (``-1``).

.. _operations-broadcast-mask:

*********
Broadcast
*********

.. cpp:function:: Mask<T, N> backend::broadcast(bool value, TypeTag<Mask<T, N>>)

   Broadcasts a single Boolean to all mask lanes.

   - x86-64: sets or clears all bits according to ``value`` (compressed or broad masks).
   - Neon: forms an all-ones or all-zeros unsigned lane value via negation and broadcasts with ``vdupq_n``.

.. _operations-set-mask:

***
Set
***

.. cpp:function:: Mask<T, N> backend::set(TypeTag<Mask<T, N>>, bool... values)

   Constructs a mask from per-lane Boolean values.

   - x86-64:

     - x86-64-v4: builds a compressed mask bitfield.
     - Earlier: forms a broad mask via ``set1`` and sign/bit casts.

   - Neon: builds an unsigned integer vector with :cpp:func:`backend::set`, then negates lanes to obtain all-ones/all-zeros mask lanes.
