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

   - **x86-64**: ``setzero`` intrinsics.
   - **Neon**: ``vdupq_n(0)``.

.. _operations-undefined-vector:

*********
Undefined
*********

.. cpp:function:: Vector<T, N> backend::undefined(TypeTag<Vector<T, N>>)

   Vector with undefined contents, suitable only as a destination.

   - **x86-64**: ``undefined`` intrinsics.
   - **Neon**: inline assembly to obtain an undefined Neon register.

.. _operations-broadcast-vector:

*********
Broadcast
*********

.. cpp:function:: Vector<T, N> backend::broadcast(T value, TypeTag<Vector<T, N>>)

   Broadcasts a scalar to all lanes.

   - **x86-64**: ``set1`` intrinsics with casts.
   - **Neon**: ``vdupq_n``.

.. _operations-set-vector:

***
Set
***

.. cpp:function:: Vector<T, N> backend::set(TypeTag<Vector<T, N>>, T... values)

   Constructs a vector from per-lane scalar values.

   - **x86-64**: ``set`` intrinsics with casts.
   - **Neon**:

     - **64-bit**: expand scalars and combine with ``vzip1q``.
     - **32-bit**: pack pairs into 64-bit temporaries (``bfi`` on GCC, shift and OR otherwise), then merge/interleave and reinterpret as ``u32``/``f32``.
     - **8/16-bit**: pack into wider temporaries with shifts and ``orr``, then expand and reinterpret.
     - **Sub-native**: specialized simplified sequences.

.. _operations-zeros-mask:

*****
Zeros
*****

.. cpp:function:: Mask<T, N> backend::zeros(TypeTag<Mask<T, N>>)

   All mask lanes cleared.

   - **x86-64**: zero-valued mask registers (x86-64-v4) or ``setzero`` (earlier).
   - **Neon**: ``vdupq_n(0)`` on the unsigned mask type.

.. _operations-ones-mask:

****
Ones
****

.. cpp:function:: Mask<T, N> backend::ones(TypeTag<Mask<T, N>>)

   All mask lanes set.

   - **x86-64**: fill compressed or broad masks with all-ones.
   - **Neon**: ``vdupq_n(-1)`` on the unsigned mask type.

.. _operations-broadcast-mask:

*********
Broadcast
*********

.. cpp:function:: Mask<T, N> backend::broadcast(bool value, TypeTag<Mask<T, N>>)

   Broadcasts a Boolean to all mask lanes.

   - **x86-64**: sets or clears all bits according to ``value``.
   - **Neon**: form all-ones/all-zeros unsigned value via negation and broadcast with ``vdupq_n``.

.. _operations-set-mask:

***
Set
***

.. cpp:function:: Mask<T, N> backend::set(TypeTag<Mask<T, N>>, bool... values)

   Constructs a mask from per-lane Booleans.

   - **x86-64**:

     - **x86-64-v4**: builds a compressed mask bitfield.
     - **Earlier**: broad mask via ``set1`` and sign/bit casts.

   - **Neon**: build an unsigned integer vector with :cpp:func:`~backend::set`, then negate lanes to all-ones/all-zeros.

.. _operations-indices:

*******
Indices
*******

.. cpp:function:: Vector<T, N> backend::indices(TypeTag<Vector<T, N>>)

   Vector of lane indices: ``[0, 1, ..., N-1]``.

   - Implemented backend-agnostically via :cpp:func:`~backend::set`.
