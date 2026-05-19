.. cpp:namespace:: grex

############
Construction
############

Vector and mask construction operations build vectors from scalars, constants, or indices.

***************
Vector Creation
***************

.. _operations-zeros-vector:

Zeros
=====

.. cpp:function:: Vector<T, N> backend::zeros(TypeTag<Vector<T, N>>)

   All elements set to zero.

   - **x86-64**: ``setzero`` intrinsics.
   - **Neon**: ``vdupq_n(0)``.
   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-undefined-vector:

Undefined
=========

.. cpp:function:: Vector<T, N> backend::undefined(TypeTag<Vector<T, N>>)

   Vector with undefined contents, suitable only as a destination.

   - **x86-64**: ``undefined`` intrinsics.
   - **Neon**: backend-specific helper that returns an undefined Neon register.
   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-broadcast-vector:

Broadcast
=========

.. cpp:function:: Vector<T, N> backend::broadcast(T value, TypeTag<Vector<T, N>>)

   Broadcasts a scalar to all lanes.

   - **x86-64**: ``set1`` intrinsics with the appropriate casts.
   - **Neon**: ``vdupq_n``.
   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-set-vector:

Set
===

.. cpp:function:: Vector<T, N> backend::set(TypeTag<Vector<T, N>>, T... values)

   Constructs a vector from per-lane scalar values.

   - **x86-64**:

     - **Native**: ``set`` intrinsics with appropriate casts.
     - **Sub-native, integer**:

       - **Size 2**: transfer :math:`v_0` using a ``movd`` instruction and merge :math:`v_1` using a ``pinsrb``/``pinsrw``/``pinsrd`` instruction, if available (``pinsrw`` on all levels, the others on x86-64-v2+), otherwise using another ``movd`` and an ``_mm_unpacklo`` operation.
       - **Larger sizes**: build lower and upper halves recursively and merge them with an appropriately sized ``_mm_unpacklo`` operation.

     - **Sub-native, floating-point**: ``unpcklps``.

   - **Neon**:

     - **64-bit**: expand scalars with :cpp:func:`~backend::expand_any` and combine with ``vzip1q``.
     - **32-bit**: pack pairs into 64-bit temporaries (``bfi`` on GCC, shift and OR otherwise), then merge/interleave and reinterpret as ``u32``/``f32``.
     - **8/16-bit**: pack into wider temporaries with shifts and ``orr``, then expand and reinterpret.
     - **Sub-native**: use dedicated sub-vector overloads.

   - **Super-native (shared implementation)**: the scalar arguments are split into lower and upper halves, each half is passed to :cpp:func:`~backend::set` on the corresponding half type, and the results are combined.

*************
Mask Creation
*************

.. _operations-zeros-mask:

Zeros
=====

.. cpp:function:: Mask<T, N> backend::zeros(TypeTag<Mask<T, N>>)

   All mask lanes cleared.

   - **x86-64**:

     - **x86-64-v4**: zero-valued compressed mask registers.
     - **Earlier**: broad masks built from ``setzero`` intrinsics.

   - **Neon**: ``vdupq_n(0)`` on the underlying unsigned mask type.
   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-ones-mask:

Ones
====

.. cpp:function:: Mask<T, N> backend::ones(TypeTag<Mask<T, N>>)

   All mask lanes set.

   - **x86-64**:

     - **x86-64-v4**: compressed mask registers filled with all ones.
     - **Earlier**: broad masks created with ``set1`` intrinsics called with :math:`-1` (all bits set).

   - **Neon**: ``vdupq_n(-1)`` on the underlying unsigned mask type.
   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-broadcast-mask:

Broadcast
=========

.. cpp:function:: Mask<T, N> backend::broadcast(bool value, TypeTag<Mask<T, N>>)

   Broadcasts a Boolean to all mask lanes.

   - **x86-64**:

     - **x86-64-v4**: sets or clears all bits in the compressed mask according to ``value``.
     - **Earlier**: creates a broad mask by broadcasting an all-ones or all-zeros integer and using sign/bit casts.

   - **Neon**: form an unsigned all-ones/all-zeros value via ``-value`` and broadcast it with ``vdupq_n``.
   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

.. _operations-set-mask:

Set
===

.. cpp:function:: Mask<T, N> backend::set(TypeTag<Mask<T, N>>, bool... values)

   Constructs a mask from per-lane Booleans.

   - **x86-64 (native and sub-native)**:

     - **x86-64-v4**: builds a compressed mask bitfield through shifting and bitwise OR.
     - **Earlier**: delegates to vector :cpp:func:`set() <Vector\<T, N\> backend::set(TypeTag\<Vector\<T, N\>\>, T... values)>` with cast/negated lanes.

   - **Neon**: build an unsigned integer vector with vector :cpp:func:`set() <Vector\<T, N\> backend::set(TypeTag\<Vector\<T, N\>\>, T... values)>` on the corresponding unsigned type, then negate lanes to obtain all-one (true) or all-zero (false) elements.
   - **Super-native (shared implementation)**: the scalar arguments are split into lower and upper halves, each half is passed to :cpp:func:`set() <Mask\<T, N\> backend::set(TypeTag\<Mask\<T, N\>\>, bool... values)>` on the corresponding half type, and the results are combined.

.. _operations-indices:

Indices
=======

.. cpp:function:: Vector<T, N> backend::indices(TypeTag<Vector<T, N>>)

   Vector of lane indices: ``[0, 1, ..., N-1]``.

   - **Native/super-native**: delegate to :cpp:func:`~backend::set` with values :math:`0, 1, \ldots, N - 1`.
   - **Sub-native**: use the native implementation on the backing register and wrap the result.
