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

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::zeros(TypeTag<Vector<T, N>>)

   All elements set to zero.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - ``setzero`` intrinsics.

   Neon
   ----

   - ``vdupq_n`` intrinsics with value ``0``.

.. _operations-undefined-vector:

Undefined
=========

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::undefined(TypeTag<Vector<T, N>>)

   Vector with undefined contents, suitable only as a destination.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - ``undefined`` intrinsics.

   Neon
   ----

   - Backend-specific helper that returns an undefined Neon register.

.. _operations-broadcast-vector:

Broadcast
=========

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::broadcast(T value, TypeTag<Vector<T, N>>)

   Broadcasts a scalar to all lanes.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing register and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - ``set1`` intrinsics with the appropriate casts.

   Neon
   ----

   - ``vdupq_n`` intrinsics.

.. _operations-set-vector:

Set
===

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::set(TypeTag<Vector<T, N>>, T... values)

   Constructs a vector from per-lane scalar values.

   Shared
   ------

   - **Super-native**: the scalar arguments are split into lower and upper halves, each half is passed to :cpp:func:`~backend::set` on the corresponding half type, and the results are combined.

   x86-64
   ------

   - **Native**: ``set`` intrinsics with appropriate casts.
   - **Sub-native, integer**:

     - **Size 2**: transfer :math:`v_0` using a ``movd`` instruction and merge :math:`v_1` using a ``pinsrb``/``pinsrw``/``pinsrd`` instruction, if available (``pinsrw`` on all levels, the others on x86-64-v2+), otherwise using another ``movd`` and an ``_mm_unpacklo`` operation.
     - **Larger sizes**: build lower and upper halves recursively and merge them with an appropriately sized ``_mm_unpacklo`` operation.

   - **Sub-native, floating-point**: ``unpcklps``.

   Neon
   ----

   - **64-bit entries**: expand scalars with :cpp:func:`~backend::expand_any` to 2-lane vectors and interleave with ``vzip1q`` intrinsics.
   - **32-bit floating point**: expand scalars with :cpp:func:`~backend::expand_any` and interleave with ``vzip1q_f32``; 4-lane vectors are built by zipping two 2-lane temporaries.
   - **32-bit integers**: merge scalar pairs into 64-bit temporaries (``bfi`` or ``std::memcpy``), expand with :cpp:func:`~backend::expand_any`, then reinterpret as ``i32``/``u32`` and combine via 64-bit interleaving.
   - **8/16-bit integers**: merge scalar pairs into wider integer temporaries (``bfi`` or ``std::memcpy``), expand with :cpp:func:`~backend::expand_any`, and recursively combine into 8/16-lane vectors.
   - **Sub-native**: dedicated sub-vector overloads that use the same approach as the native version, but with fewer merging steps.

.. _operations-indices:

Indices
=======

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::indices(TypeTag<Vector<T, N>>)

   Vector of lane indices: ``[0, 1, ..., N-1]``.

   Shared
   ------

   - **Native/super-native**: delegate to :cpp:func:`~backend::set` with values :math:`0, 1, \ldots, N - 1`.
   - **Sub-native**: use the native implementation on the backing register and wrap the result.

*************
Mask Creation
*************

.. _operations-zeros-mask:

Zeros
=====

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::zeros(TypeTag<Mask<T, N>>)

   All mask lanes cleared.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - **x86-64-v4**: zero-valued compressed mask registers.
   - **Earlier**: broad masks built from ``setzero`` intrinsics.

   Neon
   ----

   - ``vdupq_n`` with value ``0`` on the underlying unsigned vector type.

.. _operations-ones-mask:

Ones
====

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::ones(TypeTag<Mask<T, N>>)

   All mask lanes set.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - **x86-64-v4**: compressed mask registers filled with all ones.
   - **Earlier**: broad masks created with ``set1`` intrinsics called with :math:`-1` (all bits set).

   Neon
   ----

   - ``vdupq_n`` intrinsics with value ``-1`` on the underlying unsigned vector type.

.. _operations-broadcast-mask:

Broadcast
=========

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::broadcast(bool value, TypeTag<Mask<T, N>>)

   Broadcasts a Boolean to all mask lanes.

   Shared
   ------

   - **Sub-native**: use the native implementation on the backing mask and wrap the result.
   - **Super-native**: apply the native implementation to each half and combine.

   x86-64
   ------

   - **x86-64-v4**: sets or clears all bits in the compressed mask according to ``value``.
   - **Earlier**: creates a broad mask by broadcasting an all-ones or all-zeros integer and using sign/bit casts.

   Neon
   ----

   - Form an unsigned all-ones/all-zeros value via ``-value`` and broadcast it with ``vdupq_n``.

.. _operations-set-mask:

Set
===

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Mask<T, N> backend::set(TypeTag<Mask<T, N>>, bool... values)

   Constructs a mask from per-lane Booleans.

   Shared
   ------

   - **Super-native**: the scalar arguments are split into lower and upper halves, each half is passed to :cpp:func:`~backend::set` on the corresponding half type, and the results are combined.

   x86-64
   ------

   - **x86-64-v4**: builds a compressed mask bitfield through shifting and bitwise OR.
   - **Earlier**: delegates to vector :cpp:func:`~backend::set` with cast/negated lanes.

   Neon
   ----

   - Build an unsigned integer vector with vector :cpp:func:`~backend::set` on the corresponding unsigned type using Boolean values ``0``/``1``, then apply arithmetic negation so that ``true`` becomes all-one and ``false`` becomes all-zero in each lane.
