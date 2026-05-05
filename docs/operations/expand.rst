.. cpp:namespace:: grex

#########
Expansion
#########

Expansion operations build SIMD vectors from scalars or smaller vectors by placing existing elements in the lowest lanes, with upper lanes zeroed or unspecified.

****************
Scalar Expansion
****************

.. _operations-expand-scalar-any:

Expand (Any)
============

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::expand_any(Scalar<T> x, IndexTag<N>)

   Expands a scalar ``x`` to size ``N`` by writing it to lane 0; remaining lanes have unspecified contents.

   x86-64
   ------

   - **Floating point**:

     - **GCC**: inline assembly to reinterpret as ``__m128``/``__m128d``; upper lanes logically unspecified.
     - **Clang**: store to a one-element array and load with ``_mm_load_ps``/``_mm_load_pd``; only lane 0 is initialized.

   - **Integers**:

     - **8/16/32-bit**: zero-extend to 32 bits and insert into the low 32 bits with ``_mm_cvtsi32_si128``; remaining bits are zero.
     - **64-bit**: cast to signed 64-bit and insert into the low 64 bits with ``_mm_cvtsi64_si128``; remaining bits are zero.

   Neon
   ----

   - **Floating point**:

     - **GCC**: inline assembly ``"=w"`` to move into a Neon register; upper lanes unspecified.
     - **Clang**: store to a one-element array and load with ``vld1q_f32``/``vld1q_f64``; only lane 0 is initialized.

   - **Integers**:

     - **32/64-bit**: bit-cast to a floating-point type, expand via the floating-point path, then reinterpret.
     - **8/16-bit**: widen to 32 bits (ideally no code generated), then follow the 32-bit path.

   Shared
   ------

   - **Larger sizes**: recursively expand to half size, then expand that result to ``N`` (see :ref:`operations-expand-vector-any`).
   - **Sub-native**: expand to the smallest native size and wrap as a sub-vector.

.. _operations-expand-scalar-zero:

Expand (Zeros)
==============

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::expand_zero(Scalar<T> x, IndexTag<N>)

   Expands scalar ``x`` to size ``N``, writing it to lane 0 and zero-filling other lanes.

   x86-64
   ------

   - **Floating point**: ``_mm_set_ss``/``_mm_set_sd``.
   - **Integers**: same scalar injection as :cpp:func:`~backend::expand_any` (zero-extending), yielding zeros in all other bits.

   Neon
   ----

   - **All types**: start from a zero vector and insert ``x`` into lane 0.

   Shared
   ------

   - **Larger sizes**: recursively expand to half size, then build a super-vector whose upper half is :cpp:func:`~backend::zeros`.
   - **Sub-native**: expand to the smallest native size (upper lanes zero) and wrap as a sub-vector.

****************
Vector Expansion
****************

Vector expansion grows an existing vector (native, sub-native, or super-native) to a larger size by placing it in the low lanes and filling upper lanes with zeros or unspecified contents.

.. _operations-expand-vector-any:

Expand (Any)
============

.. cpp:function:: template<AnyVector V, std::size_t N> \
                  Vector<typename V::Value, N> backend::expand_any(V v, IndexTag<N>)

   Expands vector ``v`` to size ``N`` by placing its elements in the lowest lanes; remaining lanes have unspecified contents.

   x86-64
   ------

   - **Native → wider native**: uses ``cast`` intrinsics.
   - **Native/super-native → super-native**: recursively expand ``v`` to :math:`N / 2` and combine with an :cpp:func:`~backend::undefined` upper half.

   Neon
   ----

   - **Native → super-native**: recursively expand to :math:`N / 2` and build a super-vector with an :cpp:func:`~backend::undefined` upper half.

   Shared
   ------

   - **Unchanged size**: returns ``v``.
   - **Sub-native → sub-native/native**: truncate to sub-native/smallest native, then expand recursively if needed.

.. _operations-expand-vector-zero:

Expand (Zeros)
==============

.. cpp:function:: template<AnyVector V, std::size_t N> \
                  Vector<typename V::Value, N> backend::expand_zero(V v, IndexTag<N>)

   Expands vector ``v`` to size ``N`` by placing it in the lower lanes and zero-filling remaining lanes.

   x86-64
   ------

   - **Native → wider native**: ``zext`` intrinsics.
   - **Native/super-native → super-native**: recursively expand to :math:`N / 2` and combine with a :cpp:func:`~backend::zeros` upper half.

   Neon
   ----

   - **Native → super-native**: recursively expand to :math:`N / 2` and combine with a :cpp:func:`~backend::zeros` upper half.

   Shared
   ------

   - **Unchanged size**: returns ``v``.
   - **Sub-native → sub-native/native**: truncate to sub-native/smallest native and expand recursively if necessary.
