.. cpp:namespace:: grex

#########
Expansion
#########

Expansion operations build SIMD vectors from scalars or smaller vectors by placing existing elements into the lowest lanes and leaving the upper lanes either zeroed or with unspecified contents.

****************
Scalar Expansion
****************

.. _operations-expand-scalar-any:

Expand (Any)
============

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::expand_any(Scalar<T> x, IndexTag<N>)

   Expands a scalar ``x`` into a vector of size ``N`` by placing it in the lowest lane; the remaining lanes have unspecified contents.

   x86-64
   ------

   - **Floating point**:

     - **GCC**: reinterprets the scalar as an ``__m128``/``__m128d`` register via inline assembly, leaving the upper lanes unchanged (logically unspecified).
     - **Clang**: writes the scalar to a temporary array and loads it with ``_mm_load_ps``/``_mm_load_pd``; only the lowest lane is initialized, the others are unspecified.

   - **Integers**:

     - **8/16/32-bit**: zero-extended to 32 bits and inserted into the low 32 bits of a 128-bit integer register with ``_mm_cvtsi32_si128``.
       The remaining bits of the register are zero.
     - **64-bit**: cast to a signed 64-bit integer and inserted into the low 64 bits with ``_mm_cvtsi64_si128``.
       The remaining bits of the register are zero.

   Neon
   ----

   - **Floating point**:

     - **GCC**: inline assembly with ``"=w"`` moves the scalar into a Neon register, leaving upper lanes unchanged (logically unspecified).
     - **Clang**: stores the scalar to a one-element array and loads it with ``vld1q_f32``/``vld1q_f64``; only the lowest lane is initialized, the others are unspecified.

   - **Integers**:

     - **32/64-bit**: bit-cast the integer to the corresponding floating type, expand via the floating-point path, then reinterpret as an integer vector.
     - **8/16-bit**: widen to 32 bits (ideally without emitting instructions), then follow the 32-bit path.

   Shared
   ------

   - **Larger sizes**: recursively expand to half the requested size, then expand that result to ``N`` (see :ref:`vector expansion <operations-expand-vector-any>`).
   - **Sub-native sizes**: expand to the smallest native size and wrap the result into a sub-vector.

.. _operations-expand-scalar-zero:

Expand (Zeros)
==============

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::expand_zero(Scalar<T> x, IndexTag<N>)

   Expands a scalar ``x`` into a vector of size ``N``, placing it in the lowest lane and zero-filling the remaining lanes.

   x86-64
   ------

   - **Floating point**: uses ``_mm_set_ss``/``_mm_set_sd`` to place ``x`` in the lowest lane and set all higher lanes to zero.
   - **Integers**: uses the same scalar injection as :cpp:func:`~backend::expand_any` (``_mm_cvtsi32_si128`` / ``_mm_cvtsi64_si128``).
     These intrinsics are zero-extending, so all bits outside the inserted scalar are guaranteed to be zero.

   Neon
   ----

   - **All types**: start from a zero vector and insert ``x`` into the lowest lane.

   Shared
   ------

   - **Larger sizes**: recursively expand to half the requested size, then construct a super-vector whose upper half is produced by :cpp:func:`~backend::zeros`.
   - **Sub-native sizes**: expand to the smallest native size (with zeroed upper lanes) and wrap the result into a sub-vector.

****************
Vector Expansion
****************

Vector expansion grows an existing vector (native, sub-native, or super-native) to a larger size by placing it in the low lanes and filling the upper lanes with zeros or unspecified contents.

.. _operations-expand-vector-any:

Expand (Any)
============

.. cpp:function:: template<AnyVector V, std::size_t N> \
                  Vector<typename V::Value, N> backend::expand_any(V v, IndexTag<N>)

   Expands a vector ``v`` to size ``N``, placing its elements in the lowest lanes; the remaining lanes have unspecified contents.

   x86-64
   ------

   - **Native-width to wider native-width**: uses ``cast`` intrinsics.
   - **Native/super-native to super-native**: recursively expand ``v`` to half of ``N`` and merge it with an :cpp:func:`~backend::undefined` vector of the same half size for the upper part.

   Neon
   ----

   - **Native to super-native**: recursively expand ``v`` to half of ``N`` and build a super-vector whose lower half is the expanded ``v`` and whose upper half is :cpp:func:`~backend::undefined`.

   Shared
   ------

   - **Unchanged size**: returns ``v`` unchanged.
   - **Sub-native to sub-native/native**: cut off to sub-native/smallest native, then expand recursively if necessary.

.. _operations-expand-vector-zero:

Expand (Zeros)
==============

.. cpp:function:: template<AnyVector V, std::size_t N> \
                  Vector<typename V::Value, N> backend::expand_zero(V v, IndexTag<N>)

   Expands a vector ``v`` to size ``N`` by placing it in the lower lanes and zero-filling the remaining lanes.

   x86-64
   ------

   - **Native-width to wider native-width**: uses ``zext`` intrinsics.
   - **Native/super-native to super-native**: recursively expand ``v`` to half of ``N`` and merge it with a :cpp:func:`~backend::zeros` vector of the same half size for the upper part.

   Neon
   ----

   - **Native to super-native**: recursively expand ``v`` to half of ``N`` and build a super-vector whose lower half is the expanded ``v`` and whose upper half is :cpp:func:`~backend::zeros`.

   Shared
   ------

   - **Unchanged size**: returns ``v`` unchanged.
   - **Sub-native to sub-native/native**: cut off to sub-native/smallest native and expand recursively if necessary.
