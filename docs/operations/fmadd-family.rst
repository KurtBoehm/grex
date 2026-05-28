.. cpp:namespace:: grex

.. _operations-fmadd-family:

#########################
Fused Multiply–Add Family
#########################

Fused multiply–add and related operations on floating-point vectors and scalars.
Sub-native vectors are processed by applying the given operation to their underlying native vectors, while each native lane of a super-native vector is processed independently.

Only floating-point element types are supported.

.. _operations-fmadd:

******************
Fused Multiply–Add
******************

.. cpp:function:: Vector<T, N> backend::fmadd(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)

   Element-wise fused multiply–add :math:`a_i \cdot b_i + c_i`.

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: ``fmadd`` intrinsics.
   - **Earlier**: decomposed into :cpp:func:`~backend::multiply` and :cpp:func:`~backend::add`; not fused.

   Neon
   ====

   - ``vfmaq`` intrinsics.

.. _operations-fmsub:

***********************
Fused Multiply–Subtract
***********************

.. cpp:function:: Vector<T, N> backend::fmsub(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)

   Element-wise fused multiply–subtract :math:`a_i \cdot b_i - c_i`.

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: ``fmsub`` intrinsics.
   - **Earlier**: decomposed into :cpp:func:`~backend::multiply` and :cpp:func:`~backend::subtract`; not fused.

   Neon
   ====

   - Implemented as :cpp:func:`~backend::fmadd` with negated addend: :math:`a \cdot b + (-c)`.

.. _operations-fnmadd:

**************************
Fused Negated Multiply–Add
**************************

.. cpp:function:: Vector<T, N> backend::fnmadd(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)

   Element-wise fused negated multiply–add :math:`-a_i \cdot b_i + c_i` (equivalently :math:`c_i - a_i \cdot b_i`).

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: ``fnmadd`` intrinsics.
   - **Earlier**: decomposed into :cpp:func:`~backend::multiply` and :cpp:func:`~backend::subtract`; not fused.

   Neon
   ====

   - ``vfmsq`` intrinsics: :math:`c - a \cdot b`.

.. _operations-fnmsub:

*******************************
Fused Negated Multiply–Subtract
*******************************

.. cpp:function:: Vector<T, N> backend::fnmsub(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)

   Element-wise fused negated multiply–subtract :math:`-a_i \cdot b_i - c_i`.

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: ``fnmsub`` intrinsics.
   - **Earlier**: decomposed into :cpp:func:`~backend::multiply`, :cpp:func:`~backend::add`, and :cpp:func:`~backend::negate`; not fused.

   Neon
   ====

   - Implemented as negation of :cpp:func:`~backend::fmadd`: :math:`-(a \cdot b + c)`.

***************
Scalar Variants
***************

.. cpp:function:: T backend::fmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c)

.. cpp:function:: T backend::fmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c)

.. cpp:function:: T backend::fnmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c)

.. cpp:function:: T backend::fnmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c)

   Scalar counterparts of the fused multiply–add family.

   - **Neon/x86-64-v3+**: implemented via ``std::fma`` on the underlying scalar values (Neon) or by expanding to a SIMD register and using scalar FMA intrinsics (x86-64).
   - **Earlier x86-64**: computed from scalar multiply/add/subtract/negation; not fused.
