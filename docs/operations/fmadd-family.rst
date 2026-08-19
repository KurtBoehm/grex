.. cpp:namespace:: grex

.. _operations-fmadd-family:

#########################
Fused Multiply-Add Family
#########################

Fused multiply-add and related operations on floating-point vectors and scalars.
Sub-native vectors are processed via their backing native vectors, while each native lane of a super-native vector is processed independently.

Only floating-point element types are supported.
The four members of the family differ only in the signs of the product and the addend, so the backend provides a single :cpp:func:`~backend::fused`, selected by a tag, rather than four separate functions; the high-level :cpp:func:`fmadd`, :cpp:func:`fmsub`, :cpp:func:`fnmadd`, and :cpp:func:`fnmsub` are thin wrappers that pass the matching tag.

.. _operations-fused-tags:

****
Tags
****

.. cpp:struct:: backend::MultiplyAdd

   Selects :math:`a \cdot b + c`.

.. cpp:struct:: backend::MultiplySubtract

   Selects :math:`a \cdot b - c`.

.. cpp:struct:: backend::NegatedMultiplyAdd

   Selects :math:`-(a \cdot b) + c`.

.. cpp:struct:: backend::NegatedMultiplySubtract

   Selects :math:`-(a \cdot b) - c`.

.. cpp:concept:: template<typename T> backend::FusedTag

   Satisfied by exactly these four tags.

.. cpp:var:: constexpr bool backend::has_fma
             constexpr bool backend::has_f16_fma

   Indicate whether the fused operations on binary32/binary64 and on binary16, respectively, are carried out with a single rounding.
   Where they are not, the operation is decomposed — into a multiplication and an addition for binary32/binary64, and into a single binary32 fused multiply-add sandwiched between conversions for binary16, which is off by at most one unit in the last place (see :ref:`f16-accuracy`).

**************
Vector Variant
**************

.. cpp:function:: template<FloatVectorizable T, std::size_t N> \
                  Vector<T, N> backend::fused(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c, FusedTag auto tag)

   Element-wise fused multiply-add with the signs prescribed by ``tag``, i.e. :math:`\pm a_i \cdot b_i \pm c_i` computed with a single rounding where :cpp:var:`~backend::has_fma`/:cpp:var:`~backend::has_f16_fma` says so.

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: the ``fmadd``/``fmsub``/``fnmadd``/``fnmsub`` intrinsics, one per tag.
   - **Earlier**: decomposed per tag into :cpp:func:`~backend::multiply`, :cpp:func:`~backend::add`/:cpp:func:`~backend::subtract`, and :cpp:func:`~backend::negate`; not fused.

   Neon
   ====

   - ``vfmaq`` for :cpp:struct:`~backend::MultiplyAdd` and ``vfmsq``, which computes :math:`c - a \cdot b`, for :cpp:struct:`~backend::NegatedMultiplyAdd`; the two subtracting tags negate ``c`` and delegate to the corresponding adding tag.

.. _operations-fmadd-scalar:

**************
Scalar Variant
**************

.. cpp:function:: template<FloatVectorizable T> \
                  T backend::fused(T a, T b, T c, FusedTag auto tag)

   Scalar counterpart of the above, with the same sign conventions.

   x86-64
   ======

   - **x86-64-v3+ (FMA)**: expand to a SIMD register and use the scalar FMA intrinsics.
   - **Earlier**: computed from scalar multiplication, addition/subtraction, and negation; not fused.

   Neon
   ====

   - The ``__builtin_fma`` family, with ``a`` and ``c`` negated as the tag requires.
