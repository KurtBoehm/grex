.. cpp:namespace:: grex

##############
SIMD Interface
##############

.. doxygenvariable:: grex::max_native_size

.. doxygenvariable:: grex::min_native_size

.. doxygenvariable:: grex::native_sizes

.. doxygenvariable:: grex::register_bits

.. doxygenvariable:: grex::register_bytes

.. doxygenstruct:: grex::Mask
   :members:

.. doxygenstruct:: grex::Vector
   :members:

.. doxygenconcept:: grex::AnyMask
.. doxygenconcept:: grex::SizedMask
.. doxygenconcept:: grex::AnyVector
.. doxygenconcept:: grex::ValuedVector
.. doxygenconcept:: grex::SizedVector
.. doxygenconcept:: grex::IntVector
.. doxygenconcept:: grex::FpVector

.. doxygentypedef:: grex::MaskFor
.. doxygentypedef:: grex::VectorFor

.. doxygenfunction:: andnot(Mask<T, N> a, Mask<T, N> b)
.. doxygenfunction:: abs(Vector<T, N> v)
.. doxygenfunction:: sqrt(Vector<T, N> v)
.. doxygenfunction:: min(Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: max(Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: is_finite(Vector<T, N> v)
.. doxygenfunction:: make_finite(Vector<T, N> v)
.. doxygenfunction:: horizontal_add(Vector<T, N> v)
.. doxygenfunction:: horizontal_min(Vector<T, N> v)
.. doxygenfunction:: horizontal_max(Vector<T, N> v)
.. doxygenfunction:: horizontal_and(Mask<T, N> m)
.. doxygenfunction:: fmadd(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)
.. doxygenfunction:: fmsub(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)
.. doxygenfunction:: fnmadd(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)
.. doxygenfunction:: fnmsub(Vector<T, N> a, Vector<T, N> b, Vector<T, N> c)
.. doxygenfunction:: extract_single(Vector<T, N> v)
.. doxygenfunction:: blend_zero(Mask<T, N> mask, Vector<T, N> v1)
.. doxygenfunction:: blend_zero(Vector<T, N> v1)
.. doxygenfunction:: blend(Mask<T, N> mask, Vector<T, N> v0, Vector<T, N> v1)
.. doxygenfunction:: blend(Vector<T, N> v0, Vector<T, N> v1)
.. doxygenfunction:: shuffle(Vector<T, TableSize> table, Vector<TIdx, IdxSize> idxs)
.. doxygenfunction:: shuffle(Vector<T, N> table)
.. doxygenfunction:: mask_add(Mask<T, N> mask, Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: mask_subtract(Mask<T, N> mask, Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: mask_multiply(Mask<T, N> mask, Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: mask_divide(Mask<T, N> mask, Vector<T, N> a, Vector<T, N> b)
.. doxygenfunction:: gather(std::span<const V, Extent> data, Vector<TIndex, N> indices)
.. doxygenfunction:: mask_gather(std::span<const V, Extent> data, Mask<V, N> mask, Vector<TIndex, N> indices)

.. doxygenstruct:: std::tuple_size< grex::Vector< T, N > >
.. doxygenstruct:: std::tuple_element< I, grex::Vector< T, N > >

.. doxygenstruct:: std::tuple_size< grex::Mask< T, N > >
.. doxygenstruct:: std::tuple_element< I, grex::Mask< T, N > >
