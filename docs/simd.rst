################
 SIMD Interface
################

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

.. doxygenfunction:: andnot(Mask<T, tSize> a, Mask<T, tSize> b)
.. doxygenfunction:: abs(Vector<T, tSize> v)
.. doxygenfunction:: sqrt(Vector<T, tSize> v)
.. doxygenfunction:: min(Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: max(Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: is_finite(Vector<T, tSize> v)
.. doxygenfunction:: make_finite(Vector<T, tSize> v)
.. doxygenfunction:: horizontal_add(Vector<T, tSize> v)
.. doxygenfunction:: horizontal_min(Vector<T, tSize> v)
.. doxygenfunction:: horizontal_max(Vector<T, tSize> v)
.. doxygenfunction:: horizontal_and(Mask<T, tSize> m)
.. doxygenfunction:: fmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c)
.. doxygenfunction:: fmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c)
.. doxygenfunction:: fnmadd(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c)
.. doxygenfunction:: fnmsub(Vector<T, tSize> a, Vector<T, tSize> b, Vector<T, tSize> c)
.. doxygenfunction:: extract_single(Vector<T, tSize> v)
.. doxygenfunction:: blend_zero(Mask<T, tSize> mask, Vector<T, tSize> v1)
.. doxygenfunction:: blend_zero(Vector<T, tSize> v1)
.. doxygenfunction:: blend(Mask<T, tSize> mask, Vector<T, tSize> v0, Vector<T, tSize> v1)
.. doxygenfunction:: blend(Vector<T, tSize> v0, Vector<T, tSize> v1)
.. doxygenfunction:: shuffle(Vector<T, tTableSize> table, Vector<TIdx, tIdxSize> idxs)
.. doxygenfunction:: shuffle(Vector<T, tSize> table)
.. doxygenfunction:: mask_add(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: mask_subtract(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: mask_multiply(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: mask_divide(Mask<T, tSize> mask, Vector<T, tSize> a, Vector<T, tSize> b)
.. doxygenfunction:: gather(std::span<const TValue, tExtent> data, Vector<TIndex, tSize> indices)
.. doxygenfunction:: mask_gather(std::span<const TValue, tExtent> data, Mask<TValue, tSize> mask, Vector<TIndex, tSize> indices)

.. doxygenstruct:: std::tuple_size< grex::Vector< T, tSize > >
.. doxygenstruct:: std::tuple_element< tIdx, grex::Vector< T, tSize > >

.. doxygenstruct:: std::tuple_size< grex::Mask< T, tSize > >
.. doxygenstruct:: std::tuple_element< tIdx, grex::Mask< T, tSize > >

.. toctree::
   :maxdepth: 2
   :caption: Contents:
