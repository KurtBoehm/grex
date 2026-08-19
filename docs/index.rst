##################
Grex Documentation
##################

Grex is a header-only C++23 library for explicit SIMD programming.
On x86-64 and AArch64, it provides ``Vector<T, N>`` and ``Mask<T, N>`` for any :cpp:concept:`grex::Vectorizable` type ``T`` and any power-of-two lane count ``N``, whether or not the target has a register of that size, and implements every operation on them with the best instruction sequence the selected x86-64 level or Arm Neon feature set offers, falling back to a portable emulation where it offers none.

The same operations exist for single values, and those that need to be told how wide to be take an execution tag, so a kernel can be written once and instantiated for scalars, for full vectors, or for the partially filled remainder of a loop.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   simd
   operations
   f16
   backend
   expensive-operations
