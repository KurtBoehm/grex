.. cpp:namespace:: grex

#####################
Horizontal Reductions
#####################

Element-wise reductions over all lanes of a vector or mask, producing a scalar result.
Sub-native vectors/masks are reduced over their active lanes via the backing native register.
Super-native vectors/masks are reduced by combining lower and upper halves and reducing the result.

.. _operations-horizontal-add:

*******************
Horizontal Addition
*******************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::horizontal_add(Vector<T, N> v)

   Sum of all lanes:

   .. math::

      \sum_{i = 0}^{N-1} v_i.

   x86-64
   ======

   - **128-bit**:

     - **8-bit integers**: ``sad_epu8`` plus masking via :cpp:func:`~backend::blend_zero()` with compile-time selectors.
     - **Otherwise**: shuffle/add trees using ``add`` intrinsics.

   - **256/512-bit**: split into halves, add the halves element-wise, and recurse on the half-width vector.

   Neon
   ====

   - **Native**: ``vaddvq`` reduction intrinsics for all element types.
   - **Sub-native**:

     - **64 bits active**: extract the low 64 bits and use ``vaddv`` intrinsics.
     - **Otherwise**: one or two ``vpadd`` pairwise additions and a final ``vget_lane``.

   Shared
   ======

   - **Super-native**: compute :cpp:func:`~backend::add` of lower and upper halves and recursively reduce that sum.

.. _operations-horizontal-and:

**************
Horizontal AND
**************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  bool backend::horizontal_and(Mask<T, N> m)

   Logical AND of all mask lanes; returns ``true`` iff every lane is set:

   .. math::

      \bigwedge_{i = 0}^{N-1} m_i.

   x86-64
   ======

   - **Native masks**:

     - **x86-64-v4 (compressed)**: compare the mask register to an all-ones value (``__mmask`` of appropriate width).
     - **Earlier (broad)**: ``movemask_epi8`` on the underlying integer mask and compare to an all-ones bit pattern.

   - **Sub-native masks**:

     - **x86-64-v4**: mask-width compare after masking out inactive bits.
     - **Earlier**: ``_mm_movemask_epi8`` on the full backing register and mask off inactive bytes.

   Neon
   ====

   - **Native**: ``vminvq`` (or 64-bit variant via reinterpretation) on the underlying unsigned mask; test non-zero.
   - **Sub-native**: extract the relevant low 64 bits, reinterpret to a scalar integer ``U``, and compare to ``std::numeric_limits<U>::max()``.

   Shared
   ======

   - **Super-native**: recurse on lower and upper halves and ``&&`` the results.

.. _operations-horizontal-minmax:

********************
Horizontal Min / Max
********************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::horizontal_min(Vector<T, N> v)

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  T backend::horizontal_max(Vector<T, N> v)

   Minimum/maximum over all lanes:

   .. math::

      \text{horizontal\_min}(v) = \min_{0 \le i < N} v_i, \qquad
      \text{horizontal\_max}(v) = \max_{0 \le i < N} v_i.

   x86-64
   ======

   - **128-bit**:

     - **Floating-point**: shuffle/min-max trees using ``min``/``max`` intrinsics.
     - **Integers**: shuffle/min-max trees using :cpp:func:`~backend::min`/:cpp:func:`~backend::max` on integer vectors, with shifts to pair up elements (apart from 64-bit integers).

   - **256/512-bit**: split into halves with :cpp:func:`~backend::split`, apply element-wise :cpp:func:`~backend::min`/:cpp:func:`~backend::max` to the halves, then recurse on the half-width result.
   - **Sub-native**: shuffle/min-max trees analogous to the 128-bit native implementation, but with shallower trees.

   Neon
   ====

   - **Native**:

     - **Floating point**: ``vminnmvq``/``vmaxnmvq`` intrinsics.
     - **Integers**:

       - **8/16/32-bit**: ``vminvq``/``vmaxvq`` intrinsics.
       - **64-bit**: extract both lanes with ``vgetq_lane_*`` and apply ``std::min``/``std::max``.

   - **Sub-native**:

     - **64 bits active**: extract the low 64 bits and use ``vminv``/``vmaxv`` intrinsics.
     - **Otherwise**: one or two pairwise ``vpmin``/``vpmax`` operations followed by ``vget_lane``.

   Shared
   ======

   - **Super-native**: compute element-wise :cpp:func:`~backend::min`/:cpp:func:`~backend::max` of the two halves, then reduce the result.
