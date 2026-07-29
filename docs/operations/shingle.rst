.. cpp:namespace:: grex

#########
Shingling
#########

Shingling shifts a vector by one lane, dropping the element that falls off the end and filling the vacated lane either with zero or with an explicitly supplied scalar.
It is the building block for sliding-window computations, where consecutive iterations overlap by all but one element.

Sub-native vectors are processed via their backing native vectors, while a super-native vector carries the element crossing the midpoint from one half into the other.

.. _operations-shingle-up:

**********
Shingle Up
**********

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_up(Vector<T, N> v)

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_up(Scalar<T> front, Vector<T, N> v)

   Shifts all lanes up by one position, discarding the highest lane.
   The vacated lane 0 receives :math:`0` for the one-argument overload and ``front`` for the two-argument overload:

   .. math::

      r_i =
      \begin{cases}
        0 \text{ or } \mathit{front} & i = 0 \\
        v_{i-1}                      & i > 0
      \end{cases}

   Shared
   ======

   - **Super-native**:

     - Lower half: :cpp:func:`~backend::shingle_up` of ``v.lower`` (with ``front`` if given).
     - Upper half: :cpp:func:`~backend::shingle_up` of ``v.upper`` with the highest lane of ``v.lower`` — obtained via :cpp:func:`~backend::extract` — as the incoming element.

   x86-64
   ======

   - **128-bit**:

     - **Zero fill**: byte-wise shift of the whole register with ``_mm_bslli_si128``.
     - **Value fill**:

       - **64-bit**: expand the scalar with :cpp:func:`~backend::expand_any` and combine via ``_mm_unpacklo_epi64``.
       - **32-bit**: expand the scalar and merge with ``_mm_movelh_ps`` plus ``_mm_shuffle_ps``.
       - **8/16-bit**: zero-fill shingle via :cpp:func:`~backend::expand_zero` and a bitwise OR with the broadcast scalar.

   - **256-bit (x86-64-v3)**: ``_mm256_alignr_epi8`` across the two 128-bit lanes:

     - Build an auxiliary register holding zeros or the broadcast ``front`` in the lower half and the lower half of ``v`` in the upper half, using ``_mm256_inserti128_si256``.
     - ``_mm256_alignr_epi8`` then shifts by one element within each 128-bit lane, pulling in the correct neighbour.

   - **512-bit (x86-64-v4)**:

     - **32/64-bit elements**: ``_mm512_alignr_epi{32,64}`` against a zero register or the broadcast scalar.
     - **8/16-bit elements**: ``_mm512_alignr_epi64`` first moves the 128-bit lanes by one, then ``_mm512_alignr_epi8`` performs the lane-local single-element shift.

   Neon
   ====

   - **Zero fill**: ``vextq`` against ``vdupq_n(0)`` at offset :math:`N - 1`.
   - **Value fill**:

     - **Integers**: ``vextq`` of ``v`` with itself at offset :math:`N - 1`, then ``vsetq_lane`` to place ``front`` in lane 0.
     - **Floating point**: expand ``front`` with :cpp:func:`~backend::expand_any` and combine with two ``vextq`` operations.

   - **Sub-native**: the same pattern applied to the low 64 bits via ``vext``, then re-widened.

.. _operations-shingle-down:

************
Shingle Down
************

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_down(Vector<T, N> v)

.. cpp:function:: template<Vectorizable T, std::size_t N> \
                  Vector<T, N> backend::shingle_down(Vector<T, N> v, Scalar<T> back)

   Shifts all lanes down by one position, discarding lane 0.
   The vacated highest lane receives :math:`0` for the one-argument overload and ``back`` for the two-argument overload:

   .. math::

      r_i =
      \begin{cases}
        v_{i+1}                     & i < N - 1 \\
        0 \text{ or } \mathit{back} & i = N - 1
      \end{cases}

   Shared
   ======

   - **Super-native**:

     - Lower half: :cpp:func:`~backend::shingle_down` of ``v.lower`` with lane 0 of ``v.upper`` — obtained via :cpp:func:`~backend::extract` — as the incoming element.
     - Upper half: :cpp:func:`~backend::shingle_down` of ``v.upper`` (with ``back`` if given).

   x86-64
   ======

   - **128-bit**:

     - **Zero fill**: byte-wise shift of the whole register with ``_mm_bsrli_si128``.
     - **Value fill**:

       - **x86-64-v2+**: shift down with ``_mm_bsrli_si128``, then write ``back`` into the highest lane with an ``_mm_insert`` intrinsic (``_mm_shuffle_pd``/``_mm_shuffle_ps`` for floating point).
       - **x86-64-v1**: shifts and unpacks combined with a broadcast of ``back``, since the ``insert`` intrinsics are unavailable.

   - **256-bit (x86-64-v3)**: ``_mm256_alignr_epi8`` as for shingling up, with the auxiliary register built by ``_mm256_zextsi128_si256`` (zero fill) or ``_mm256_permute2x128_si256`` (value fill).
   - **512-bit (x86-64-v4)**: same split as for shingling up — ``alignr_epi{32,64}`` for 32/64-bit elements, and a ``_mm512_alignr_epi64`` plus ``_mm512_alignr_epi8`` pair for 8/16-bit elements.
   - **Sub-native**: dedicated per-shape paths using ``_mm_shuffle_epi32``/``_mm_bsrli_si128`` (zero fill) and ``_mm_insert_epi8``/``_mm_insert_epi16`` (value fill), so that only the active lanes are touched.

   Neon
   ====

   - **Zero fill**: ``vextq`` of ``v`` with ``vdupq_n(0)`` at offset 1.
   - **Value fill**: expand ``back`` with :cpp:func:`~backend::expand_any` and combine with ``vextq`` at offset 1.
   - **Sub-native**: per-shape paths on the low 64 bits using ``vext``, ``vtrn2``, and ``vset_lane``, then re-widened.
