.. cpp:namespace:: grex

##########
Conversion
##########

Element-wise type conversion between vectors and masks.

.. _operations-convert-vector:

*****************
Vector Conversion
*****************

.. cpp:function:: template<Vectorizable Src, Vectorizable Dst, std::size_t N> \
                  Vector<Dst, N> backend::convert(Vector<Src, N> v, TypeTag<Dst>)

   Element-wise conversion from ``Src`` to ``Dst``.

Shared
======

- **Identical source/destination type**: returns ``v`` unchanged.
- **Same-width integers**: bitwise reinterpretation; values are unchanged.
- **Sub-native → sub-native**:

  - Expand the sub-native vector to the smallest size at which at least one of the source or the destination is native.
  - Convert this temporary vector to ``Dst`` using one of the above paths.
  - Reinterpret the low :math:`N` lanes of the converted temporary as a sub-native vector of ``Dst``.

- **Super-native → super-native**: convert each half separately and recombine.

Only this limited subset of generic conversion operations is shared because the two backends have different instructions to make use of: Arm Neon only provides integer widening/narrowing instructions which double/halve the size of the input elements, whereas x86-64 provides instructions for larger increases/decreases (starting on level 2).

x86-64
======

- **Integer widening (same signedness)**:

  - **x86-64-v2+**: integer-extend intrinsics for all factors (×2/×4/×8).
  - **x86-64-v1**:

    - Unsigned ×2: unpack with zeros in the high half.
    - Signed ×2: unpack plus arithmetic shift to propagate the sign.
    - ×4/×8: multiple ×2 widening steps.

- **Integer widening (mixed signedness)**: first widen to a temporary integer whose signedness matches the source and whose width matches the destination, then reinterpret that temporary as ``Dst``.
- **Integer narrowing** (always truncating low bits):

  - **x86-64-v4**: direct truncation intrinsics for all integer widths.
  - **x86-64-v2/v3**:

    - Byte/word shuffles or permutes to place the desired low lanes first, then keep only those.
    - 256-bit 64→16 bits: blend high words with zero and use packing instructions so that only the low bits survive.

  - **x86-64-v1**:

    - Shuffles plus ``pack``-style operations; masks are used where needed to clear high bits before packing.
    - Sub- and super-native sizes have dedicated cases so only active lanes are preserved.

  - For mixed signedness, truncation is performed via the corresponding unsigned narrowing and then reinterpreted.

- **Floating-point ↔ floating-point**:

  - Use conversion intrinsics between ``f32`` and ``f64`` on all levels.

- **Integer → floating-point**:

  For an integer :math:`n`, produce the nearest representable ``f32``/``f64`` to :math:`n` (exact when :math:`n` is in range).

  - **32-bit inputs**:

    - ``i32`` → ``f32``/``f64``: direct conversion intrinsics.
    - ``u32`` → ``f64``:

      - **x86-64-v4**: direct conversion.
      - **Earlier**: treat ``u32`` as the low 32 bits of a 64-bit mantissa:

        - Form a value whose mantissa contains ``n`` and exponent :math:`2^{52}`, so the ``f64`` equals :math:`n + 2^{52}`.
        - Subtract :math:`2^{52}` encoded with the same exponent to obtain exactly :math:`n`.

    - ``u32`` → ``f32``:

      - **x86-64-v4**: direct conversion.
      - **Earlier** (x86-64-v1 and x86-64-v2/v3 use slightly different shuffles):

        - Decompose :math:`n = \text{lo} + 2^{16} \cdot \text{hi}` with 16-bit ``lo``/``hi``.
        - Embed ``lo`` and ``hi`` into mantissas with exponents :math:`2^{23}` and :math:`2^{39}` respectively, so both are exact integers but each gains an extra hidden bit.
        - Subtract the hidden-bit contributions :math:`2^{23}` and :math:`2^{39}` and add the resulting ``f32`` values.

  - **64-bit inputs**:

    - **x86-64-v4**: direct conversion intrinsics.
    - **Earlier**:

      - ``i64`` → ``f32``/``f64``:

        - **x86-64-v4**: direct packed conversion intrinsics.
        - **Earlier**: extract each lane to a scalar register, perform scalar ``i64`` → ``f32``/``f64``, then repack.

      - ``u64`` → ``f64``:

        - Decompose :math:`n = \text{lo} + 2^{32} \cdot \text{hi}` with 32-bit ``lo``/``hi``.
        - Embed into mantissas with exponents :math:`2^{52}` and :math:`2^{84}` to obtain exact integer values with extra hidden bits.
        - Subtract the hidden-bit terms :math:`2^{52}` and :math:`2^{84}` from the constructed ``f64`` values and add the results.

      - ``u64`` → ``f32``:

        - Compute :math:`h = \lfloor n / 2 \rceil` as ``i64`` (right shift plus rounding to even).
        - Convert :math:`h` to ``f32``, and multiply by 2.

          The division by 2 keeps the value non-negative so a signed conversion suffices; the final scaling restores the original magnitude with the same rounding as a direct ``u64`` → ``f32``.

  - **Small integers (< 32 bits)**: widen to ``i32``, then convert to the desired floating-point type using the above paths.

- **Floating-point → integer**:

  Let :math:`x` be a floating-point lane value; all conversions use truncation toward zero.

  - ``f32``/``f64`` → ``i32``: direct conversion intrinsics.
  - ``f32``/``f64`` → ``i64``:

    - **x86-64-v4**: direct truncating intrinsics.
    - **Earlier**: scalar lane-wise truncation and repacking.

  - ``f64``/``f32`` → ``u64``/``u32``:

    - **x86-64-v4**: direct truncating intrinsics.
    - **Earlier**:

      - Let :math:`B` be the number of bits in the destination type (32/64) and let :math:`c_i` denote the hardware conversion from floating-point to a *signed* integer with :math:`B` bits.
      - Compute a signed truncation :math:`v = c_i(x)`.
        For :math:`x \in [0, 2^{B-1})`, this already equals the desired unsigned result.
        For :math:`x \in [2^{B-1}, 2^B)`, this produces the *indefinite integer value* :math:`2^{B-1}`.
      - Compute an *offset* truncation :math:`o = c_i(x - 2^{B-1})`.
        For :math:`x \in [2^{B-1}, 2^B)`, :math:`x - 2^{B-1} \in [0, 2^{B-1})`, so :math:`o` is the exact integer :math:`c_u(x) - 2^{B-1}`, where :math:`c_u` is the unsigned conversion.
      - Use the sign bit of :math:`v` as a mask: it is zero for :math:`x \in [0, 2^{B-1})` and all ones for :math:`x \in [2^{B-1}, 2^B)`.
        The implementation forms :math:`m = o \land \text{sign}(v)` and returns :math:`v \lor m`.

        - If :math:`x \in [0, 2^{B-1})`, :math:`\text{sign}(v) = 0`, so :math:`m = 0` and the result is :math:`v`.
        - If :math:`x \in [2^{B-1}, 2^B)`, :math:`v` contributes the high bit :math:`2^{B-1}` and :math:`m = c_u(x) - 2^{B-1}`; the bitwise OR reconstructs :math:`c_u(x)`.

      As in the C++ standard, the behaviour is only specified for :math:`x \in [0, 2^B)`.

  - **Small integers (< 32 bits)**: convert to ``i32`` with truncation, then narrow using the integer paths above.

- **Conversions involving super-native vectors**:

  - **Native → super-native**:

    - Split the native source vector into low and high halves.
    - Convert each half independently to the destination element type.
    - Merge the converted halves into the super-native result.

  - **Sub-native → super-native (integer sources)**:

    - First widen the integer element type so that an :math:`N`-lane vector fits exactly into one native register (i.e. use an integer type of size :math:`16 / N` bytes on SSE/AVX).
    - Convert this native-width integer vector to the destination type (which may itself be super-native) using the rules above.

Neon
====

- **Floating-point ↔ floating-point**: ``vcvt``/``vcvt_high`` intrinsics.
- **Integer ↔ floating-point**:

  - **Same bit count**: ``vcvt`` intrinsics.
  - **Integer → larger floating-point**: first widen to an integer type matching the destination size, then convert that integer to floating point.
  - **Floating-point → larger integer**: first convert to a floating-point type matching the destination integer size, then convert to integer (with truncation).
  - **Integer → smaller floating-point**: convert to a floating-point type matching the original integer size, then convert down.
  - **Floating-point → smaller integer**: convert to an integer type matching the original floating-point size (with truncation), then narrow.

- **Integer widening**:

  - **Factor 2**: ``vmovl``/``vmovl_high`` intrinsics.
  - **Factor 4/8**: multiple factor-2 widening steps.

- **Integer narrowing**:

  - **Factor 2**: implemented with ``vmovn`` for native-width vectors; super-native 64→32-bit narrowing uses ``vuzp1q`` on the two native halves to select the low halves.
  - **Factor 4/8**: multiple factor-2 narrowing steps.

- **Same-width integers with different signedness**: bitwise reinterpretation between signed and unsigned types.

.. _operations-convert-mask:

***************
Mask Conversion
***************

.. cpp:function:: template<AnyMask MSrc, typename Dst> \
                  Mask<Dst, MSrc::size> backend::convert(MSrc m, TypeTag<Dst>)

   Element-wise mask conversion between element types, preserving Boolean lane values.

x86-64
======

- **x86-64-v4 (compressed masks)**:

  - Native masks: reinterpret the underlying mask register between element types; bit layout is unchanged.
  - Super-native masks: split into halves, convert each half, then merge.

- **Earlier (broad masks)**:

  - **Baseline**: convert the mask to its integer-vector form, convert that vector to a signed integer type whose width matches ``Dst``, then reinterpret this vector as a mask of element type ``Dst``.
  - **128-bit masks**:

    - **Widening**: replicate mask bits with ``unpack``-style operations; for larger ratios, widen in multiple doubling steps.
    - **Narrowing**: compress to smaller element widths using ``_mm_packs_epi16``, with recursive halving for larger ratios; sub-native masks convert via the corresponding full native mask and then re-wrap.

  - **Super-native masks**: convert lower and upper halves independently and merge the results.

Neon
====

- Convert the mask to its integer-vector form, convert that vector to a signed integer type whose width matches ``Dst``, then reinterpret this vector as a mask of element type ``Dst``.
- Super-native masks are converted by processing lower and upper halves separately and merging the results.
