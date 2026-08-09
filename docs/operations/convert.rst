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

   Conversions with a binary16 source or destination are described in :ref:`operations-convert-f16`.

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
     Only signed 32-bit sources — and, on x86-64-v4, all of them — have direct instructions; everything else is reduced to those.

     - **Small integers (< 32 bits)**: widen to ``i32`` first.
     - **Unsigned before x86-64-v4**: exploit that writing :math:`n` into the mantissa of a constant exponent yields :math:`n + 2^e` exactly, so subtracting :math:`2^e` as a floating-point value leaves :math:`n`.
       Where the mantissa is too narrow for the whole value — ``u32`` → ``f32`` and ``u64`` → ``f64`` — :math:`n` is split into two halves that are converted this way and added.
       ``u64`` → ``f32`` instead halves :math:`n` with rounding to even, converts the result as a (now non-negative) ``i64``, and doubles it again, which rounds exactly as the direct conversion would.
     - ``i64`` **before x86-64-v4**: extract each lane to a scalar register, convert it there, and repack.

   - **Floating-point → integer**: all conversions truncate toward zero.

     - ``f32``/``f64`` → ``i32``: direct conversion intrinsics.
     - **Small integers (< 32 bits)**: convert to ``i32``, then narrow using the integer paths above.
     - ``f32``/``f64`` → ``i64``: direct truncating intrinsics on x86-64-v4, scalar lane-wise truncation and repacking earlier.
     - ``f32``/``f64`` → ``u32``/``u64`` before x86-64-v4: let :math:`B` be the destination width and :math:`c_i` the hardware truncation to a *signed* :math:`B`-bit integer.
       For :math:`x < 2^{B-1}`, :math:`c_i(x)` is already the desired result, and for larger :math:`x` it yields the indefinite value :math:`2^{B-1}`, i.e. exactly the high bit, while :math:`c_i(x - 2^{B-1})` yields the remaining bits.
       Masking the latter with the sign bit of the former and combining both with a bitwise OR therefore covers both ranges without a branch.
       As in the C++ standard, the behaviour is only specified for :math:`x \in [0, 2^B)`.

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

.. _operations-convert-f16:

*******************
Binary16 Conversion
*******************

.. cpp:function:: template<std::size_t N> \
                  Vector<f32, N> backend::f16_to_f32(Vector<f16, N> v)

   Widens a binary16 vector to binary32, which is always exact.

.. cpp:function:: template<std::size_t N> \
                  Vector<f16, N> backend::f32_to_f16(Vector<f32, N> v)

   Narrows a binary32 vector to the nearest binary16 values, rounding ties to even.

   These two are the pivot of all binary16 support: :cpp:func:`~backend::convert` uses them directly for ``f32``, routes every other type through binary32, and every operation without binary16 instructions is emulated with them (see :ref:`f16-implementation`).
   Since a binary32 vector needs twice the register space of a binary16 vector with the same lane count, one of the two is super-native whenever the other is native.

   x86-64
   ======

   - **With AVX512-FP16**: direct ``cvt`` intrinsics between binary16 and every other numeric type, so the detour through binary32 is unnecessary; only 8-bit integers have no instruction and go through 16-bit ones.
   - **x86-64-v3 and later**: the F16C instructions ``vcvtph2ps``/``vcvtps2ph`` at the widest applicable width, splitting the source or merging the result where the other side does not fit into a single register.
   - **x86-64-v1 and x86-64-v2**: F16C is unavailable, so both directions are emulated with SSE2, using x86-64-v2 instructions where they are cheaper.
     The exponent is re-biased by an integer addition, subnormals are normalized by a single floating-point addition, and the normal, subnormal, and infinity/not-a-number cases are combined by blending.
     Rounding is to nearest, ties to even, matching the hardware instructions.
     Eight lanes is the widest vector converted this way, since F16C is unconditionally available from x86-64-v3 on.

   Neon
   ====

   - ``vcvt_f32_f16``/``vcvt_high_f32_f16`` and ``vcvt_f16_f32``/``vcvt_high_f16_f32``, which ARM64 always provides, so no software fallback is needed.
   - With the FP16 extension, binary16 ↔ ``i16``/``u16`` additionally use ``vcvtq`` intrinsics; ``i8``/``u8`` use ``i16``/``u16`` as intermediary type when converting, all remaining types use binary32.

.. cpp:function:: template<std::size_t N> \
                  Vector<f64, N> backend::f16_to_f64(Vector<f16, N> v)

   Widens a binary16 vector to binary64, which is always exact.

.. cpp:function:: template<std::size_t N> \
                  Vector<f16, N> backend::f64_to_f16(Vector<f64, N> v)

   Narrows a binary64 vector to the nearest binary16 values, rounding ties to even — in a *single* rounding step, unlike a plain conversion by way of binary32.

   Binary64 is the one type that cannot be routed through binary32 naively.
   Narrowing twice with round-to-nearest rounds twice: a value just past a binary16 rounding boundary can be pulled exactly onto it by the first step and then sent the wrong way by the tie rule of the second.
   No amount of intermediate precision repairs this for an arbitrary binary64 input, since the trap window merely shrinks with the intermediate format.
   Rather than repeat the bit manipulation of :cpp:func:`~backend::f32_to_f16` for the wider format, the narrowing step therefore rounds *to odd*: it truncates the binary64 significand towards zero to the 24 bits of binary32 and forces the lowest surviving bit to one whenever anything was discarded, which yields whichever of the two neighbouring binary32 values has an odd significand.

   Rounding that intermediate to binary16 then gives the correctly rounded result.
   Write :math:`p = 11` for the binary16 significand and :math:`q = 24` for the binary32 one, and let :math:`x` lie between the adjacent binary16 values :math:`a` and :math:`b` with midpoint :math:`m`:

   - :math:`m` is a multiple of half a binary16 ulp, so it needs :math:`p + 1` significand bits and is exactly representable in binary32; its :math:`q`-bit significand ends in :math:`q - p - 1 \ge 1` zeros and is therefore *even*.
   - Rounding to odd never produces an even significand unless it leaves the value untouched, so the intermediate equals :math:`m` only if :math:`x` did — a genuine tie stays a tie and is broken identically.
   - Rounding to odd moves :math:`x` to an adjacent binary32 value, and :math:`m` is itself one, so the intermediate cannot cross :math:`m`.

   The intermediate therefore lies strictly on the same side of :math:`m` as :math:`x`, and the second rounding picks the same :math:`a` or :math:`b` that rounding :math:`x` directly would.
   This needs only :math:`q \ge p + 2 = 13` bits, which binary32 exceeds comfortably.

   Two details ensure that special cases are handled correctly: forcing the lowest bit to one can never carry into the exponent, because a significand of all ones is already odd, and it also keeps a not-a-number whose payload lives entirely in the discarded bits from collapsing into an infinity.
   Where the intermediate falls outside the binary32 exponent range the hardware narrowing is not exact, but those magnitudes are far beyond binary16’s own range and round to zero or infinity either way.

   Widening needs no such care, since both steps are exact.

   Shared
   ======

   - The sticky bit is obtained without a comparison: adding the mask of the discarded bits to those bits carries into the lowest surviving bit exactly if any of them is set.
   - Only the native register widths convert; wider vectors split into halves and merge the results.

   x86-64
   ======

   - **With AVX512-FP16**: ``vcvtph2pd``/``vcvtpd2ph`` convert directly in one instruction, so :cpp:func:`~backend::convert` uses those and the round-to-odd path does not exist.
   - **Otherwise**: round to odd, then ``cvtpd_ps``/``cvtps_pd`` at the widest applicable width, then :cpp:func:`~backend::f32_to_f16`/:cpp:func:`~backend::f16_to_f32`.

   Neon
   ====

   - ARM64 has no binary16 ↔ binary64 instruction at any extension level, so the round-to-odd path is always used, with ``fcvtn``/``fcvtl`` and their ``_high`` forms for the binary64 half.

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
