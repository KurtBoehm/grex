##########
Operations
##########

The name of most operations links to the documentation of that operation and some notes on its implementation in the different backends.

Throughout this documentation, ``Vector<T, N>`` denotes any backend vector type with value type ``T`` and lane count ``N``, i.e. one of:

- ``backend::NativeVector<T, N>``
- ``backend::SubVector<T, N, M>`` with some ``M`` for which ``backend::NativeVector<T, M>`` is defined.
- ``backend::SuperVector<THalf>`` where ``THalf::Value == T`` and ``2 * THalf::size == N``.

Similarly, ``Mask<T, N>`` denotes any backend mask type for value type ``T`` and lane count ``N``, i.e. one of:

- ``backend::NativeMask<T, N>``
- ``backend::SubMask<T, N, M>`` with some ``M`` for which ``backend::NativeMask<T, M>`` is defined.
- ``backend::SuperMask<THalf>`` where ``THalf::Value == T`` and ``2 * THalf::size == N``.

##########################
Vector-Specific Operations
##########################

.. list-table::
   :header-rows: 1

   * - Operation
     - Signature/Description

   * - :ref:`Construct zero vector <operations-zeros-vector>`
     - :cpp:func:`Vector::Vector() <Vector grex::Vector::Vector()>`

   * - :ref:`Broadcast scalar <operations-broadcast-vector>`
     - :cpp:func:`Vector::Vector(T value) <Vector grex::Vector::Vector(T)>`

   * - :ref:`Construct from per-lane values <operations-set-vector>`
     - ``Vector::Vector(T... values)``

   * - Construct from backend
     - ``Vector::Vector(Backend v)``

   * - Expand scalar (undefined upper lanes)
     - :cpp:func:`Vector::expanded_any(T value) <Vector grex::Vector::expanded_any(T)>`

   * - Expand scalar (zero upper lanes)
     - :cpp:func:`Vector::expanded_zero(T value) <Vector grex::Vector::expanded_zero(T)>`

   * - :ref:`Load (unaligned) <operations-load>`
     - :cpp:func:`Vector::load(const T* ptr) <Vector grex::Vector::load(const T*)>`

   * - :ref:`Load (aligned) <operations-load-aligned>`
     - :cpp:func:`Vector::load_aligned(const T* ptr) <Vector grex::Vector::load_aligned(const T*)>`

   * - :ref:`Load partial (runtime) <operations-load-part-runtime>`
     - :cpp:func:`Vector::load_part(const T* ptr, std::size_t num) <Vector grex::Vector::load_part(const T*, std::size_t)>`

   * - :ref:`Load partial (compile-time) <operations-load-part-ct>`
     - :cpp:func:`Vector::load_part(const T* ptr, AnyIndexTag auto num) <Vector grex::Vector::load_part(const T*, AnyIndexTag)>`

   * - Load multibyte
     - | :cpp:func:`Vector::load_multibyte(const std::byte* data, AnyIndexTag auto src_bytes) <template<std::size_t tSrcBytes> Vector grex::Vector::load_multibyte(const std::byte*, IndexTag<tSrcBytes>)>`
       | :cpp:func:`Vector::load_multibyte(TIt it) <template<MultiByteIterator TIt> Vector grex::Vector::load_multibyte(TIt)>`

   * - :ref:`Undefined vector <operations-undefined-vector>`
     - :cpp:func:`Vector::undefined() <Vector grex::Vector::undefined()>`

   * - :ref:`Zero vector <operations-zeros-vector>`
     - :cpp:func:`Vector::zeros() <Vector grex::Vector::zeros()>`

   * - Lane indices
     - | :cpp:func:`Vector::indices() <Vector grex::Vector::indices()>`
       | :cpp:func:`Vector::indices(T start) <Vector grex::Vector::indices(T)>`

   * - :ref:`Unary minus <operations-negation>`
     - :cpp:func:`Vector::operator-() const <Vector grex::Vector::operator-() const>`

   * - :ref:`Bitwise NOT <operations-bitwise-not>`
     - :cpp:func:`Vector::operator~() const <Vector grex::Vector::operator~() const>`

   * - :ref:`Addition <operations-addition>`
     - | :cpp:func:`operator+(Vector a, Vector b) <Vector grex::Vector::operator+(Vector, Vector)>`
       | :cpp:func:`operator+(Vector a, Value b) <Vector grex::Vector::operator+(Vector, Value)>`
       | :cpp:func:`operator+(Value a, Vector b) <Vector grex::Vector::operator+(Value, Vector)>`
       | :cpp:func:`operator+=(Vector b) <Vector& grex::Vector::operator+=(Vector)>`
       | :cpp:func:`operator+=(Value b) <Vector& grex::Vector::operator+=(Value)>`

   * - :ref:`Subtraction <operations-subtraction>`
     - | :cpp:func:`operator-(Vector a, Vector b) <Vector grex::Vector::operator-(Vector, Vector)>`
       | :cpp:func:`operator-(Vector a, Value b) <Vector grex::Vector::operator-(Vector, Value)>`
       | :cpp:func:`operator-(Value a, Vector b) <Vector grex::Vector::operator-(Value, Vector)>`
       | :cpp:func:`operator-=(Vector b) <Vector& grex::Vector::operator-=(Vector)>`
       | :cpp:func:`operator-=(Value b) <Vector& grex::Vector::operator-=(Value)>`

   * - :ref:`Multiplication <operations-multiplication>`
     - | :cpp:func:`operator*(Vector a, Vector b) <Vector grex::Vector::operator*(Vector, Vector)>`
       | :cpp:func:`operator*(Vector a, Value b) <Vector grex::Vector::operator*(Vector, Value)>`
       | :cpp:func:`operator*(Value a, Vector b) <Vector grex::Vector::operator*(Value, Vector)>`
       | :cpp:func:`operator*=(Vector b) <Vector& grex::Vector::operator*=(Vector)>`
       | :cpp:func:`operator*=(Value b) <Vector& grex::Vector::operator*=(Value)>`

   * - :ref:`Division <operations-division>`
     - | :cpp:func:`operator/(Vector a, Vector b) <Vector grex::Vector::operator/(Vector, Vector)>`
       | :cpp:func:`operator/(Vector a, Value b) <Vector grex::Vector::operator/(Vector, Value)>`
       | :cpp:func:`operator/(Value a, Vector b) <Vector grex::Vector::operator/(Value, Vector)>`
       | :cpp:func:`operator/=(Vector b) <Vector& grex::Vector::operator/=(Vector)>`
       | :cpp:func:`operator/=(Value b) <Vector& grex::Vector::operator/=(Value)>`

   * - :ref:`Bitwise AND <operations-bitwise-and>`
     - | :cpp:func:`operator&(Vector a, Vector b) <Vector grex::Vector::operator&(Vector, Vector)>`
       | :cpp:func:`operator&(Vector a, Value b) <Vector grex::Vector::operator&(Vector, Value)>`
       | :cpp:func:`operator&(Value a, Vector b) <Vector grex::Vector::operator&(Value, Vector)>`
       | :cpp:func:`operator&=(Vector b) <Vector& grex::Vector::operator&=(Vector)>`
       | :cpp:func:`operator&=(Value b) <Vector& grex::Vector::operator&=(Value)>`

   * - :ref:`Bitwise OR <operations-bitwise-or>`
     - | :cpp:func:`operator|(Vector a, Vector b) <Vector grex::Vector::operator|(Vector, Vector)>`
       | :cpp:func:`operator|(Vector a, Value b) <Vector grex::Vector::operator|(Vector, Value)>`
       | :cpp:func:`operator|(Value a, Vector b) <Vector grex::Vector::operator|(Value, Vector)>`
       | :cpp:func:`operator|=(Vector b) <Vector& grex::Vector::operator|=(Vector)>`
       | :cpp:func:`operator|=(Value b) <Vector& grex::Vector::operator|=(Value)>`

   * - :ref:`Bitwise XOR <operations-bitwise-xor>`
     - | :cpp:func:`operator^(Vector a, Vector b) <Vector grex::Vector::operator^(Vector, Vector)>`
       | :cpp:func:`operator^(Vector a, Value b) <Vector grex::Vector::operator^(Vector, Value)>`
       | :cpp:func:`operator^(Value a, Vector b) <Vector grex::Vector::operator^(Value, Vector)>`
       | :cpp:func:`operator^=(Vector b) <Vector& grex::Vector::operator^=(Vector)>`
       | :cpp:func:`operator^=(Value b) <Vector& grex::Vector::operator^=(Value)>`

   * - :ref:`Shift left <operations-shift-left>`
     - | :cpp:func:`operator\<\<(Vector, AnyIndexTag auto) <Vector grex::Vector::operator<<(Vector, AnyIndexTag)>`
       | :cpp:func:`operator\<\<=(AnyIndexTag auto) <Vector& grex::Vector::operator<<=(AnyIndexTag)>`

   * - :ref:`Shift right <operations-shift-right>`
     - | :cpp:func:`operator>>(Vector, AnyIndexTag auto) <Vector grex::Vector::operator>>(Vector, AnyIndexTag)>`
       | :cpp:func:`operator>>=(AnyIndexTag auto) <Vector& grex::Vector::operator>>=(AnyIndexTag)>`

   * - Cut off lanes
     - :cpp:func:`Vector::cutoff(std::size_t i) const <Vector grex::Vector::cutoff(std::size_t) const>`

   * - Convert type
     - :cpp:func:`Vector::convert(AnyTypeTag) const <template<Vectorizable TDst> Vector<TDst, size> grex::Vector::convert(TypeTag<TDst>) const>`

   * - Element access
     - | :cpp:func:`Vector::operator[](std::size_t i) const <T grex::Vector::operator[](std::size_t) const>`
       | :cpp:func:`Vector::operator[](AnyIndexTag auto i) const <T grex::Vector::operator[](AnyIndexTag) const>`
       | :cpp:func:`get\<index>(const Vector&) <template<std::size_t tIdx> T grex::Vector::get(const Vector&)>`
       | :cpp:func:`Vector::insert(std::size_t i, T value) const <Vector grex::Vector::insert(std::size_t, T) const>`
       | :cpp:func:`Vector::insert(AnyIndexTag auto i, T value) const <Vector grex::Vector::insert(AnyIndexTag, T) const>`

   * - :ref:`Store (unaligned) <operations-store>`
     - :cpp:func:`Vector::store(T* ptr) const <void grex::Vector::store(T*) const>`

   * - :ref:`Store (aligned) <operations-store-aligned>`
     - :cpp:func:`Vector::store_aligned(T* ptr) const <void grex::Vector::store_aligned(T*) const>`

   * - :ref:`Store partial (runtime) <operations-store-part-runtime>`
     - :cpp:func:`Vector::store_part(T* ptr, std::size_t num) const <void grex::Vector::store_part(T*, std::size_t) const>`

   * - :ref:`Store partial (compile-time) <operations-store-part-ct>`
     - :cpp:func:`Vector::store_part(T* ptr, AnyIndexTag auto num) const <void grex::Vector::store_part(T*, AnyIndexTag) const>`

   * - Comparison
     - | :cpp:func:`operator==(Vector, Vector) <Mask grex::Vector::operator==(Vector, Vector)>`
       | :cpp:func:`operator!=(Vector, Vector) <Mask grex::Vector::operator!=(Vector, Vector)>`
       | :cpp:func:`operator\<(Vector, Vector) <Mask grex::Vector::operator<(Vector, Vector)>`
       | :cpp:func:`operator>(Vector, Vector) <Mask grex::Vector::operator>(Vector, Vector)>`
       | :cpp:func:`operator\<=(Vector, Vector) <Mask grex::Vector::operator<=(Vector, Vector)>`
       | :cpp:func:`operator>=(Vector, Vector) <Mask grex::Vector::operator>=(Vector, Vector)>`

   * - Expand (undefined upper)
     - :cpp:func:`Vector::expand_any(AnyIndexTag) const <template<std::size_t tDstSize> Vector<T, tDstSize> grex::Vector::expand_any(IndexTag<tDstSize>) const>`

   * - Expand (zero upper)
     - :cpp:func:`Vector::expand_zero(AnyIndexTag) const <template<std::size_t tDstSize> Vector<T, tDstSize> grex::Vector::expand_zero(IndexTag<tDstSize>) const>`

   * - Shingle up
     - | :cpp:func:`Vector::shingle_up() const <Vector grex::Vector::shingle_up() const>`
       | :cpp:func:`Vector::shingle_up(Value front) const <Vector grex::Vector::shingle_up(Value) const>`

   * - Shingle down
     - | :cpp:func:`Vector::shingle_down() const <Vector grex::Vector::shingle_down() const>`
       | :cpp:func:`Vector::shingle_down(Value back) const <Vector grex::Vector::shingle_down(Value) const>`

   * - Backend access
     - | :cpp:func:`Vector::backend() const <Backend grex::Vector::backend() const>`
       | :cpp:func:`Vector::as_array() const <std::array grex::Vector::as_array() const>`

########################
Mask-Specific Operations
########################

.. list-table::
   :header-rows: 1

   * - Operation
     - Signature/Description

   * - :ref:`Construct all-false mask <operations-zeros-mask>`
     - :cpp:func:`Mask::Mask() <Mask grex::Mask::Mask()>`

   * - :ref:`Broadcast Boolean <operations-broadcast-mask>`
     - :cpp:func:`Mask::Mask(bool value) <Mask grex::Mask::Mask(bool)>`

   * - :ref:`Construct from per-lane values <operations-set-mask>`
     - :cpp:func:`Mask::Mask(bool... values) <template<typename... Ts> Mask grex::Mask::Mask(Ts...)>`

   * - Construct from backend
     - :cpp:func:`Mask::Mask(Backend v) <Mask grex::Mask::Mask(Backend)>`

   * - :ref:`All-false mask <operations-zeros-mask>`
     - :cpp:func:`Mask::zeros() <Mask grex::Mask::zeros()>`

   * - :ref:`All-true mask <operations-ones-mask>`
     - :cpp:func:`Mask::ones() <Mask grex::Mask::ones()>`

   * - Cut-off mask
     - :cpp:func:`Mask::cutoff_mask(std::size_t i) <Mask grex::Mask::cutoff_mask(std::size_t)>`

   * - Convert to different scalar type
     - :cpp:func:`Mask::convert(AnyTypeTag) const <template<Vectorizable TDst> Mask<TDst, tSize> grex::Mask::convert(TypeTag<TDst>) const>`

   * - :ref:`Logical NOT <operations-logical-not>`
     - :cpp:func:`Mask::operator!() const <Mask grex::Mask::operator!() const>`

   * - :ref:`Logical AND <operations-logical-and>`
     - :cpp:func:`operator&&(Mask, Mask) <Mask grex::Mask::operator&&(Mask, Mask)>`

   * - :ref:`Logical OR <operations-logical-or>`
     - :cpp:func:`operator||(Mask, Mask) <Mask grex::Mask::operator||(Mask, Mask)>`

   * - :ref:`Logical XOR <operations-logical-xor>`
     - :cpp:func:`operator!=(Mask, Mask) <Mask grex::Mask::operator!=(Mask, Mask)>`

   * - Mask equality
     - :cpp:func:`operator==(Mask, Mask) <Mask grex::Mask::operator==(Mask, Mask)>`

   * - Element access
     - | :cpp:func:`Mask::operator[](std::size_t i) const <bool grex::Mask::operator[](std::size_t) const>`
       | :cpp:func:`Mask::operator[](AnyIndexTag auto i) const <bool grex::Mask::operator[](AnyIndexTag) const>`
       | :cpp:func:`get\<index>(const Mask&) <template<std::size_t tIdx> bool grex::Mask::get(const Mask&)>`
       | :cpp:func:`Mask::insert(std::size_t i, bool value) const <Mask grex::Mask::insert(std::size_t, bool) const>`
       | :cpp:func:`Mask::insert(AnyIndexTag auto i, bool value) const <Mask grex::Mask::insert(AnyIndexTag, bool) const>`

   * - Backend access
     - | :cpp:func:`Mask::backend() const <Backend grex::Mask::backend() const>`
       | :cpp:func:`Mask::as_array() const <std::array grex::Mask::as_array() const>`

#######################
Free-Function Utilities
#######################

.. list-table::
   :header-rows: 1

   * - Operation
     - Signature/Description

   * - :ref:`Logical AND NOT <operations-logical-andnot>`
     - :cpp:func:`grex::andnot(Mask a, Mask b) <template<Vectorizable T, std::size_t tSize> Mask<T, tSize> grex::andnot(Mask<T, tSize>, Mask<T, tSize>)>`

   * - Absolute value
     - :cpp:func:`grex::abs(Vector v) <template<SignedVectorizable T, std::size_t tSize> Vector<T, tSize> grex::abs(Vector<T, tSize>)>`

   * - Square root
     - :cpp:func:`grex::sqrt(Vector v) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::sqrt(Vector<T, tSize>)>`

   * - Minimum/maximum
     - | :cpp:func:`grex::min(Vector a, Vector b) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::min(Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::max(Vector a, Vector b) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::max(Vector<T, tSize>, Vector<T, tSize>)>`

   * - Is/make finite
     - | :cpp:func:`grex::is_finite(Vector v) <template<FloatVectorizable T, std::size_t tSize> Mask<T, tSize> grex::is_finite(Vector<T, tSize>)>`
       | :cpp:func:`grex::make_finite(Vector v) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::make_finite(Vector<T, tSize>)>`

   * - Horizontal addition/minimum/maximum
     - | :cpp:func:`grex::horizontal_add(Vector v) <template<Vectorizable T, std::size_t tSize> T grex::horizontal_add(Vector<T, tSize>)>`
       | :cpp:func:`grex::horizontal_min(Vector v) <template<Vectorizable T, std::size_t tSize> T grex::horizontal_min(Vector<T, tSize>)>`
       | :cpp:func:`grex::horizontal_max(Vector v) <template<Vectorizable T, std::size_t tSize> T grex::horizontal_max(Vector<T, tSize>)>`

   * - Horizontal AND
     - :cpp:func:`grex::horizontal_and(Mask m) <template<Vectorizable T, std::size_t tSize> bool grex::horizontal_and(Mask<T, tSize>)>`

   * - Fused multiply-add family
     - | :cpp:func:`grex::fmadd(Vector a, Vector b, Vector c) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::fmadd(Vector<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::fmsub(Vector a, Vector b, Vector c) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::fmsub(Vector<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::fnmadd(Vector a, Vector b, Vector c) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::fnmadd(Vector<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::fnmsub(Vector a, Vector b, Vector c) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::fnmsub(Vector<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`

   * - Extract single value
     - :cpp:func:`grex::extract_single(Vector v) <template<Vectorizable T, std::size_t tSize> T grex::extract_single(Vector<T, tSize>)>`

   * - Blend zeros (masked)
     - :cpp:func:`grex::blend_zero(Mask mask, Vector v1) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::blend_zero(Mask<T, tSize>, Vector<T, tSize>)>`

   * - Blend zeros (compile-time selectors)
     - :cpp:func:`grex::blend_zero\<selectors>(Vector v1) <template<BlendZeroSelector... tBzs, Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::blend_zero(Vector<T, tSize>)>`

   * - Blend (masked)
     - :cpp:func:`grex::blend(Mask mask, Vector v0, Vector v1) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::blend(Mask<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`

   * - Blend (compile-time selectors)
     - :cpp:func:`grex::blend\<selectors>(Vector v0, Vector v1) <template<BlendSelector... tBls, Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::blend(Vector<T, tSize>, Vector<T, tSize>)>`

   * - Shuffle (indexed)
     - :cpp:func:`grex::shuffle(Vector table, Vector idxs) <template<Vectorizable T, UnsignedIntVectorizable TIdx, std::size_t tTableSize, std::size_t tIdxSize> Vector<T, tIdxSize> grex::shuffle(Vector<T, tTableSize>, Vector<TIdx, tIdxSize>)>`

   * - Shuffle (compile-time indices)
     - :cpp:func:`grex::shuffle(Vector table) <template<ShuffleIndex... tIdxs, Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::shuffle(Vector<T, tSize>)>`

   * - Masked arithmetic
     - | :cpp:func:`grex::mask_add(Mask mask, Vector a, Vector b) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::mask_add(Mask<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::mask_subtract(Mask mask, Vector a, Vector b) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::mask_subtract(Mask<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::mask_multiply(Mask mask, Vector a, Vector b) <template<Vectorizable T, std::size_t tSize> Vector<T, tSize> grex::mask_multiply(Mask<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`
       | :cpp:func:`grex::mask_divide(Mask mask, Vector a, Vector b) <template<FloatVectorizable T, std::size_t tSize> Vector<T, tSize> grex::mask_divide(Mask<T, tSize>, Vector<T, tSize>, Vector<T, tSize>)>`

   * - Gather
     - :cpp:func:`grex::gather(std::span\<const T, extend> data, Vector indices) <template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize> Vector<TValue, tSize> grex::gather(std::span<const TValue, tExtent>, Vector<TIndex, tSize>)>`

   * - Masked gather
     - :cpp:func:`grex::mask_gather(std::span\<const T, extend> data, Mask mask, Vector indices) <template<Vectorizable TValue, std::size_t tExtent, Vectorizable TIndex, std::size_t tSize> Vector<TValue, tSize> grex::mask_gather(std::span<const TValue, tExtent>, Mask<TValue, tSize>, Vector<TIndex, tSize>)>`
