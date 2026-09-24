##########
Operations
##########

Most operation names below link to their dedicated documentation, which also describes backend-specific implementation details.

In the documentation of each operation, ``Vector<T, N>`` denotes any backend vector type with value type ``T`` and lane count ``N``, i.e. one of:

- ``backend::NativeVector<T, N>``
- ``backend::SubVector<T, N>``
- ``backend::SuperVector<Half>`` where ``Half::Value == T`` and ``2 * Half::size == N``

Similarly, ``Mask<T, N>`` denotes any backend mask type with value type ``T`` and lane count ``N``, i.e. one of:

- ``backend::NativeMask<T, N>``
- ``backend::SubMask<T, N>``
- ``backend::SuperMask<Half>`` where ``Half::Value == T`` and ``2 * Half::size == N``

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

   * - Construct from backend vector
     - ``Vector::Vector(Backend v)``

   * - :ref:`Expand scalar (undefined upper lanes) <operations-expand-scalar-any>`
     - :cpp:func:`Vector::expanded_any(T value) <Vector grex::Vector::expanded_any(T)>`

   * - :ref:`Expand scalar (zero upper lanes) <operations-expand-scalar-zero>`
     - :cpp:func:`Vector::expanded_zero(T value) <Vector grex::Vector::expanded_zero(T)>`

   * - :ref:`Load (unaligned) <operations-load>`
     - :cpp:func:`Vector::load(const T* ptr) <Vector grex::Vector::load(const T*)>`

   * - :ref:`Load (aligned) <operations-load-aligned>`
     - :cpp:func:`Vector::load_aligned(const T* ptr) <Vector grex::Vector::load_aligned(const T*)>`

   * - :ref:`Load partial (runtime count) <operations-load-part-runtime>`
     - :cpp:func:`Vector::load_part(const T* ptr, std::size_t num) <Vector grex::Vector::load_part(const T*, std::size_t)>`

   * - :ref:`Load partial (compile-time count) <operations-load-part-ct>`
     - :cpp:func:`Vector::load_part(const T* ptr, AnyIndexTag auto num) <Vector grex::Vector::load_part(const T*, AnyIndexTag)>`

   * - :ref:`Load multibyte <operations-load-multibyte>`
     - | :cpp:func:`Vector::load_multibyte(const std::byte* data, AnyIndexTag auto src_bytes) <template\<std::size_t SrcBytes\> Vector grex::Vector::load_multibyte(const std::byte*, IndexTag\<SrcBytes\>)>`
       | :cpp:func:`Vector::load_multibyte(It it) <template\<MultiByteIterator It\> Vector grex::Vector::load_multibyte(It)>`

   * - :ref:`Undefined vector <operations-undefined-vector>`
     - :cpp:func:`Vector::undefined() <Vector grex::Vector::undefined()>`

   * - :ref:`Zero vector <operations-zeros-vector>`
     - :cpp:func:`Vector::zeros() <Vector grex::Vector::zeros()>`

   * - :ref:`Lane indices <operations-indices>`
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
     - | :cpp:func:`operator\<\<(Vector, AnyIndexTag auto) <Vector grex::Vector::operator\<\<(Vector, AnyIndexTag)>`
       | :cpp:func:`operator\<\<=(AnyIndexTag auto) <Vector& grex::Vector::operator\<\<=(AnyIndexTag)>`

   * - :ref:`Shift right <operations-shift-right>`
     - | :cpp:func:`operator\>\>(Vector, AnyIndexTag auto) <Vector grex::Vector::operator\>\>(Vector, AnyIndexTag)>`
       | :cpp:func:`operator\>\>=(AnyIndexTag auto) <Vector& grex::Vector::operator\>\>=(AnyIndexTag)>`

   * - :ref:`Cut off lanes <operations-cutoff>`
     - :cpp:func:`Vector::cutoff(std::size_t i) const <Vector grex::Vector::cutoff(std::size_t) const>`

   * - :ref:`Convert element type <operations-convert-vector>`
     - :cpp:func:`Vector::convert(AnyTypeTag) const <template\<Vectorizable Dst\> Vector\<Dst, size\> grex::Vector::convert(TypeTag\<Dst\>) const>`

   * - :ref:`Extract element (runtime index) <operations-extract-value-runtime>`
     - :cpp:func:`Vector::operator[](std::size_t i) const <T grex::Vector::operator[](std::size_t) const>`

   * - :ref:`Extract element (compile-time index) <operations-extract-value-ct>`
     - | :cpp:func:`Vector::operator[](AnyIndexTag auto i) const <T grex::Vector::operator[](AnyIndexTag) const>`
       | :cpp:func:`get\<index\>(const Vector&) <template\<std::size_t I\> T grex::Vector::get(const Vector&)>`

   * - :ref:`Insert element (runtime index) <operations-insert-value-runtime>`
     - :cpp:func:`Vector::insert(std::size_t i, T value) const <Vector grex::Vector::insert(std::size_t, T) const>`

   * - :ref:`Insert element (compile-time index) <operations-insert-value-ct>`
     - :cpp:func:`Vector::insert(AnyIndexTag auto i, T value) const <Vector grex::Vector::insert(AnyIndexTag, T) const>`

   * - :ref:`Store (unaligned) <operations-store>`
     - :cpp:func:`Vector::store(T* ptr) const <void grex::Vector::store(T*) const>`

   * - :ref:`Store (aligned) <operations-store-aligned>`
     - :cpp:func:`Vector::store_aligned(T* ptr) const <void grex::Vector::store_aligned(T*) const>`

   * - :ref:`Store partial (runtime count) <operations-store-part-runtime>`
     - :cpp:func:`Vector::store_part(T* ptr, std::size_t num) const <void grex::Vector::store_part(T*, std::size_t) const>`

   * - :ref:`Store partial (compile-time count) <operations-store-part-ct>`
     - :cpp:func:`Vector::store_part(T* ptr, AnyIndexTag auto num) const <void grex::Vector::store_part(T*, AnyIndexTag) const>`

   * - :ref:`Equality <operations-compare-eq>`
     - :cpp:func:`operator==(Vector, Vector) <Mask grex::Vector::operator==(Vector, Vector)>`

   * - :ref:`Inequality <operations-compare-neq>`
     - :cpp:func:`operator!=(Vector, Vector) <Mask grex::Vector::operator!=(Vector, Vector)>`

   * - :ref:`Strict inequality <operations-compare-lt>`
     - | :cpp:func:`operator\<(Vector, Vector) <Mask grex::Vector::operator\<(Vector, Vector)>`
       | :cpp:func:`operator\>(Vector, Vector) <Mask grex::Vector::operator\>(Vector, Vector)>`

   * - :ref:`Non-strict inequality <operations-compare-ge>`
     - | :cpp:func:`operator\<=(Vector, Vector) <Mask grex::Vector::operator\<=(Vector, Vector)>`
       | :cpp:func:`operator\>=(Vector, Vector) <Mask grex::Vector::operator\>=(Vector, Vector)>`

   * - :ref:`Expand (undefined upper lanes) <operations-expand-vector-any>`
     - :cpp:func:`Vector::expand_any(AnyIndexTag) const <template\<std::size_t DstN\> Vector\<T, DstN\> grex::Vector::expand_any(IndexTag\<DstN\>) const>`

   * - :ref:`Expand (zero upper lanes) <operations-expand-vector-zero>`
     - :cpp:func:`Vector::expand_zero(AnyIndexTag) const <template\<std::size_t DstN\> Vector\<T, DstN\> grex::Vector::expand_zero(IndexTag\<DstN\>) const>`

   * - :ref:`Shingle up (insert zero) <operations-shingle-up-zero>`
     - :cpp:func:`Vector::shingle_up() const <Vector grex::Vector::shingle_up() const>`

   * - :ref:`Shingle up (insert scalar) <operations-shingle-up-front>`
     - :cpp:func:`Vector::shingle_up(Value front) const <Vector grex::Vector::shingle_up(Value) const>`

   * - :ref:`Shingle down (insert zero) <operations-shingle-down-zero>`
     - :cpp:func:`Vector::shingle_down() const <Vector grex::Vector::shingle_down() const>`

   * - :ref:`Shingle down (insert scalar) <operations-shingle-down-back>`
     - :cpp:func:`Vector::shingle_down(Value back) const <Vector grex::Vector::shingle_down(Value) const>`

   * - :ref:`Conversion to Array <operations-to-array-vector-std-array>`
     - :cpp:func:`Vector::as_array() const <std::array grex::Vector::as_array() const>`

   * - Backend access
     - :cpp:func:`Vector::backend() const <Backend grex::Vector::backend() const>`

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
     - :cpp:func:`Mask::Mask(bool... values) <template\<typename... Ts\> Mask grex::Mask::Mask(Ts...)>`

   * - Construct from backend mask
     - :cpp:func:`Mask::Mask(Backend v) <Mask grex::Mask::Mask(Backend)>`

   * - :ref:`All-false mask <operations-zeros-mask>`
     - :cpp:func:`Mask::zeros() <Mask grex::Mask::zeros()>`

   * - :ref:`All-true mask <operations-ones-mask>`
     - :cpp:func:`Mask::ones() <Mask grex::Mask::ones()>`

   * - :ref:`Cut-off mask <operations-cutoff-mask>`
     - :cpp:func:`Mask::cutoff_mask(std::size_t i) <Mask grex::Mask::cutoff_mask(std::size_t)>`

   * - :ref:`Single-lane mask <operations-single-mask>`
     - :cpp:func:`Mask::single_mask(std::size_t i) <Mask grex::Mask::single_mask(std::size_t)>`

   * - :ref:`Convert scalar type <operations-convert-mask>`
     - :cpp:func:`Mask::convert(AnyTypeTag) const <template\<Vectorizable Dst\> Mask\<Dst, N\> grex::Mask::convert(TypeTag\<Dst\>) const>`

   * - :ref:`Logical NOT <operations-logical-not>`
     - :cpp:func:`Mask::operator!() const <Mask grex::Mask::operator!() const>`

   * - :ref:`Logical AND <operations-logical-and>`
     - :cpp:func:`operator&&(Mask, Mask) <Mask grex::Mask::operator&&(Mask, Mask)>`

   * - :ref:`Logical OR <operations-logical-or>`
     - :cpp:func:`operator||(Mask, Mask) <Mask grex::Mask::operator||(Mask, Mask)>`

   * - :ref:`Logical XOR <operations-logical-xor>`
     - :cpp:func:`operator!=(Mask, Mask) <Mask grex::Mask::operator!=(Mask, Mask)>`

   * - :ref:`Mask equality <operations-compare-eq-mask>`
     - :cpp:func:`operator==(Mask, Mask) <Mask grex::Mask::operator==(Mask, Mask)>`

   * - :ref:`Extract element (runtime index) <operations-extract-mask-runtime>`
     - :cpp:func:`Mask::operator[](std::size_t i) const <bool grex::Mask::operator[](std::size_t) const>`

   * - :ref:`Extract element (compile-time index) <operations-extract-mask-ct>`
     - | :cpp:func:`Mask::operator[](AnyIndexTag auto i) const <bool grex::Mask::operator[](AnyIndexTag) const>`
       | :cpp:func:`get\<index\>(const Mask&) <template\<std::size_t I\> bool grex::Mask::get(const Mask&)>`

   * - :ref:`Insert element (runtime index) <operations-insert-mask-runtime>`
     - :cpp:func:`Mask::insert(std::size_t i, bool value) const <Mask grex::Mask::insert(std::size_t, bool) const>`

   * - :ref:`Insert element (compile-time index) <operations-insert-mask-ct>`
     - :cpp:func:`Mask::insert(AnyIndexTag auto i, bool value) const <Mask grex::Mask::insert(AnyIndexTag, bool) const>`

   * - :ref:`Conversion to Array <operations-to-array-mask-std-array>`
     - :cpp:func:`Mask::as_array() const <std::array grex::Mask::as_array() const>`

   * - Backend access
     - :cpp:func:`Mask::backend() const <Backend grex::Mask::backend() const>`

#######################
Free-Function Utilities
#######################

.. list-table::
   :header-rows: 1

   * - Operation
     - Signature/Description

   * - :ref:`Logical AND NOT <operations-logical-andnot>`
     - :cpp:func:`grex::andnot(Mask a, Mask b) <template\<Vectorizable T, std::size_t N\> Mask\<T, N\> grex::andnot(Mask\<T, N\>, Mask\<T, N\>)>`

   * - :ref:`Absolute value <operations-abs>`
     - :cpp:func:`grex::abs(Vector v) <template\<SignedVectorizable T, std::size_t N\> Vector\<T, N\> grex::abs(Vector\<T, N\>)>`

   * - :ref:`Square root <operations-sqrt>`
     - :cpp:func:`grex::sqrt(Vector v) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::sqrt(Vector\<T, N\>)>`

   * - :ref:`Minimum <operations-min>`
     - :cpp:func:`grex::min(Vector a, Vector b) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::min(Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Maximum <operations-max>`
     - :cpp:func:`grex::max(Vector a, Vector b) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::max(Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Is finite <operations-is-finite>`
     - :cpp:func:`grex::is_finite(Vector v) <template\<FloatVectorizable T, std::size_t N\> Mask\<T, N\> grex::is_finite(Vector\<T, N\>)>`

   * - :ref:`Make finite <operations-make-finite-vector>`
     - :cpp:func:`grex::make_finite(Vector v) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::make_finite(Vector\<T, N\>)>`

   * - :ref:`Horizontal addition <operations-horizontal-add>`
     - :cpp:func:`grex::horizontal_add(Vector v) <template\<Vectorizable T, std::size_t N\> T grex::horizontal_add(Vector\<T, N\>)>`

   * - :ref:`Horizontal minimum/maximum <operations-horizontal-minmax>`
     - | :cpp:func:`grex::horizontal_min(Vector v) <template\<Vectorizable T, std::size_t N\> T grex::horizontal_min(Vector\<T, N\>)>`
       | :cpp:func:`grex::horizontal_max(Vector v) <template\<Vectorizable T, std::size_t N\> T grex::horizontal_max(Vector\<T, N\>)>`

   * - :ref:`Horizontal AND <operations-horizontal-and>`
     - :cpp:func:`grex::horizontal_and(Mask m) <template\<Vectorizable T, std::size_t N\> bool grex::horizontal_and(Mask\<T, N\>)>`

   * - :ref:`Fused multiply-add family <operations-fmadd-family>`
     - | :cpp:func:`grex::fmadd(Vector a, Vector b, Vector c) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::fmadd(Vector\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::fmsub(Vector a, Vector b, Vector c) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::fmsub(Vector\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::fnmadd(Vector a, Vector b, Vector c) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::fnmadd(Vector\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::fnmsub(Vector a, Vector b, Vector c) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::fnmsub(Vector\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Extract single value <operations-extract-single>`
     - :cpp:func:`grex::extract_single(Vector v) <template\<Vectorizable T, std::size_t N\> T grex::extract_single(Vector\<T, N\>)>`

   * - :ref:`Blend zeros (masked) <operations-blend-zero>`
     - :cpp:func:`grex::blend_zero(Mask mask, Vector v1) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::blend_zero(Mask\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Blend zeros (compile-time selectors) <operations-blend-zero-static>`
     - :cpp:func:`grex::blend_zero\<selectors\>(Vector v1) <template\<BlendZeroSelector... Bzs, Vectorizable T, std::size_t N\> Vector\<T, N\> grex::blend_zero(Vector\<T, N\>)>`

   * - :ref:`Blend (masked) <operations-blend>`
     - :cpp:func:`grex::blend(Mask mask, Vector v0, Vector v1) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::blend(Mask\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Blend (compile-time selectors) <operations-blend-static>`
     - :cpp:func:`grex::blend\<selectors\>(Vector v0, Vector v1) <template\<BlendSelector... Bls, Vectorizable T, std::size_t N\> Vector\<T, N\> grex::blend(Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Shuffle (indexed) <operations-shuffle-dynamic>`
     - :cpp:func:`grex::shuffle(Vector table, Vector idxs) <template\<Vectorizable T, UnsignedIntVectorizable Idx, std::size_t TableSize, std::size_t IdxN\> Vector\<T, IdxN\> grex::shuffle(Vector\<T, TableSize\>, Vector\<Idx, IdxN\>)>`

   * - :ref:`Shuffle (compile-time indices) <operations-shuffle-static>`
     - :cpp:func:`grex::shuffle(Vector table) <template\<ShuffleIndex... I, Vectorizable T, std::size_t N\> Vector\<T, N\> grex::shuffle(Vector\<T, N\>)>`

   * - :ref:`Masked arithmetic <operations-mask-arithmetic>`
     - | :cpp:func:`grex::mask_add(Mask mask, Vector a, Vector b) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::mask_add(Mask\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::mask_subtract(Mask mask, Vector a, Vector b) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::mask_subtract(Mask\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::mask_multiply(Mask mask, Vector a, Vector b) <template\<Vectorizable T, std::size_t N\> Vector\<T, N\> grex::mask_multiply(Mask\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`
       | :cpp:func:`grex::mask_divide(Mask mask, Vector a, Vector b) <template\<FloatVectorizable T, std::size_t N\> Vector\<T, N\> grex::mask_divide(Mask\<T, N\>, Vector\<T, N\>, Vector\<T, N\>)>`

   * - :ref:`Gather <operations-gather>`
     - :cpp:func:`grex::gather(std::span\<const T, extent\> data, Vector indices) <template\<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N\> Vector\<V, N\> grex::gather(std::span\<const V, Extent\>, Vector\<Index, N\>)>`

   * - :ref:`Masked gather <operations-mask-gather>`
     - :cpp:func:`grex::mask_gather(std::span\<const T, extent\> data, Mask mask, Vector indices) <template\<Vectorizable V, std::size_t Extent, Vectorizable Index, std::size_t N\> Vector\<V, N\> grex::mask_gather(std::span\<const V, Extent\>, Mask\<V, N\>, Vector\<Index, N\>)>`
