#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_HPP

// IWYU pragma: begin_exports
#include "operations/abs.hpp"
#include "operations/arithmetic-mask.hpp"
#include "operations/arithmetic.hpp"
#include "operations/bitwise.hpp"
#include "operations/blend.hpp"
#include "operations/classification.hpp"
#include "operations/compare.hpp"
#include "operations/convert.hpp"
#include "operations/expand-scalar.hpp"
#include "operations/expand-vector.hpp"
#include "operations/extract.hpp"
#include "operations/fmadd-family.hpp"
#include "operations/gather.hpp"
#include "operations/horizontal-add.hpp"
#include "operations/horizontal-and.hpp"
#include "operations/horizontal-minmax.hpp"
#include "operations/insert.hpp"
#include "operations/load.hpp"
#include "operations/mask-index.hpp"
#include "operations/merge.hpp"
#include "operations/minmax.hpp"
#include "operations/multibyte.hpp"
#include "operations/set.hpp"
#include "operations/shingle.hpp"
#include "operations/split.hpp"
#include "operations/store.hpp"
#include "operations/subnative.hpp"
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_HPP
