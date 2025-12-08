#ifndef INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_HPP

#include "grex/backend/defs.hpp" // IWYU pragma: keep

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86/operations.hpp"
#elif GREX_BACKEND_NEON
#include "grex/backend/neon/operations.hpp"
#elif GREX_BACKEND_SCALAR
#include "grex/backend/scalar/operations.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_HPP
