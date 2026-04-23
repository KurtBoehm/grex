#ifndef INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_SET_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_SET_HPP

#include "grex/backend/defs.hpp" // IWYU pragma: keep

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86/operations/set.hpp"
#elif GREX_BACKEND_NEON
#include "grex/backend/neon/operations/set.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_SET_HPP
