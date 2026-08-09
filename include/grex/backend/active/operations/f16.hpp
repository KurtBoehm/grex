#ifndef INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_F16_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_F16_HPP

#include "grex/backend/defs.hpp" // IWYU pragma: keep

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86/operations/f16.hpp"
#elif GREX_BACKEND_NEON
#include "grex/backend/neon/operations/f16.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_OPERATIONS_F16_HPP
