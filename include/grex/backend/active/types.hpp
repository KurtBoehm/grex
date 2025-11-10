#ifndef INCLUDE_GREX_BACKEND_ACTIVE_TYPES_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_TYPES_HPP

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86/types.hpp"
#elif GREX_BACKEND_NEON
#include "grex/backend/neon/types.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_TYPES_HPP
