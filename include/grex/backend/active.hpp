#ifndef INCLUDE_GREX_BACKEND_ACTIVE_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_HPP

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86.hpp"
#elif GREX_BACKEND_SCALAR
#include "grex/backend/scalar.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_HPP
