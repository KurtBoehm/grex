#ifndef INCLUDE_GREX_BACKEND_ACTIVE_SIZES_HPP
#define INCLUDE_GREX_BACKEND_ACTIVE_SIZES_HPP

// IWYU pragma: begin_exports
#if GREX_BACKEND_X86_64
#include "grex/backend/x86/sizes.hpp"
#elif GREX_BACKEND_NEON
#include "grex/backend/neon/sizes.hpp"
#elif GREX_BACKEND_SCALAR
#include "grex/backend/scalar/sizes.hpp"
#endif
// IWYU pragma: end_exports

#endif // INCLUDE_GREX_BACKEND_ACTIVE_SIZES_HPP
