// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP
#define INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP

// reinterpret-casts between signed integer vectors and masks → only defined for levels below 4
#include "grex/backend/x86/instruction-sets.hpp"

#if GREX_X86_64_LEVEL < 4
#include "grex/backend/shared/operations/mask-convert.hpp" // IWYU pragma: export
#endif

#endif // INCLUDE_GREX_BACKEND_X86_OPERATIONS_MASK_CONVERT_HPP
