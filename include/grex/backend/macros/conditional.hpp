// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_CONDITIONAL_HPP
#define INCLUDE_GREX_BACKEND_MACROS_CONDITIONAL_HPP

#include "grex/backend/macros/bool-conversion.hpp"

#define GREX_IF_0(TRUE, FALSE) FALSE
#define GREX_IF_1(TRUE, FALSE) TRUE
#define GREX_IF_II(COND, TRUE, FALSE) GREX_IF_##COND(TRUE, FALSE)
#define GREX_IF_I(COND, TRUE, FALSE) GREX_IF_II(COND, TRUE, FALSE)
#define GREX_IF(COND, TRUE, FALSE) GREX_IF_I(GREX_INT2BOOL(COND), TRUE, FALSE)
#define GREX_COMMA_IF(COND) GREX_IF(COND, GREX_COMMA, GREX_EMPTY)()

#endif // INCLUDE_GREX_BACKEND_MACROS_CONDITIONAL_HPP
