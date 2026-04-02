// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_NEON_MACROS_CAST_HPP
#define INCLUDE_GREX_BACKEND_NEON_MACROS_CAST_HPP

#include <bit> // IWYU pragma: keep

#define GREX_KINDCAST_ii(BITS, X) X
#define GREX_KINDCAST_iu(BITS, X) std::bit_cast<i##BITS>(X)
#define GREX_KINDCAST_ui(BITS, X) std::bit_cast<u##BITS>(X)
#define GREX_KINDCAST_uu(BITS, X) X
#define GREX_KINDCAST(DSTKIND, SRCKIND, BITS, X) GREX_KINDCAST_##DSTKIND##SRCKIND(BITS, X)

#endif // INCLUDE_GREX_BACKEND_NEON_MACROS_CAST_HPP
