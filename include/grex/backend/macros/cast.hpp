// This file is part of https://github.com/KurtBoehm/grex.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_GREX_BACKEND_MACROS_CAST_HPP
#define INCLUDE_GREX_BACKEND_MACROS_CAST_HPP

#define GREX_OPCAST_i8(KIND, BITS, X) KIND##BITS(X)
#define GREX_OPCAST_i16(KIND, BITS, X) KIND##BITS(X)
#define GREX_OPCAST_i32(KIND, BITS, X) X
#define GREX_OPCAST_i64(KIND, BITS, X) X
#define GREX_OPCAST_f32(KIND, BITS, X) X
#define GREX_OPCAST_f64(KIND, BITS, X) X
#define GREX_OPCAST_f(KIND, BITS, X) GREX_OPCAST_f##BITS(KIND, BITS, X)
#define GREX_OPCAST_i(KIND, BITS, X) GREX_OPCAST_i##BITS(KIND, BITS, X)
#define GREX_OPCAST_u(KIND, BITS, X) GREX_OPCAST_i##BITS(KIND, BITS, X)
#define GREX_OPCAST(KIND, BITS, X) GREX_OPCAST_##KIND(KIND, BITS, X)

#endif // INCLUDE_GREX_BACKEND_MACROS_CAST_HPP
