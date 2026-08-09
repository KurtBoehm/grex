#!/bin/sh
# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Meson `exe_wrapper` for cross/sapphirerapids-sde.ini: runs a built binary under Intel SDE
# emulating Sapphire Rapids (`-spr`), rather than requiring actual Sapphire Rapids hardware. Meson
# invokes this both for configure-time checks that run compiled probes (e.g. the
# microarchitecture-level/extension detection in tools/cpuid.cpp) and for `meson test`.
#
# The SDE binary is looked up as `sde64` on PATH by default; set GREX_SDE_PATH to override, e.g.:
#   export GREX_SDE_PATH=/opt/intel/sde/sde64
exec "${GREX_SDE_PATH:-sde64}" -spr -- "$@"
