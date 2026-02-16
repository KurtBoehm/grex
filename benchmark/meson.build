# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options_sub = subproject('options')
args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
pcg_dep = dependency('pcg-cpp')

if backend == 'neon'
  foreach name : ['load-part', 'store-part']
    executable(
      name,
      f'neon/@name@.cpp',
      cpp_args: args,
      dependencies: [benchmark_dep, grex_dep, pcg_dep],
    )
  endforeach
endif
