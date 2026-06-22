# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options_sub = subproject('options')
basic_args = options_sub.get_variable('basic_args')
all_args = options_sub.get_variable('all_args')

benchmark_dep = dependency('benchmark')
pcg_dep = dependency('pcg-cpp')

if backend == 'neon'
  foreach name : ['load-part', 'store-part']
    executable(
      f'bm-@name@',
      f'neon/@name@.cpp',
      dependencies: [benchmark_dep, grex_dep, pcg_dep],
      cpp_args: all_args,
    )
  endforeach
endif

if backend == 'x86_64'
  bm_arches = {}
  foreach arch : arches
    bm_arches += {arch: [f'-march=@arch@']}
  endforeach
  bm_arches += {'native': options_sub.get_variable('optimization_args')}

  # Only native
  foreach name : ['multibyte']
    executable(
      f'bm-@name@',
      f'x86/@name@.cpp',
      dependencies: [benchmark_dep, grex_dep, pcg_dep],
      cpp_args: all_args,
    )
  endforeach

  # All micro-architecture levels
  foreach name : ['load-part']
    targets = []
    foreach arch_name, arch_args : test_arches
      exe = executable(
        f'bm-@name@-@arch_name@',
        f'x86/@name@.cpp',
        dependencies: [benchmark_dep, grex_dep, pcg_dep],
        cpp_args: basic_args + arch_args,
      )
      targets += exe
    endforeach
    if targets.length() > 1
      alias_target(f'bm-@name@', targets)
    endif
  endforeach
endif
