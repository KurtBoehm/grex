# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options_sub = subproject('options')
args = options_sub.get_variable('basic_args')

test_arches = {}
foreach arch : arches
  test_arches += {arch: [f'-march=@arch@']}
endforeach
test_arches += {'native': options_sub.get_variable('optimization_args')}

fmt_sub = subproject('fmt', default_options: ['default_library=static'])
fmt_dep = fmt_sub.get_variable('fmt_dep')
omp_dep = dependency('openmp')
pcg_dep = dependency('pcg-cpp')
thesauros_dep = dependency('thesauros')

# tests that are separated based on the value type
foreach name, conf : {
  'blend': [['x86_64', 'neon'], true],
  'convert': [['scalar', 'x86_64', 'neon'], true],
  'shuffle': [['x86_64'], true], # neon
}
  backends = conf[0]
  parallel = conf[1]
  if backends.contains(backend)
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    foreach type : ['f64', 'f32', 'i64', 'i32', 'i16', 'i8', 'u64', 'u32', 'u16', 'u8']
      foreach arch_name, arch_args : test_arches
        space_name = name.replace('-', ' ')
        exe = executable(
          f'test-@name@-@type@-@arch_name@',
          [f'@name@.cpp'],
          cpp_args: args + arch_args + [f'-DGREX_TEST_TYPE=@type@'],
          dependencies: [grex_dep] + deps,
        )
        test(
          f'test @space_name@ @type@ @arch_name@',
          exe,
          suite: f'test @space_name@',
          is_parallel: parallel,
          timeout: -1,
        )
      endforeach
    endforeach
  endif
endforeach

# monolithic tests
foreach name, conf : {
  'componentwise': [['scalar', 'x86_64', 'neon'], true],
  'expand': [['x86_64', 'neon'], true],
  'extract': [['scalar', 'x86_64', 'neon'], true],
  'gather': [['scalar', 'x86_64', 'neon'], false],
  'general': [['scalar', 'x86_64', 'neon'], true],
  'horizontal': [['scalar', 'x86_64', 'neon'], true],
  'mem': [['scalar', 'x86_64', 'neon'], true],
  'multibyte': [['scalar', 'x86_64', 'neon'], true],
  'set': [['scalar', 'x86_64', 'neon'], true],
  'shingle': [['scalar', 'x86_64', 'neon'], true],
}
  backends = conf[0]
  parallel = conf[1]
  if backends.contains(backend)
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    foreach arch_name, arch_args : test_arches
      space_name = name.replace('-', ' ')
      exe = executable(
        f'test-@name@-@arch_name@',
        [f'@name@.cpp'],
        cpp_args: args + arch_args,
        dependencies: deps,
      )
      test(
        f'test @space_name@ @arch_name@',
        exe,
        suite: f'test @space_name@',
        is_parallel: parallel,
        timeout: -1,
      )
    endforeach
  endif
endforeach
