# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options_sub = subproject('options')
args = options_sub.get_variable('basic_args') + ['-fconstexpr-depth=2048', '-ftemplate-depth=2048']
if meson.get_compiler('cpp').get_id() == 'clang'
  args += ['-fbracket-depth=1024']
endif

test_arches = {}
foreach arch : arches
  test_arches += {arch: [f'-march=@arch@']}
endforeach

# Optional ISA extensions (e.g. AVX512-FP16, only available from Sapphire Rapids on) are not
# implied by any microarchitecture level, so each is tested independently on top of the x86-64-v4
# baseline: combining several into one binary would make it impossible to tell which extension a
# given failure comes from.
# Some extensions never ship without another: no CPU has AVX-512-VBMI2 without AVX-512-VBMI, despite
# them being independent CPUID bits and compiler flags, so testing the former without the latter
# would exercise a combination that does not correspond to any real CPU.
ext_implies = {'avx512vbmi2': ['avx512vbmi']}
if arches.contains('x86-64-v4')
  foreach ext : extensions
    ext_args = []
    foreach name : [ext] + ext_implies.get(ext, [])
      ext_args += [f'-m@name@']
    endforeach
    test_arches += {f'x86-64-v4-@ext@': ['-march=x86-64-v4'] + ext_args}
  endforeach
endif

fmt_sub = subproject('fmt', default_options: ['default_library=static'])
fmt_dep = fmt_sub.get_variable('fmt_dep')
omp_dep = dependency('openmp')
pcg_dep = dependency('pcg-cpp')
thesauros_dep = dependency('thesauros')

# ==============================================================================
# Tests with separated value and index type
# ==============================================================================

dynamic_shuffles = []
foreach name, conf : {
  'shuffle-dynamic': [['x86_64', 'neon'], true],
}
  backends = conf[0]
  parallel = conf[1]
  if backends.contains(backend)
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    foreach value : ['f64', 'f32', 'f16', 'i64', 'i32', 'i16', 'i8', 'u64', 'u32', 'u16', 'u8']
      foreach index : ['u64', 'u32', 'u16', 'u8']
        foreach arch_name, arch_args : test_arches
          space_name = name.replace('-', ' ')
          exe = executable(
            f'@name@-@value@-@index@-@arch_name@',
            [f'@name@.cpp'],
            dependencies: [grex_dep] + deps,
            cpp_args: args
            + arch_args
            + [f'-DGREX_VALUE_TYPE=@value@', f'-DGREX_INDEX_TYPE=@index@'],
          )
          test(
            f'@space_name@ @value@ @index@ @arch_name@',
            exe,
            suite: f'@space_name@ @arch_name@',
            is_parallel: parallel,
            timeout: -1,
          )
          dynamic_shuffles += [[arch_name, exe]]
        endforeach
      endforeach
    endforeach
  endif
endforeach
if dynamic_shuffles.length() > 0
  all_targets = []
  foreach pair : dynamic_shuffles
    all_targets += pair[1]
  endforeach
  alias_target(f'shuffle-dynamic', all_targets)

  foreach arch_name, arch_args : test_arches
    targets = []
    foreach pair : dynamic_shuffles
      if pair[0] == arch_name
        targets += pair[1]
      endif
    endforeach
    alias_target(f'shuffle-dynamic-@arch_name@', targets)
  endforeach
endif

# ==============================================================================
# Tests with separated value type
# ==============================================================================

foreach name, conf : {
  'blend': [['x86_64', 'neon'], true],
  'convert': [['scalar', 'x86_64', 'neon'], true],
  'nary': [['scalar', 'x86_64', 'neon'], true],
  'shuffle': [['x86_64', 'neon'], true],
}
  backends = conf[0]
  parallel = conf[1]
  if backends.contains(backend)
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    targets = []
    foreach type : ['f64', 'f32', 'f16', 'i64', 'i32', 'i16', 'i8', 'u64', 'u32', 'u16', 'u8']
      foreach arch_name, arch_args : test_arches
        space_name = name.replace('-', ' ')
        exe = executable(
          f'@name@-@type@-@arch_name@',
          [f'@name@.cpp'],
          dependencies: [grex_dep] + deps,
          cpp_args: args + arch_args + [f'-DGREX_TEST_TYPE=@type@'],
        )
        test(
          f'@space_name@ @type@ @arch_name@',
          exe,
          suite: space_name,
          is_parallel: parallel,
          timeout: -1,
        )
        targets += [exe]
      endforeach
    endforeach
    alias_target(name, targets)
  endif
endforeach

# ==============================================================================
# Monolithic tests
# ==============================================================================

foreach name, conf : {
  'componentwise': [['scalar', 'x86_64', 'neon'], true],
  'expand': [['x86_64', 'neon'], true],
  'extract': [['scalar', 'x86_64', 'neon'], true],
  'f16': [['scalar', 'x86_64', 'neon'], true],
  'gather': [['scalar', 'x86_64', 'neon'], false],
  'general': [['scalar', 'x86_64', 'neon'], true],
  'horizontal': [['scalar', 'x86_64', 'neon'], true],
  'load': [['scalar', 'x86_64', 'neon'], true],
  'multibyte': [['scalar', 'x86_64', 'neon'], true],
  'set': [['scalar', 'x86_64', 'neon'], true],
  'shingle': [['scalar', 'x86_64', 'neon'], true],
  'store': [['scalar', 'x86_64', 'neon'], true],
}
  backends = conf[0]
  parallel = conf[1]
  if backends.contains(backend)
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    targets = []
    foreach arch_name, arch_args : test_arches
      space_name = name.replace('-', ' ')
      exe = executable(
        f'@name@-@arch_name@',
        [f'@name@.cpp'],
        dependencies: deps,
        cpp_args: args + arch_args,
      )
      test(
        f'@space_name@ @arch_name@',
        exe,
        suite: space_name,
        is_parallel: parallel,
        timeout: -1,
      )
      targets += [exe]
    endforeach
    if targets.length() > 1
      alias_target(name, targets)
    endif
  endif
endforeach
