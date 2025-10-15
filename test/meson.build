options_sub = subproject('options')
args = options_sub.get_variable('all_args')

test_arches = arches + ['native']
fmt_dep = dependency('fmt')
omp_dep = dependency('openmp')
pcg_dep = dependency('pcg-cpp')
thesauros_dep = dependency('thesauros')

# tests that are split based on the value type
foreach name, conf : {
  'blend': [true, true],
  'convert': [false, true],
  'shuffle': [true, true],
}
  simd_only = conf[0]
  parallel = conf[1]
  if not simd_only or backend != 'scalar'
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    foreach type : ['f64', 'f32', 'i64', 'i32', 'i16', 'i8', 'u64', 'u32', 'u16', 'u8']
      foreach arch : test_arches
        space_name = name.replace('-', ' ')
        exe = executable(
          f'test-@name@-@type@-@arch@',
          [f'@name@.cpp'],
          cpp_args: args
          + [
            f'-march=@arch@',
            f'-DGREX_TEST_TYPE=@type@',
          ],
          dependencies: [grex_dep] + deps,
        )
        test(
          f'test @space_name@ @type@ @arch@',
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
  'componentwise': [false, true],
  'expand': [true, true],
  'extract': [false, true],
  'gather': [false, false],
  'general': [false, true],
  'horizontal': [false, true],
  'mem': [false, true],
  'multibyte': [false, true],
  'set': [false, true],
  'shingle': [false, true],
}
  simd_only = conf[0]
  parallel = conf[1]
  if not simd_only or backend != 'scalar'
    deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
    if not parallel
      deps += [omp_dep]
    endif
    foreach arch : test_arches
      space_name = name.replace('-', ' ')
      exe = executable(
        f'test-@name@-@arch@',
        [f'@name@.cpp'],
        cpp_args: args + [f'-march=@arch@'],
        dependencies: deps,
      )
      test(
        f'test @space_name@ @arch@',
        exe,
        suite: f'test @space_name@',
        is_parallel: parallel,
        timeout: -1,
      )
    endforeach
  endif
endforeach
