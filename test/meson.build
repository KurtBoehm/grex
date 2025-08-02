options_sub = subproject('options')
args = options_sub.get_variable('all_args')
if ['gcc'].contains(options_sub.get_variable('compiler_id'))
  args += ['-Wno-useless-cast']
endif

arches = x86_64_levels + ['native']
fmt_dep = dependency('fmt')
omp_dep = dependency('openmp')
pcg_dep = dependency('pcg-cpp')
thesauros_dep = dependency('thesauros')

foreach name, parallel : {
  'componentwise': true,
  'convert': true,
  'expand': true,
  'gather': false,
  'general': false,
  'horizontal': true,
  'mem': true,
  'multibyte': true,
  'set': true,
  'shingle': true,
  'shuffle': true,
}
  deps = [fmt_dep, grex_dep, pcg_dep, thesauros_dep]
  if not parallel
    deps += [omp_dep]
  endif
  foreach arch : arches
    space_name = name.replace('-', ' ')
    exe = executable(
      f'test-@name@-@arch@',
      [f'@name@.cpp'],
      cpp_args: args + [f'-march=@arch@'],
      dependencies: deps,
    )
    test(f'test @space_name@ @arch@', exe, is_parallel: parallel, timeout: -1)
  endforeach
endforeach
