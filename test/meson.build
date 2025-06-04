options_sub = subproject('options')
args = options_sub.get_variable('all_args')
if ['gcc'].contains(options_sub.get_variable('compiler_id'))
  args += ['-Wno-useless-cast']
endif

arches = x86_64_levels + ['native']

foreach name : ['componentwise', 'horizontal', 'mem', 'set']
  foreach arch : arches
    space_name = name.replace('-', ' ')
    exe = executable(
      f'test-@name@-@arch@',
      [f'@name@.cpp'],
      cpp_args: args + [f'-march=@arch@'],
      dependencies: [grex_dep],
    )
    test(f'test @space_name@ @arch@', exe)
  endforeach
endforeach
