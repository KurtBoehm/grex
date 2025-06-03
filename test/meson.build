options_sub = subproject('options')
args = options_sub.get_variable('all_args')

arches = x86_64_levels + ['native']

foreach name : ['basic', 'componentwise', 'horizontal', 'mem', 'set']
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
