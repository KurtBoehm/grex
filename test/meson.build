options_sub = subproject('options')
args = options_sub.get_variable('all_args')

arches = x86_64_levels + ['native']

foreach name, deps : {'basic': [], 'componentwise': [], 'set': []}
  foreach arch : arches
    space_name = name.replace('-', ' ')
    exe = executable(
      f'test-@name@-@arch@',
      [f'@name@.cpp'],
      cpp_args: args + [f'-march=@arch@'],
      dependencies: [grex_dep] + deps,
    )
    test(f'test @space_name@ @arch@', exe)
  endforeach
endforeach
