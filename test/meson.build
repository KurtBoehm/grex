options_sub = subproject('options')
args = options_sub.get_variable('all_args')

foreach name, deps : {
  'basic': []
}
  space_name = name.replace('-', ' ')
  test(
    f'test @space_name@',
    executable(
      f'test-@name@',
      [f'@name@.cpp'],
      cpp_args: args,
      dependencies: [grex_dep] + deps,
    ),
  )
endforeach
