options_sub = subproject('options')
opts = options_sub.get_variable('all_opts')
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
      override_options: opts,
      cpp_args: args,
      dependencies: [grex_dep] + deps,
    ),
  )
endforeach
