options_sub = subproject('options')
args = options_sub.get_variable('all_args')

foreach name, deps : {'basic': []}
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
foreach name : ['set']
  foreach type : ['f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']
    foreach size : [2, 4, 8, 16, 32, 64]
      space_name = name.replace('-', ' ')
      test(
        f'test @space_name@ @type@x@size@',
        executable(
          f'test-@name@-@type@-@size@',
          [f'@name@.cpp'],
          cpp_args: args + [f'-DGREX_TEST_TYPE=@type@', f'-DGREX_TEST_SIZE=@size@'],
          dependencies: [grex_dep] + deps,
        ),
      )
    endforeach
  endforeach
endforeach
