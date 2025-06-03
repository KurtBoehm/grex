project('grex', 'cpp', default_options: ['cpp_std=c++20', 'warning_level=3'])

deps = [dependency('boost-preprocessor'), dependency('fmt')]

grex_dep = declare_dependency(
  include_directories: include_directories('include'),
  dependencies: deps,
)

install_subdir(
  'include',
  install_dir: 'include',
)

pc = import('pkgconfig')
pc.generate(
  name: 'grex',
  version: meson.project_version(),
  description: 'Generic C++ SIMD',
)

x86_64_levels = run_command(
  import('python').find_installation('python3'),
  files('tools/x86_64_levels.py'),
  check: true,
).stdout().split()

if get_option('test')
  subdir('test')
endif
