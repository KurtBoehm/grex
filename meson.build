# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project(
  'grex',
  'cpp',
  default_options: ['cpp_std=c++23', 'warning_level=3'],
  meson_version: '>=1.5.0',
)

backend = get_option('simd_backend')
if backend == 'auto'
  if target_machine.cpu_family() == 'x86_64'
    backend = 'x86_64'
  elif target_machine.cpu_family() == 'aarch64'
    backend = 'neon'
  else
    backend = 'scalar'
  endif
endif

grex_args = []
if backend == 'x86_64'
  grex_args += ['-DGREX_BACKEND_X86_64=1']
elif backend == 'neon'
  grex_args += ['-DGREX_BACKEND_NEON=1']
elif backend == 'scalar'
  grex_args += ['-DGREX_BACKEND_SCALAR=1']
else
  error(f'Unsupported backend: @backend@')
endif
grex_dep = declare_dependency(
  include_directories: include_directories('include'),
  compile_args: grex_args,
)

if not meson.is_subproject()
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
endif

if get_option('build_tests')
  if backend == 'x86_64'
    cpp = meson.get_compiler('cpp')
    cpuid = import('fs').read('tools/cpuid.cpp')
    arches = cpp.run(
      cpuid,
      include_directories: include_directories('include'),
      required: true,
    ).stdout().split(';')
  else
    arches = []
  endif

  subdir('test')
endif
