# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project(
  'benchmark',
  'cpp',
  version: '1.9.5',
  default_options: ['cpp_std=c++17'],
  license: 'Apache-2.0',
)

benchmark_sources = files(
  'src/benchmark.cc',
  'src/benchmark_api_internal.cc',
  'src/benchmark_main.cc',
  'src/benchmark_name.cc',
  'src/benchmark_register.cc',
  'src/benchmark_runner.cc',
  'src/check.cc',
  'src/colorprint.cc',
  'src/commandlineflags.cc',
  'src/complexity.cc',
  'src/console_reporter.cc',
  'src/counter.cc',
  'src/csv_reporter.cc',
  'src/json_reporter.cc',
  'src/perf_counters.cc',
  'src/reporter.cc',
  'src/statistics.cc',
  'src/string_util.cc',
  'src/sysinfo.cc',
  'src/timers.cc',
)

benchmark_main_sources = files('src/benchmark_main.cc')
inc = include_directories('include')

cpp_args = ['-DNDEBUG']
if get_option('default_library') != 'static'
  cpp_args += '-Dbenchmark_EXPORTS'
else
  cpp_args += '-DBENCHMARK_STATIC_DEFINE'
endif

lib_benchmark_deps = [dependency('threads')]

if host_machine.system() == 'windows'
  cpp = meson.get_compiler('cpp')

  if cpp.get_argument_syntax() == 'msvc'
    add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', language: 'cpp')
  endif

  lib_benchmark_deps += cpp.find_library('shlwapi', required: true)
endif

lib_benchmark = library(
  'gbenchmark',
  benchmark_sources,
  include_directories: inc,
  dependencies: lib_benchmark_deps,
  cpp_args: cpp_args,
  gnu_symbol_visibility: 'hidden',
)

dep_cpp_args = []
if get_option('default_library') == 'static'
  dep_cpp_args += '-DBENCHMARK_STATIC_DEFINE'
endif

google_benchmark_dep = declare_dependency(
  include_directories: inc,
  link_with: lib_benchmark,
  compile_args: dep_cpp_args,
)

lib_benchmark_main = library(
  'benchmark-main',
  benchmark_main_sources,
  include_directories: inc,
  dependencies: google_benchmark_dep,
  cpp_args: cpp_args,
)

google_benchmark_main_dep = declare_dependency(
  dependencies: google_benchmark_dep,
  link_with: lib_benchmark_main,
  compile_args: dep_cpp_args,
)

if not get_option('tests').disabled()
  subdir('test')
endif
