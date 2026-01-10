# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

benchmark_tests = {
  'benchmark_test': '--benchmark_min_time=0.01',
  'spec_arg_test': '--benchmark_filter=BM_NotChosen',
  'spec_arg_verbosity_test': '--v=42',
  'benchmark_setup_teardown_test': [],
  'options_test': '--benchmark_min_time=0.01',
  'basic_test': '--benchmark_min_time=0.01',
  'donotoptimize_test': '--benchmark_min_time=0.01',
  'fixture_test': '--benchmark_min_time=0.01',
  'map_test': '--benchmark_min_time=0.01',
  'multiple_ranges_test': '--benchmark_min_time=0.01',
  'args_product_test': '--benchmark_min_time=0.01',
}

foreach t, a : benchmark_tests
  test(
    t,
    executable(t, f'@t@.cc', dependencies: google_benchmark_dep),
    args: a,
  )
endforeach
