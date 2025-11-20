# This file is part of https://github.com/KurtBoehm/grex.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.20)
project(grex LANGUAGES CXX)

# Build options
option(GREX_BUILD_TESTS "Build tests" OFF)
set(GREX_SIMD_BACKEND "auto" CACHE STRING "SIMD backend (auto, x86_64, scalar)")

if(GREX_SIMD_BACKEND STREQUAL "auto")
  include(CheckCXXSourceCompiles)
  check_cxx_source_compiles(
    "#if !defined(__x86_64__) && !defined(_M_X64)\n#error\n#endif\nint main() {}"
    grex_is_x86_64
  )
  check_cxx_source_compiles(
    "#if !defined(__aarch64__) && !defined(_M_ARM64)\n#error\n#endif\nint main() {}"
    grex_is_arm64
  )
  if(grex_is_x86_64)
    set(grex_backend "x86_64")
  elseif(grex_is_arm64)
    set(grex_backend "neon")
  else()
    set(grex_backend "scalar")
  endif()
else()
  set(grex_backend ${GREX_SIMD_BACKEND})
endif()

include(FetchContent)

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)
if(grex_backend STREQUAL "x86_64")
  target_compile_definitions(${PROJECT_NAME} INTERFACE "GREX_BACKEND_X86_64=1")
elseif(grex_backend STREQUAL "neon")
  target_compile_definitions(${PROJECT_NAME} INTERFACE "GREX_BACKEND_NEON=1")
elseif(grex_backend STREQUAL "scalar")
  target_compile_definitions(${PROJECT_NAME} INTERFACE "GREX_BACKEND_SCALAR=1")
else()
  message(FATAL_ERROR "Unsupported backend: ${grex_backend}")
endif()

target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(GREX_BUILD_TESTS)
  # Fetch cmop to add functions that add target options
  FetchContent_Declare(
    cmop
    GIT_REPOSITORY https://github.com/KurtBoehm/cmop.git
    GIT_TAG main
  )
  FetchContent_MakeAvailable(cmop)

  find_package(OpenMP COMPONENTS CXX)

  FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/archive/12.0.0.tar.gz
    URL_HASH
      SHA256=aa3e8fbb6a0066c03454434add1f1fc23299e85758ceec0d7d2d974431481e40
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(fmt)

  # pcg-cpp does not provide CMake build files
  FetchContent_Declare(
    pcg_cpp
    GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
    GIT_TAG 428802d1a5634f96bcd0705fab379ff0113bcf13
  )
  FetchContent_MakeAvailable(pcg_cpp)

  FetchContent_Declare(
    thesauros
    GIT_REPOSITORY https://github.com/KurtBoehm/thesauros.git
    GIT_TAG main
  )
  FetchContent_MakeAvailable(thesauros)

  if(grex_backend STREQUAL "x86_64")
    try_run(
      cpuid_run_res
      cpuid_compile_res
      SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tools/cpuid.cpp"
      CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_CURRENT_SOURCE_DIR}/include"
      RUN_OUTPUT_VARIABLE cpuid_output
    )
    if(NOT cpuid_compile_res)
      message(FATAL_ERROR "CPUID failed!")
    endif()

    set(grex_arches ${cpuid_output})
  endif()

  include(CTest)

  function(create_grex_test test_target src_name test_name)
    add_executable(${test_target} test/${src_name}.cpp)
    target_compile_features(${test_target} PRIVATE cxx_std_23)
    target_link_libraries(
      ${test_target}
      PRIVATE ${PROJECT_NAME} OpenMP::OpenMP_CXX thesauros::thesauros
      # fmt::fmt is part of thesauros::thesauros already, and linking it twice
      # (which CMake does not optimize away) leads to warnings with Apple Clang.
    )
    target_include_directories(
      ${test_target}
      PRIVATE ${pcg_cpp_SOURCE_DIR}/include
    )
    cmop_add_warnings(${test_target})
    cmop_add_info(${test_target})
    add_test(NAME ${test_name} COMMAND ${test_target})
  endfunction()

  # tests that are separated based on the value type
  foreach(
    test_info
    IN
    ITEMS "blend;x86_64;neon" "convert;scalar;x86_64;neon" "shuffle;x86_64;neon"
  )
    list(GET test_info 0 test_name)
    list(SUBLIST test_info 1 -1 test_backends)
    list(FIND test_backends ${grex_backend} grex_backend_index)

    if(${grex_backend_index} GREATER_EQUAL 0)
      foreach(
        test_type
        f64
        f32
        i64
        i32
        i16
        i8
        u64
        u32
        u16
        u8
      )
        foreach(test_arch ${grex_arches})
          set(
            test_target_name
            "${PROJECT_NAME}_${test_name}_${test_type}_${test_arch}"
          )
          create_grex_test(${test_target_name} ${test_name} "${PROJECT_NAME}.${test_name}.${test_type}.${test_arch}")
          target_compile_definitions(
            ${test_target_name}
            PRIVATE "GREX_TEST_TYPE=${test_type}"
          )
          target_compile_options(
            ${test_target_name}
            PRIVATE "-march=${test_arch}"
          )
        endforeach()

        set(test_target_name "${PROJECT_NAME}_${test_name}_${test_type}_native")
        create_grex_test(${test_target_name} ${test_name} "${PROJECT_NAME}.${test_name}.${test_type}.native")
        target_compile_definitions(
          ${test_target_name}
          PRIVATE "GREX_TEST_TYPE=${test_type}"
        )
        cmop_add_optimization(${test_target_name})
      endforeach()
    endif()
  endforeach()

  foreach(
    test_info
    IN
    ITEMS
      "componentwise;scalar;x86_64;neon"
      "expand;x86_64;neon"
      "extract;scalar;x86_64;neon"
      "gather;scalar;x86_64;neon"
      "general;scalar;x86_64;neon"
      "horizontal;scalar;x86_64;neon"
      "mem;scalar;x86_64;neon"
      "multibyte;scalar;x86_64;neon"
      "set;scalar;x86_64;neon"
      "shingle;scalar;x86_64;neon"
  )
    list(GET test_info 0 test_name)
    list(SUBLIST test_info 1 -1 test_backends)
    list(FIND test_backends ${grex_backend} grex_backend_index)

    if(${grex_backend_index} GREATER_EQUAL 0)
      foreach(test_arch ${grex_arches})
        set(test_target_name "${PROJECT_NAME}_${test_name}_${test_arch}")
        create_grex_test(${test_target_name} ${test_name} "${PROJECT_NAME}.${test_name}.${test_arch}")
        target_compile_options(
          ${test_target_name}
          PRIVATE "-march=${test_arch}"
        )
      endforeach()

      set(test_target_name "${PROJECT_NAME}_${test_name}_native")
      create_grex_test(${test_target_name} ${test_name} "${PROJECT_NAME}.${test_name}.native")
      cmop_add_optimization(${test_target_name})
    endif()
  endforeach()
endif()
