cmake_minimum_required(VERSION 3.2)

project(MatrixMultiplicationHPX)

option(DISABLE_BIND_FOR_CIRCLE_CI "Circle CI containers don't support binding" OFF)

if (DISABLE_BIND_FOR_CIRCLE_CI)
  MESSAGE(WARNING "HPX is used with BIND=NONE (meant for Circle CI")
  add_definitions(-DDISABLE_BIND_FOR_CIRCLE_CI)
endif()

find_package(PkgConfig REQUIRED)

find_package(OpenMP REQUIRED)

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${HPX_ROOT}/lib/pkgconfig")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

pkg_search_module(HPX_APPLICATION REQUIRED hpx_application)

include_directories(src)
include_directories(${HPX_APPLICATION_INCLUDE_DIRS})
link_directories(${HPX_APPLICATION_LIBRARY_DIRS})
link_libraries(${HPX_APPLICATION_LIBRARIES} "libhpx_iostreams.so" )

# include_directories(${Vc_ROOT}/include)
# include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-Wno-ignored-attributes ${CMAKE_CXX_FLAGS}")

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES_COMMON
  "src/*.cpp"
  "src/memory_layout/*.cpp"
  "src/reference_kernels/*.cpp"
  "src/util/*.cpp"
  "src/variants/*.cpp"
  "src/variants/components/*.cpp")
message("source files: " ${SOURCES_COMMON})

file(GLOB SOURCES_MATRIX_MULTIPLY_APPLICATION "src/matrix_multiply_application/*.cpp")
file(GLOB SOURCES_TESTS "src/tests/*.cpp")

# file(GLOB_RECURSE SOURCES "src/*.cpp")

set(SOURCES_MATRIX_MULTIPLY ${SOURCES_COMMON} ${SOURCES_MATRIX_MULTIPLY_APPLICATION})
set(SOURCES_TESTS ${SOURCES_COMMON} ${SOURCES_TESTS})

add_executable(matrix_multiply ${SOURCES_MATRIX_MULTIPLY})
target_compile_options(matrix_multiply PUBLIC -std=c++14 -march=native -mtune=native)
target_compile_options(matrix_multiply PUBLIC ${HPX_APPLICATION_CFLAGS} ${OpenMP_CXX_FLAGS})
target_link_libraries(matrix_multiply PUBLIC ${HPX_APPLICATION_LDFLAGS} ${OpenMP_CXX_FLAGS})
INSTALL_TARGETS(/bin matrix_multiply)

add_executable(boost_tests ${SOURCES_TESTS})
target_compile_options(boost_tests PUBLIC -std=c++14 -march=native -mtune=native)
target_compile_options(boost_tests PUBLIC ${HPX_APPLICATION_CFLAGS} ${OpenMP_CXX_FLAGS})
target_link_libraries(boost_tests PUBLIC ${HPX_APPLICATION_LDFLAGS} ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
INSTALL_TARGETS(/bin boost_tests)
