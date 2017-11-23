cmake_minimum_required(VERSION 3.5)

project(MatrixMultiplication)

option(DISABLE_BIND_FOR_CIRCLE_CI "Circle CI containers don't support binding" OFF)

# find_package(PkgConfig REQUIRED)

find_package(OpenMP REQUIRED)

find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)


include_directories(src)
# ${AUTOTUNETMP_INCLUDE}
# include_directories(${AutoTuneTMP_ROOT}/src ${AutoTuneTMP_ROOT}/dependencies/cppjit/include)
# include_directories(${Vc_ROOT}/include)

message(WARNING "AutoTuneTMP_ROOT: ${AUTOTUNETMP_ROOT}")
find_package(AutoTuneTMP NO_MODULE REQUIRED PATHS ${AUTOTUNETMP_ROOT})

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(CMAKE_CXX_FLAGS "-Wno-ignored-attributes ${CMAKE_CXX_FLAGS}")

#However, the file(GLOB...) allows for wildcard additions:
file(GLOB SOURCES_COMMON
  "src/*.cpp"
  "src/memory_layout/*.cpp"
  "src/reference_kernels/*.cpp"
  "src/util/*.cpp"
  "src/variants/*.cpp"
  )
message("source files: " ${SOURCES_COMMON})

file(GLOB SOURCES_MATRIX_MULTIPLY_APPLICATION "src/matrix_multiply_application/*.cpp")
file(GLOB SOURCES_TESTS "src/tests/*.cpp")

# file(GLOB_RECURSE SOURCES "src/*.cpp")

set(SOURCES_MATRIX_MULTIPLY ${SOURCES_COMMON} ${SOURCES_MATRIX_MULTIPLY_APPLICATION})
set(SOURCES_TESTS ${SOURCES_COMMON} ${SOURCES_TESTS})

add_executable(matrix_multiply ${SOURCES_MATRIX_MULTIPLY})
target_compile_options(matrix_multiply PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
target_link_libraries(matrix_multiply PUBLIC ${OpenMP_CXX_FLAGS} dl boost_program_options AutoTuneTMP::AutoTuneTMP)
INSTALL_TARGETS(/bin matrix_multiply)

# add_executable(boost_tests ${SOURCES_TESTS})
# target_compile_options(boost_tests PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
# target_link_libraries(boost_tests PUBLIC ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
# INSTALL_TARGETS(/bin boost_tests)

set(SOURCES_COMBINED_TUNING ${SOURCES_COMMON} "src/combined_tuning_application/combined_tuning.cpp")
add_executable(combined_tuning ${SOURCES_COMBINED_TUNING})
target_compile_options(combined_tuning PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
target_link_libraries(combined_tuning PUBLIC ${OpenMP_CXX_FLAGS} dl boost_program_options  AutoTuneTMP::AutoTuneTMP)
INSTALL_TARGETS(/bin combined_tuning)
