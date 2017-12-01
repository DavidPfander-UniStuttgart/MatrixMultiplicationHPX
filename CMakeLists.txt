cmake_minimum_required(VERSION 3.5)

project(MatrixMultiplication)
set(MatrixMultiplication_VERSION 0.5.0)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework)
find_package(AutoTuneTMP NO_MODULE REQUIRED PATHS ${AUTOTUNETMP_ROOT})

find_package(Vc ${Vc_FIND_VERSION} NO_MODULE PATHS ${Vc_ROOT})
include_directories(${Vc_INCLUDE_DIR})

set(SOURCES_COMMON "src/variants/kernel_test.cpp" "src/variants/kernel_tiled.cpp" "src/variants/combined.cpp")

set(SOURCES_MATRIX_MULTIPLY ${SOURCES_COMMON} "src/matrix_multiply_application/matrix_multiply.cpp")
add_executable(matrix_multiply ${SOURCES_MATRIX_MULTIPLY})
target_compile_options(matrix_multiply PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
target_include_directories(matrix_multiply PRIVATE src)
target_link_libraries(matrix_multiply PUBLIC ${OpenMP_CXX_FLAGS} AutoTuneTMP::AutoTuneTMP Boost::boost Boost::program_options)

set(SOURCES_TESTS "src/tests/test_iterate_tiles.cpp" "src/tests/test_naive.cpp" "src/tests/test_combined.cpp" "src/tests/tests.cpp" "src/tests/test_tile_view.cpp")
# set(SOURCES_TESTS ${SOURCES_COMMON} ${SOURCES_TESTS})
add_executable(boost_tests ${SOURCES_COMMON} ${SOURCES_TESTS})
target_compile_options(boost_tests PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
target_include_directories(boost_tests PRIVATE src)
target_link_libraries(boost_tests PUBLIC ${OpenMP_CXX_FLAGS} AutoTuneTMP::AutoTuneTMP Boost::boost Boost::unit_test_framework)
# INSTALL_TARGETS(/bin boost_tests)

set(SOURCES_COMBINED_TUNING ${SOURCES_COMMON} "src/combined_tuning_application/combined_tuning.cpp")
add_executable(combined_tuning ${SOURCES_COMBINED_TUNING})
target_compile_options(combined_tuning PUBLIC ${OpenMP_CXX_FLAGS} -std=c++1z -march=native -mtune=native)
target_include_directories(combined_tuning PRIVATE src)
target_link_libraries(combined_tuning PUBLIC ${OpenMP_CXX_FLAGS} AutoTuneTMP::AutoTuneTMP Boost::boost Boost::program_options)
install(TARGETS matrix_multiply combined_tuning RUNTIME DESTINATION bin)
