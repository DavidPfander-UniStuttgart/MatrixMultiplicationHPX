#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "memory_layout/tile_array.hpp"
#include "memory_layout/tile_view.hpp"

template <size_t dim>
void compare_tile(std::vector<double> &tiled_matrix,
                  std::vector<memory_layout::tiling_info_dim> &tiling_info,
                  size_t (&tile_index)[dim], std::vector<double> reference) {
  memory_layout::tile_view<2, double, std::allocator<double>> view(tiled_matrix, tile_index, tiling_info);

  for (size_t i = 0; i < tiling_info[0].tile_size_dir; i++) {
    for (size_t j = 0; j < tiling_info[1].tile_size_dir; j++) {
      BOOST_CHECK_CLOSE(view(i, j), reference[i * 4 + j], 1E-15);
    }
  }
}

BOOST_AUTO_TEST_SUITE(test_tile_view)

BOOST_AUTO_TEST_CASE(view) {
  const size_t N = 256;
  const size_t untiled_stride = 256;
  const size_t tile_size = 4;
  std::vector<double> m(N * N);
  std::for_each(m.begin(), m.end(), [](double &element) {
      static int counter = 0;
      element = counter;
      counter++;
    });

  std::vector<memory_layout::tiling_info_dim> tiling_info(2);
  tiling_info[0].tile_size_dir = tile_size;
  tiling_info[0].stride = untiled_stride;
  tiling_info[1].tile_size_dir = tile_size;
  tiling_info[1].stride = untiled_stride;

  std::vector<double> tiled_matrix = memory_layout::make_tiled<2>(m, tiling_info);

  // view on first tile
  size_t tile_index[] = {0, 0};

  std::vector<double> reference_first = {0,   1,   2,   3,   256, 257, 258, 259,
                                         512, 513, 514, 515, 768, 769, 770, 771};

  compare_tile<2>(tiled_matrix, tiling_info, tile_index, reference_first);

  // view tile at end of first rows
  tile_index[0] = 0;
  tile_index[1] = (untiled_stride / tile_size) - 1;

  std::vector<double> reference_first_end = {252, 253, 254, 255, 508,  509,  510,  511,
                                             764, 765, 766, 767, 1020, 1021, 1022, 1023};

  compare_tile<2>(tiled_matrix, tiling_info, tile_index, reference_first_end);

  // view tile at end of first cols
  tile_index[0] = (untiled_stride / tile_size) - 1;
  tile_index[1] = 0;

  std::vector<double> reference_first_last = {64512, 64513, 64514, 64515, 64768, 64769, 64770, 64771,
                                        65024, 65025, 65026, 65027, 65280, 65281, 65282, 65283};

  compare_tile<2>(tiled_matrix, tiling_info, tile_index, reference_first_last);

  // view on last tile
  tile_index[0] = (untiled_stride / tile_size) - 1;
  tile_index[1] = (untiled_stride / tile_size) - 1;

  std::vector<double> reference_last = {64764, 64765, 64766, 64767, 65020, 65021, 65022, 65023,
                                        65276, 65277, 65278, 65279, 65532, 65533, 65534, 65535};

  compare_tile<2>(tiled_matrix, tiling_info, tile_index, reference_last);

  // double a = 1.0;
  // double b = 3.0/3.0;
  // BOOST_CHECK_CLOSE(a, b, 1E-10);
}

BOOST_AUTO_TEST_SUITE_END()
