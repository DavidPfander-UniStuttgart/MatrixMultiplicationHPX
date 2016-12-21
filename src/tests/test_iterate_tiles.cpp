#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "memory_layout/tile_array.hpp"
#include "memory_layout/tile_view.hpp"
#include "memory_layout/tile_iterator.hpp"

BOOST_AUTO_TEST_SUITE(test_iterate_tiles)

BOOST_AUTO_TEST_CASE(iterate) {
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

  size_t i = 0;
  size_t j = 0;

  bool cases[4];
  cases[0] = false;
  cases[1] = false;
  cases[2] = false;
  cases[3] = false;

  memory_layout::iterate_tiles<2>(tiled_matrix, tiling_info, [&i, &j, &tiling_info, &cases](auto view) {
    auto &tile_index = view.get_tile_index();
    BOOST_CHECK_EQUAL(tile_index[0], i);
    BOOST_CHECK_EQUAL(tile_index[1], j);

    if (i == 0 && j == 0) {
      cases[0] = true;
      std::vector<double> reference_first = {0,   1,   2,   3,   256, 257, 258, 259,
                                             512, 513, 514, 515, 768, 769, 770, 771};
      for (size_t x = 0; x < tiling_info[0].tile_size_dir; x++) {
	  for (size_t y = 0; y < tiling_info[1].tile_size_dir; y++) {
	      BOOST_CHECK_EQUAL(view(x, y), reference_first[x * 4 + y]);
        }
      }
    }
    else if (i == 0 && j == 63) {
      cases[1] = true;
        std::vector<double> reference_first_end = {252, 253, 254, 255, 508,  509,  510,  511,
                                             764, 765, 766, 767, 1020, 1021, 1022, 1023};
      for (size_t x = 0; x < tiling_info[0].tile_size_dir; x++) {
        for (size_t y = 0; y < tiling_info[1].tile_size_dir; y++) {
          BOOST_CHECK_EQUAL(view(x, y), reference_first_end[x * 4 + y]);
        }
      }
    }
    else if (i == 63 && j == 0) {
      cases[2] = true;
      std::vector<double> reference_first_last = {64512, 64513, 64514, 64515, 64768, 64769,
                                                  64770, 64771, 65024, 65025, 65026, 65027,
                                                  65280, 65281, 65282, 65283};
      for (size_t x = 0; x < tiling_info[0].tile_size_dir; x++) {
        for (size_t y = 0; y < tiling_info[1].tile_size_dir; y++) {
          BOOST_CHECK_EQUAL(view(x, y), reference_first_last[x * 4 + y]);
        }
      }
    }
    else if (i == 63 && j == 63) {
      cases[3] = true;
      std::vector<double> reference_last = {64764, 64765, 64766, 64767, 65020, 65021, 65022, 65023,
					      65276, 65277, 65278, 65279, 65532, 65533, 65534, 65535};
      for (size_t x = 0; x < tiling_info[0].tile_size_dir; x++) {
        for (size_t y = 0; y < tiling_info[1].tile_size_dir; y++) {
          BOOST_CHECK_EQUAL(view(x, y), reference_last[x * 4 + y]);
        }
      }
    }

    //TODO: add last case

    j += 1;
    if (j == 64) {
      i += 1;
      j = 0;
    }
  });
  BOOST_CHECK_EQUAL(cases[0], true);
  BOOST_CHECK_EQUAL(cases[1], true);
  BOOST_CHECK_EQUAL(cases[2], true);
  BOOST_CHECK_EQUAL(cases[3], true);
}

BOOST_AUTO_TEST_SUITE_END()
