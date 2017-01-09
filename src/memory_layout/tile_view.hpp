#pragma once

#include <vector>

#include "tiling_info_dim.hpp"

namespace memory_layout {

struct tiling_info_dim;

template <size_t dim, typename T, typename U> class tile_view {
private:
  std::vector<T, U> &tiled;
  size_t base_offset;
	std::array<size_t, dim> tile_index;
	std::array<size_t, dim> tiles_dir;
  std::vector<tiling_info_dim> tiling_info;
  size_t tile_size;

public:
  tile_view(std::vector<T, U> &tiled, size_t (&tile_index)[dim],
            std::vector<tiling_info_dim> tiling_info)
      : tiled(tiled), tiling_info(tiling_info) {

    this->tile_index[0] = tile_index[0];
    this->tile_index[1] = tile_index[1];

    for (size_t d = 0; d < dim; d++) {
      this->tiles_dir[d] = tiling_info[d].stride / tiling_info[d].tile_size_dir;
    }

    tile_size = 1;
    for (size_t d = 0; d < dim; d++) {
      tile_size *= tiling_info[d].tile_size_dir;
    }

    size_t cur_stride = 1;
    base_offset = 0;
    for (size_t d = 0; d < dim; d++) {
      base_offset += tile_index[(dim - 1) - d] * cur_stride;
      cur_stride *= (tiling_info[(dim - 1) - d].stride /
                     tiling_info[(dim - 1) - d].tile_size_dir);
    }

    base_offset *= tile_size;
  }

  T &operator[](const size_t tile_offset) { return tiled[base_offset + tile_offset]; }

	T &operator[](const std::array<size_t, dim> &inner_index) {
		size_t tile_offset = flat_index(inner_index);
		return tiled[base_offset + tile_offset];
	}

  template <typename... index_types>
  typename std::enable_if<sizeof...(index_types) == dim, T &>::type
  operator()(index_types... indices) {
		std::array<size_t, dim> inner_index = {indices...};
    return tiled[base_offset + flat_index(inner_index)];
  }

  size_t size() const { return tile_size; }

	std::array<size_t, dim> &get_tile_index() { return this->tile_index; }

	std::array<size_t, dim> &get_tiles_dir() { return this->tiles_dir; }

  size_t flat_index(const std::array<size_t, dim> &coord) const {
    size_t inner_flat_index = 0;
    size_t cur_stride = 1;
    for (size_t d = 0; d < dim; d++) {
      inner_flat_index += coord[(dim - 1) - d] * cur_stride;
      cur_stride *= tiling_info[(dim - 1) - d].tile_size_dir;
    }
    return inner_flat_index;
  }
};
}
