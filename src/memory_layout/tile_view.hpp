#pragma once

#include <vector>

#include "tile_array.hpp"

namespace memory_layout {

template<size_t dim, typename T>
class tile_view {
private:
    std::vector<T> &tiled;
    size_t base_offset;
    size_t tile_index[dim];
    std::vector<tiling_info_dim> tiling_info;
    size_t tile_size;

public:
    tile_view(std::vector<T> &tiled, size_t (&tile_index)[dim], std::vector<tiling_info_dim> tiling_info) :
            tiled(tiled), tiling_info(tiling_info) {

        this->tile_index[0] = tile_index[0];
        this->tile_index[1] = tile_index[1];

        tile_size = 1;
        for (size_t d = 0; d < dim; d++) {
            tile_size *= tiling_info[d].tile_size_dir;
        }

        size_t cur_stride = 1;
        base_offset = 0;
        for (size_t d = 0; d < dim; d++) {
            base_offset += tile_index[(dim - 1) - d] * cur_stride;
            cur_stride *= (tiling_info[(dim - 1) - d].stride / tiling_info[(dim - 1) - d].tile_size_dir);
        }

        base_offset *= tile_size;
    }

    T& operator[](size_t tile_offset) {
        return tiled[base_offset + tile_offset];
    }

    template <typename... index_types>
    typename std::enable_if<sizeof...(index_types) == dim, T &>::type
    operator()(index_types... indices) {
      size_t inner_index[] = {indices...};
      return tiled[base_offset + flat_index(inner_index)];
    }

    size_t size() const {
        return tile_size;
    }

    size_t (&get_tile_index())[2] {
	return this->tile_index;
    }

    size_t flat_index(const size_t (&coord)[dim]) const {
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
