#pragma once

namespace memory_layout {
template <size_t dim>
size_t flat_index(const size_t (&index)[dim], const size_t (&strides)[dim]) {
  size_t flat_index = 0;
  size_t cur_stride = 1;
  for (size_t d = 0; d < dim; d++) {
    flat_index += index[(dim - 1) - d] * cur_stride;
    cur_stride *= strides[d];
  }
  return flat_index;
}

template <size_t dim>
size_t flat_index(const std::array<size_t, dim> &index,
                  const std::array<size_t, dim> &strides) {
  size_t flat_index = 0;
  size_t cur_stride = 1;
  for (size_t d = 0; d < dim; d++) {
    flat_index += index[(dim - 1) - d] * cur_stride;
    cur_stride *= strides[d];
  }
  return flat_index;
}
}
