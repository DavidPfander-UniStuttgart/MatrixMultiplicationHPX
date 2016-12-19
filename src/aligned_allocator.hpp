#pragma once

#include <memory>

namespace alignment {

struct aligned_allocator {
  char *data;
  void *p;
  aligned_allocator() : data(nullptr), p(nullptr) {}

  ~aligned_allocator() {
    if (data) {
      delete[] data;
    }
  }

  template <typename T> T *alloc(size_t size, size_t alignment) {
    // can only manage a single aligned pointer per instance
    if (data) {
      throw;
    }
    size_t size_bytes = sizeof(T) * size + alignment; // sizeof(char) = 1 byte
    data = new char[size_bytes];
    p = data;
    p = std::align(alignment, sizeof(T), p, size_bytes);
    if (!p) {
      throw;
    }
    return reinterpret_cast<T *>(p);
  }
};
}
