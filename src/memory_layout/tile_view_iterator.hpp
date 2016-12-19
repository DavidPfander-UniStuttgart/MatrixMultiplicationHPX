#pragma once

#include "hpx/util/iterator_facade.hpp"

#include "tile_view.hpp"

namespace memory_layout {

template<size_t dim, typename T>
class tile_view_iterator: public hpx::util::iterator_facade<tile_view_iterator<dim, T>, T, std::forward_iterator_tag, const T&> {
private:

//    typedef hpx::util::iterator_facade<tile_view_iterator<T>, T, std::forward_iterator_tag, T&> base_type;

    friend class hpx::util::iterator_core_access;

    size_t inner_flat_index;
    std::reference_wrapper<tile_view<dim, T>> v;

    void increment() {
        inner_flat_index += 1;
    }

    bool equal(tile_view_iterator const& other) const {
        return inner_flat_index == other.inner_flat_index;
    }

    const T &dereference() const {
        return v.get()[inner_flat_index];
    }

public:

    tile_view_iterator(tile_view<dim, T> &v) :
            inner_flat_index(0), v(v) {
    }

    tile_view_iterator(tile_view<dim, T> &v, const size_t (&start_coord)[dim]) :
            v(v) {
        inner_flat_index = v.flat_index(start_coord);
    }

    tile_view_iterator &operator=(tile_view_iterator const &other) {
        if (this != &other) {
            inner_flat_index = other.inner_flat_index;
            v = other.v;
        }
        return *this;
    }
};

}
