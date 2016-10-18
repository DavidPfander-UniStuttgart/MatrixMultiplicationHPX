//============================================================================
// Name        : MetaProgrammingCPP.cpp
// Author      : David Pfander
// Version     :
// Copyright   : 
// Description :
//============================================================================

#include <iostream>

template<size_t dim, typename T, typename F, typename ... Args>
typename std::enable_if<dim == 0, void>::type execute_looped(T min, T max, F f,
        Args ... args) {
    f(args...);
}

template<size_t dim, typename T, typename F, typename ... Args>
typename std::enable_if<dim != 0, void>::type execute_looped(T min, T max, F f,
        Args ... args) {
    for (T cur = min; cur < max; cur++) {
        execute_looped<dim - 1>(min, max, f, args..., cur);
    }
}

template<size_t dim, typename T>
class Looper {
private:
    T min, max;
public:
    Looper(T min, T max) :
            min(min), max(max) {
    }

    template<typename F>
    void iterate_simple(F f) {
        execute_looped<dim, T, F>(min, max, f);
//		for (T cur = min; cur < max; cur++) {
//			f(cur);
//		}
    }
};

int main() {

    Looper<2, size_t> loop(0, 10);

    loop.iterate_simple([](size_t cur0, size_t cur1) {
        if (cur0 > 0 || cur1 > 0) {
            std::cout << ", ";
        }
        std::cout << "(" << cur0 << ", " << cur1 << ")";
    });
    std::cout << std::endl;

    return 0;
}
