/*
 * matrix_multiplication_component.hpp
 *
 *  Created on: Sep 5, 2016
 *      Author: pfandedd
 */

#include <cinttypes>
#include <hpx/include/components.hpp>

extern uint64_t small_block_size;
extern uint64_t verbose;

struct matrixMultiply_server: hpx::components::component_base<
        matrixMultiply_server> {

    size_t N;
    std::vector<double> A;
    std::vector<double> B;

    // TODO: why does this get called?
    matrixMultiply_server();

    matrixMultiply_server(size_t N, std::vector<double> &A,
            std::vector<double> &B) :
            N(N), A(A), B(B) {
    }

    template<typename T>
    void matrixMultiply(std::uint64_t x, std::uint64_t y, size_t blockSize);

//    // This will define the action type 'some_member_action' which
//    // represents the member function 'some_member_function' of the
//    // object type 'some_component'.
//    HPX_DEFINE_COMPONENT_ACTION(matrixMultiply_server, matrixMultiply<double>,
//            matrixMultiply_action);

    template<typename T>
    std::vector<double> matrixMultiply(std::uint64_t x, std::uint64_t y,
            size_t blockSize) {
        std::vector<T> C(blockSize * blockSize);
        if (blockSize <= small_block_size) {
            return matrix_multiply_kernel(A, B, C, N, x, y, blockSize);
        } else {
            if (verbose >= 1) {
                std::cout
                        << "handling large matrix, more work... (blocksize == "
                        << blockSize << ")" << std::endl;
            }
//            // We restrict ourselves to execute the matrixMultiply function locally.
//            hpx::naming::id_type const locality_id = hpx::find_here();
//
//            // split the current submatrix into four further submatrices
//            matrixMultiply_action mat;
//            uint64_t n_new = blockSize / 2;
//            std::vector<T> f1 = hpx::async(mat, locality_id, N, x, y, n_new);
//            std::vector<T> f2 = hpx::async(mat, locality_id, N, x + n_new, y,
//                    n_new);
//            std::vector<T> f3 = hpx::async(mat, locality_id, N, x, y + n_new,
//                    n_new);
//            std::vector<T> f4 = hpx::async(mat, locality_id, N, x + n_new,
//                    y + n_new, n_new);
//
//            std::vector<T> C_small;
//            C_small = f1.get();
//            extract_submatrix(C, C_small, x, y, n_new);
//            C_small = f2.get();
//            extract_submatrix(C, C_small, x + n_new, y, n_new);
//            C_small = f3.get();
//            extract_submatrix(C, C_small, x, y + n_new, n_new);
//            C_small = f4.get();
//            extract_submatrix(C, C_small, x + n_new, y + n_new, n_new);
        }
    }
};

//HPX_REGISTER_COMPONENT(hpx::components::component<matrixMultiply_server>,
//        matrixMultiply_server_component);

//HPX_REGISTER_ACTION_DECLARATION(matrixMultiply_server::matrixMultiply_action);

//
//HPX_REGISTER_ACTION(matrixMultiply_server::matrixMultiply_action);
//
//// Define a client side representation type for the component type
//// 'some_component' defined in the previous section.
////
//struct matrixMultiplyComponent_client: hpx::components::client_base<
//        matrixMultiplyComponent_client, matrixMultiply_server> {
//
//    using base_type = hpx::components::client_base<
//    matrixMultiplyComponent_client, matrixMultiply_server>;
//
//    matrixMultiplyComponent_client(hpx::future<hpx::id_type> && id) :
//            base_type(std::move(id)) {
//    }
//
//    hpx::future<void> matrixMultiplyClient(std::uint64_t x,
//            std::uint64_t y, size_t blockSize) {
//        return hpx::async<matrixMultiply_server::matrixMultiply_action>(
//                get_id(), x, y, blockSize);
//    }
//};

