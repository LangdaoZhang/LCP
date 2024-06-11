//
// Created by Kai Zhao on 4/21/20.
//

#ifndef SZ_MEMORYOPS_HPP
#define SZ_MEMORYOPS_HPP
#include <cassert>
#include <cstring>
#include <vector>
#include "SZ3/def.hpp"
namespace SZ3 {
    // read array
    template<class T1>
    void read(T1 *array, size_t num_elements, uchar const *&compressed_data_pos, size_t &remaining_length) {
        assert(num_elements * sizeof(T1) <= remaining_length);
        memcpy(array, compressed_data_pos, num_elements * sizeof(T1));
        remaining_length -= num_elements * sizeof(T1);
        compressed_data_pos += num_elements * sizeof(T1);
    }

    // read array
    template<class T1>
    void read(T1 *array, size_t num_elements, uchar const *&compressed_data_pos) {
        memcpy(array, compressed_data_pos, num_elements * sizeof(T1));
        compressed_data_pos += num_elements * sizeof(T1);
    }

    // read variable
    template<class T1>
    void read(T1 &var, uchar const *&compressed_data_pos) {
        memcpy(&var, compressed_data_pos, sizeof(T1));
        compressed_data_pos += sizeof(T1);
    }

    // read variable
    template<class T1>
    void read(T1 &var, uchar const *&compressed_data_pos, size_t &remaining_length) {
        assert(sizeof(T1) <= remaining_length);
        memcpy(&var, compressed_data_pos, sizeof(T1));
        remaining_length -= sizeof(T1);
        compressed_data_pos += sizeof(T1);
    }

    // read vector
    template<class T1>
    void read(std::vector<T1> &vec, uchar const *&compressed_data_pos) {
        size_t num_elements = 0;
        read(num_elements, compressed_data_pos);
        vec.resize(num_elements);
        read(vec.data(), num_elements, compressed_data_pos);
    }

    // write array
    template<class T1>
    void write(T1 const *array, size_t num_elements, uchar *&compressed_data_pos) {
        memcpy(compressed_data_pos, array, num_elements * sizeof(T1));
        compressed_data_pos += num_elements * sizeof(T1);
    }

    // write variable
    template<class T1>
    void write(T1 const var, uchar *&compressed_data_pos) {
        memcpy(compressed_data_pos, &var, sizeof(T1));
        compressed_data_pos += sizeof(T1);
    }

    // write vector
    template<class T1>
    void write(std::vector<T1> vec, uchar *&compressed_data_pos) {
        write(vec.size(), compressed_data_pos);
        write(vec.data(), vec.size(), compressed_data_pos);
    }

}
#endif //SZ_MEMORYOPS_HPP
