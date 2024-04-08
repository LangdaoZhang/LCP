#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include "SZ3/utils/Timer.hpp"

#define BASE_BITS 8
#define BASE (1 << BASE_BITS)
#define MASK (BASE - 1)
#define DIGITS(v, shift) (((v) >> shift) & MASK)

template <typename T>
void radix_sort(T *start, T *end) {
    size_t numElements = end - start;
    T* buffer = new T[numElements];
    int total_digits = sizeof(size_t) * 8;

    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
        size_t bucket[BASE] = {0};
        size_t offset[BASE] = {0};

        for(size_t i = 0; i < numElements; i++){
            // mask out the current digit number
            bucket[DIGITS(start[i].id, shift)]++;
        }

        // update bucket to prefix-sum
        for (size_t i = 1; i < BASE; i++) {
            offset[i] = offset[i - 1] + bucket[i - 1];
        }

        for(size_t i = 0; i < numElements; i++) {
            // according to the current digits, get bin index in local_bucket
            size_t cur_num_digit = DIGITS(start[i].id, shift);
            // according to the value in the current thread's bin index, get the position that the number should be assigned to
            size_t pos = offset[cur_num_digit]++;
            // assgin the number to the new position
            buffer[pos] = start[i];
        }

        // move data
        T* tmp = start;
        start = buffer;
        buffer = tmp;
    }

    free(buffer);

//    SZ3::Timer timer(true);

    T *l = start, *r = l;
    while(l < end){
        r = l;
        while(r + 1 < end && l -> id == (r + 1) -> id){
            ++r;
        }
        if(l < r) std::sort(l, r + 1, [&](T u, T v){return u.reid < v.reid;});
        l = r + 1;
    }

//    double sort_time = timer.stop();
//    printf("second sort time = %fs\n", sort_time);
}
