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

//    SZ3::Timer timer(true);

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

//    double sort_time = timer.stop();
//    printf("first sort time = %fs\n", sort_time);


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

//template <typename T>
//void radix_sort(T* start, T* end) {
//
//    SZ3::Timer timer(true);
//
//    size_t numElements = end - start;
//    T* buffer = new T[numElements];
//    int total_digits = sizeof(size_t) * 8;
//
//    // Each thread use local_bucket to move data
//    for(int shift = 0; shift < total_digits; shift+=BASE_BITS) {
//        size_t bucket[BASE] = {0};
//
//        // size needed in each bucket/thread
//        size_t local_bucket[BASE] = {0};
//        //1st pass, scan whole and check the count
//#pragma omp parallel firstprivate(local_bucket) //num_threads(8)
//        {
//            // calculate occurance of each number (current bits)
//#pragma omp for schedule(static) nowait
//            for(size_t i = 0; i < numElements; i++){
//                // mask out the current digit number
//                local_bucket[DIGITS(start[i].id, shift)]++;
//            }
//
//            // counts in each local bucket will sum up to bucket
//#pragma omp critical
//            for(size_t i = 0; i < BASE; i++) {
//                bucket[i] += local_bucket[i];
//            }
//#pragma omp barrier
//
//            // update bucket to prefix-sum for example, [1,2,2,2] to [1,3,5,7] to get each bin's end index
//            // only one thread will run, others will wait here
//#pragma omp single
//            for (size_t i = 1; i < BASE; i++) {
//                bucket[i] += bucket[i - 1];
//            }
//            int nthreads = omp_get_num_threads();
//            int tid = omp_get_thread_num();
//            for(int cur_t = nthreads - 1; cur_t >= 0; cur_t--) {
//                if(cur_t == tid) {
//                    for(size_t i = 0; i < BASE; i++) {
//                        bucket[i] -= local_bucket[i];
//                        local_bucket[i] = bucket[i];
//                    }
//                }
//#pragma omp barrier
//            }
//
//            // move value from start to bins in buffer
//#pragma omp for schedule(static)
//            for(size_t i = 0; i < numElements; i++) {
//                // according to the current digits, get bin index in local_bucket
//                size_t cur_num_digit = DIGITS(start[i].id, shift);
//                // according to the value in the current thread's bin index, get the position that the number should be assigned to
//                size_t pos = local_bucket[cur_num_digit]++;
//                // assgin the number to the new position
//                buffer[pos] = start[i];
//            }
//        }
//        // move data
//        T* tmp = start;
//        start = buffer;
//        buffer = tmp;
//    }
//
//    double sort_time = timer.stop();
//    printf("first sort time = %fs\n", sort_time);
//
//    T *l = start, *r = l;
//    while(l < end){
//        r = l;
//        while(r + 1 < end && l -> id == (r + 1) -> id){
//            ++r;
//        }
//        if(l < r) std::sort(l, r + 1, [&](T u, T v){return u.reid < v.reid;});
//        l = r + 1;
//    }
//
//    free(buffer);
//}
