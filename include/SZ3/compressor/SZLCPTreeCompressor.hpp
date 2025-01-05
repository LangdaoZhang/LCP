#ifndef SZ3_SZLCPTREECOMPRESSOR_HPP
#define SZ3_SZLCPTREECOMPRESSOR_HPP

#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/frontend/Frontend.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/def.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <cmath>
#include <utility>

namespace SZ3 {
    template<class T, class Encoder, class Lossless>
    class SZLCPTreeCompressor {
    public:

        explicit SZLCPTreeCompressor(Encoder encoder, Lossless lossless) :
                encoder(encoder), lossless(lossless) {
            static_assert(std::is_base_of<concepts::EncoderInterface<int64_t>, Encoder>::value,
                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");
        }

        class Point {
        public:
            size_t a[3];

            explicit Point(size_t x = 0, size_t y = 0, size_t z = 0) {
                a[0] = x, a[1] = y, a[2] = z;
            }

            size_t &operator[](size_t i) {
                return this->a[i];
            }
            const size_t &operator[](size_t i) const {
                return this->a[i];
            }
        };

        inline void getRange(T *a, size_t n, T &l, T &r) {
            l = r = a[0];
            for (size_t i = 1; i < n; i++) {
                l = std::min(l, a[i]);
                r = std::max(r, a[i]);
            }

        }

        inline void getRangeQuantizePoints(const Config &conf, T *datax, T *datay, T *dataz,
                                           std::array<T, 6> &range, std::array<size_t, 3> &qrange, Point *p) {
            const size_t &n = conf.num;

            T &lx = range[0], &rx = range[1];
            getRange(datax, n, lx, rx);
            for (size_t i = 0; i < n; i++) {
                p[i][0] = (datax[i] - lx) / (conf.absErrorBound * 2);
            }
            qrange[0] = (rx - lx) / (conf.absErrorBound * 2);

            T &ly = range[2], &ry = range[3];
            getRange(datay, n, ly, ry);
            for (size_t i = 0; i < n; i++) {
                p[i][1] = (datay[i] - ly) / (conf.absErrorBound * 2);
            }
            qrange[1] = (ry - ly) / (conf.absErrorBound * 2);

            T &lz = range[4], &rz = range[5];
            getRange(dataz, n, lz, rz);
            for (size_t i = 0; i < n; i++) {
                p[i][2] = (dataz[i] - lz) / (conf.absErrorBound * 2);
            }
            qrange[2] = (rz - lz) / (conf.absErrorBound * 2);
        }

        size_t ceildiv(size_t a, size_t b) {
            return (a + b - 1) / b;
        }

        uchar selectAxis(const Point *l, const Point *r, const std::array<size_t, 6> &range, uchar last_axis) {
            return (last_axis + 1) % 3;
            uchar best_axis = 0;
            int64_t best_num = 0;
            for (uchar axis = 0; axis < 3; axis++) {
                size_t pivot = range[axis * 2] + (range[axis * 2 + 1] - range[axis * 2]) / 2;
                int64_t num = 0;
                for (auto it = l; it < r; it++) {
                    if ((*it)[axis] < pivot) {
                        num++;
                    }
                    else{
                        num--;
                    }
                }
                num = abs(num);
                if (num > best_num) {
                    best_num = num;
                    best_axis = axis;
                }
            }
            return best_axis;
        }

        uchar selectAxis(uchar last_axis) {
            return (last_axis + 1) % 3;
        }

        void compressLCP(Point *l, Point *r, uint8_t depth, std::array<size_t, 6> range, const size_t &index_offset, size_t *ord = nullptr) {

            if(l == r) return;
            size_t stateNum = (range[1] - range[0]) * (range[3] - range[2]) * (range[5] - range[4]);
//            printf("%zu %zu %zu [%zu] %.2lf\n", range[1] - range[0], range[3] - range[2], range[5] - range[4],
//                   stateNum, log2(stateNum));
//            printf("%zu\n", r - l);

            class NodeWithOrder {
            public:
                explicit NodeWithOrder(size_t id, size_t reid, size_t ord) :
                id(id), reid(reid), ord(ord) {}
                size_t id, reid, ord;
            };

            std::array<size_t, 3> b = getBlockSize();
            size_t &bx = b[0], &by = b[1], &bz = b[2];

            size_t &lx = range[0], &rx = range[1];
            size_t &ly = range[2], &ry = range[3];
            size_t &lz = range[4], &rz = range[5];
            size_t nx = ceildiv(rx - lx, bx), ny = ceildiv(ry - ly, by), nz = ceildiv(rz - lz, bz);

            std::vector<NodeWithOrder> vec;
            vec.reserve(r - l);

            for (auto it = l; it < r; it++) {
                size_t x = (*it)[0] - lx;
                size_t cx = x / bx;
                size_t dx = x % bx;

                size_t y = (*it)[1] - ly;
                size_t cy = y / by;
                size_t dy = y % by;

                size_t z = (*it)[2] - lz;
                size_t cz = z / bz;
                size_t dz = z % bz;

                vec.push_back(NodeWithOrder(cx + cy * nx + cz * nx * ny, dx + dy * bx + dz * bx * by, it - l + index_offset));
            }

            std::sort(vec.begin(), vec.end(), [](const NodeWithOrder &u, const NodeWithOrder &v) {
                return u.id < v.id;
            });

            if(maximum_depth < depth) {
                ++maximum_depth;
                blkst.resize(depth + 1);
                blkcnt.resize(depth + 1);
                repos.resize(depth + 1);
            }

            auto &current_blkst = blkst[depth];
            auto &current_blkcnt = blkcnt[depth];
            auto &current_repos = repos[depth];

            size_t i = -1;
            size_t j = 0;
            size_t n = r - l;
            size_t pre = -1;

            for (; j < n; j++) {
                size_t &id = vec[j].id;
                size_t reid = vec[j].reid;

                if (vec[j].id != pre) {
                    current_blkst.push_back(id - pre);
                    current_blkcnt.push_back(0);
                    pre = id;
                }
                ++*current_blkcnt.rbegin();

                current_repos.push_back(reid);
            }
            if (ord != nullptr) {
                for (size_t i = 0; i < n; i++) {
                    ord[index_offset + i] = vec[i].ord;
                }
            }

        }

        /*
         * TODO: Find a method to determine the value of numblockPointLimit
         */
        size_t getNumBlockPointLimit() {
            return 1024;
        }

        /*
         * TODO: Find a method to determine the block size
         */
        std::array<size_t, 3> getBlockSize() {
            const size_t b = 1;
            return {b, b, b};
        }

        void compressTreeSplitting(Point *l, Point *r, const std::array<size_t, 3> &qrange, uchar *&tail, size_t *ord = nullptr) {

            size_t n = r - l;
            maximum_depth = 0;
            blkst.reserve(32);
            blkcnt.reserve(32);
            repos.reserve(32);

            class Status {
            public:

                explicit Status(Point *l, Point *r, uint8_t depth, uchar last_axis, std::array<size_t, 6> &&range, size_t index_offset, std::string &&cluster) :
                    l(l), r(r), depth(depth), last_axis(last_axis), range(range), index_offset(index_offset), cluster(std::move(cluster)) {
                }
                explicit Status(Point *l, Point *r, uint8_t depth, uchar last_axis, std::array<size_t, 6> &range, size_t index_offset, std::string &&cluster) :
                    l(l), r(r), depth(depth), last_axis(last_axis), range(range), index_offset(index_offset), cluster(std::move(cluster)) {
                }

                Point *l, *r;
                uint8_t depth;
                uchar last_axis;
                std::array<size_t, 6> range;
                size_t index_offset;
                std::string cluster;
                /*
                 * l, r are the start and end of the current point set
                 * range is the range of the current point set
                 * range[0] and range[1] are the range of x
                 * range[2] and range[3] are the range of y
                 * range[4] and range[5] are the range of z
                 */
            };

            class Splitter {
            public:
                Splitter(uchar axis, size_t value) : axis(axis), value(value) {}
                bool operator()(const Point &u) const {
                    return u[axis] < value;
                }

            private:
                const size_t value;
                const uchar axis;
            };

            numblockPointLimit = getNumBlockPointLimit();
            printf("numblockPointLimit = %zu\n", numblockPointLimit);

            std::stack<Status> stk;

            stk.push(Status(l, r, 0, 3 - 1, {0, qrange[0] + 1, 0, qrange[1] + 1, 0, qrange[2] + 1}, 0, "#"));

            while(!stk.empty()) {
                Status current_status = std::move(stk.top());
                stk.pop();

                l = current_status.l;
                r = current_status.r;
                uint8_t depth = current_status.depth;
                std::string &cluster = current_status.cluster;
//                writePointsCSV(l, r, cluster);
                size_t num_remaining_points = static_cast<size_t>(r - l);
                auto &current_range = current_status.range;
                size_t &current_index_offset = current_status.index_offset;

                if (num_remaining_points < numblockPointLimit) {
                    compressLCP(l, r, depth, current_range, current_index_offset, ord);
                    continue;
                }

                auto &current_axis = current_status.last_axis;
                auto next_axis = selectAxis(l, r, current_range, current_axis);

                if (current_range[next_axis * 2 + 1] - current_range[next_axis * 2] <= 4) {
                    compressLCP(l, r, depth, current_range, current_index_offset, ord);
                    continue;
                }

                size_t pivot = current_range[next_axis * 2] +
                        (current_range[next_axis * 2 + 1] - current_range[next_axis * 2]) / 2;
//                    largestPowerOf2LessThan(current_range[next_axis * 2 + 1] - current_range[next_axis * 2]);
                Point *mid = std::partition(l, r, Splitter(next_axis, pivot));
                int64_t weight_difference = static_cast<int64_t>(r - mid) - static_cast<int64_t>(mid - l);

                tree_nums.push_back(weight_difference);

                size_t current_range_next_axis_l = current_range[next_axis * 2];
                current_range[next_axis * 2] = pivot;
                stk.push(Status(mid, r, depth + 1, next_axis, current_range, current_index_offset + mid - l, cluster + "1"));
                current_range[next_axis * 2] = current_range_next_axis_l;
                current_range[next_axis * 2 + 1] = pivot;
                stk.push(Status(l, mid, depth + 1, next_axis, current_range, current_index_offset, cluster + "0"));

            }

            write(numblockPointLimit, tail);

            printf("tree_nums.size() = %zu\n", tree_nums.size());
            write(tree_nums.size(), tail);
            encoder.preprocess_encode(tree_nums, 0);
            encoder.save(tail);
            encoder.encode(tree_nums, tail);
            encoder.postprocess_encode();

            for (uint8_t depth = 0; depth <= maximum_depth; ++depth) {

                auto &current_blkst = blkst[depth];
                auto &current_blkcnt = blkcnt[depth];
                auto &current_repos = repos[depth];

//                printf("depth = %u\n", depth);
//                printf("current points number = %zu\n", current_repos.size());

                if(current_blkst.empty()) continue;

                write(current_blkst.size(), tail);
                encoder.preprocess_encode(current_blkst.data(), current_blkst.size(), 0, 0x01);
                encoder.save(tail);
                encoder.encode(current_blkst, tail);
                encoder.postprocess_encode();

                write(current_blkcnt.size(), tail);
                encoder.preprocess_encode(current_blkcnt, 0);
                encoder.save(tail);
                encoder.encode(current_blkcnt, tail);
                encoder.postprocess_encode();

                write(current_repos.size(), tail);
                encoder.preprocess_encode(current_repos.data(), current_repos.size(), 0, 0x01);
                encoder.save(tail);
                encoder.encode(current_repos, tail);
                encoder.postprocess_encode();
            }
        }

        uchar *compress(const Config &conf, T *datax, T *datay, T *dataz, size_t &compressed_size,
                        size_t *ord = nullptr) {
            const size_t &n = conf.num;
//            numblockPointLimit = pow(n, .75);
//            printf("numblockPointLimit = %zu\n", numblockPointLimit);

            uchar *head = new uchar[n * 16], *tail = head;
            Point *p = new Point[n];

            conf.save(tail);

            std::array<T, 6> range = {0};
            std::array<size_t, 3> qrange = {0};
            getRangeQuantizePoints(conf, datax, datay, dataz, range, qrange, p);
            write(range.begin(), 6 * sizeof(T), tail);

            compressTreeSplitting(p, p + n, qrange, tail, ord);
            delete[] p;

            uchar *lossless_data = lossless.compress(head, tail - head, compressed_size);

            printf("compressed_size = %zu\n", compressed_size);

            return lossless_data;
        }

//        void decompressLCP(T *datax, T *datay, T *dataz, const size_t n, std::array<size_t, 6> &qrange, const double eb,
//                            size_t &blkst_blkcnt_index, size_t &repos_index) {
//
//            std::array<size_t, 3> b = getBlockSize();
//            size_t &bx = b[0], &by = b[1], &bz = b[2];
//            size_t nx = ceildiv(qrange[1] - qrange[0], bx),
//                ny = ceildiv(qrange[3] - qrange[2], by),
//                nz = ceildiv(qrange[5] - qrange[4], bz);
//
//            size_t initial_repos_index = repos_index;
//
//            while(repos_index - initial_repos_index < n) {
//                auto blkst_it = blkst[blkst_blkcnt_index];
//                auto blknum_it = blkcnt[blkst_blkcnt_index];
//                ++ blkst_blkcnt_index;
//
//                size_t position_base_x = (blkst_it % nx) * bx;
//                size_t position_base_y = (blkst_it / nx % ny) * by;
//                size_t position_base_z = (blkst_it / nx / ny) * bz;
//
//                while(blknum_it--) {
//                    size_t quantized_position_x = position_base_x + repos[repos_index] % bx + qrange[0];
//                    size_t quantized_position_y = position_base_y + repos[repos_index] / bx % by + qrange[2];
//                    size_t quantized_position_z = position_base_z + repos[repos_index] / bx / by + qrange[4];
//                    datax[repos_index] = quantized_position_x * (2 * eb) + eb;
//                    datay[repos_index] = quantized_position_y * (2 * eb) + eb;
//                    dataz[repos_index] = quantized_position_z * (2 * eb) + eb;
//                    ++repos_index;
//                }
//            }
//        }
//
//        void decompressTreeSplitting(T *datax, T *datay, T *dataz, const size_t &n,
//                                     const std::array<size_t, 3> &qrange, const double eb, const uchar *cmpData) {
//            class Status {
//            public:
//
//                explicit Status(size_t l, size_t r, uchar last_axis, std::array<size_t, 6> &&range) :
//                        l(l), r(r), last_axis(last_axis), range(range) {
//                }
//                explicit Status(size_t l, size_t r, uchar last_axis, std::array<size_t, 6> &range) :
//                        l(l), r(r), last_axis(last_axis), range(range) {
//                }
//
//                size_t l, r;
//                uchar last_axis;
//                std::array<size_t, 6> range;
//                /*
//                 * l, r are the start and end of the current point set
//                 * range is the range of the current point set
//                 * range[0] and range[1] are the range of x
//                 * range[2] and range[3] are the range of y
//                 * range[4] and range[5] are the range of z
//                 */
//            };
//
//            std::stack<Status> stk;
//            stk.push(Status(0, n, 3 - 1, {0, qrange[0] + 1, 0, qrange[1] + 1, 0, qrange[2] + 1}));
//
//            size_t tree_nums_index = -1;
//            size_t blkst_blkcnt_index = 0;
//            size_t repos_index = 0;
//
//            while(!stk.empty()) {
//
//                Status current_status = std::move(stk.top());
//                stk.pop();
//
//                size_t &l = current_status.l;
//                size_t &r = current_status.r;
//                uchar &current_axis = current_status.last_axis;
//                std::array<size_t, 6> &current_range = current_status.range;
//
//                uchar next_axis = selectAxis(current_axis);
//
//                if (r - l < numblockPointLimit || current_range[next_axis * 2 + 1] - current_range[next_axis * 2] <= 4) {
//                    decompressLCP(datax, datay, dataz, r - l, current_range, eb, blkst_blkcnt_index, repos_index);
//                    continue;
//                }
//
//                size_t left_remaining_points = (static_cast<int64_t>(r - l) - tree_nums[++tree_nums_index]) >> 1;
//                assert((static_cast<int64_t>(r - l) - tree_nums[++tree_nums_index]) & 1 == 0);
////                size_t right_remaining_points = r - l - left_remaining_points;
////                assert(static_cast<int64_t>(r - l) - left_remaining_points >= 0);
//                size_t pivot = current_range[next_axis * 2] +
//                        (current_range[next_axis * 2 + 1] - current_range[next_axis * 2]) / 2;
//
//                size_t current_range_next_axis_l = current_range[next_axis * 2];
//                current_range[next_axis * 2] = pivot;
//                stk.push(Status(l + left_remaining_points, r, next_axis, current_range));
//                current_range[next_axis * 2] = current_range_next_axis_l;
//                current_range[next_axis * 2 + 1] = pivot;
//                stk.push(Status(l, l + left_remaining_points, next_axis, current_range));
//            }
//        }
//
//        void decompress(const uchar *lossless_data, T *&datax, T *&datay, T *&dataz, size_t &n, size_t cmpSize) {
//
//            uchar const *cmpData = lossless.decompress(lossless_data, cmpSize);
//
//            SZ3::Config conf;
//            conf.load(cmpData);
//            n = conf.num;
//
//            if (datax == nullptr && datay == nullptr && dataz == nullptr) {
//                datax = new T[n];
//                datay = new T[n];
//                dataz = new T[n];
//            }
//            if (datax == nullptr) datax = new T[n];
//            if (datay == nullptr) datay = new T[n];
//            if (dataz == nullptr) dataz = new T[n];
//
//            std::array<T, 6> range = {0};
//            std::array<size_t, 3> qrange = {0};
//
//            read(range.begin(), 6 * sizeof(T), cmpData);
//            qrange[0] = (range[1] - range[0]) / (conf.absErrorBound * 2);
//            qrange[1] = (range[3] - range[2]) / (conf.absErrorBound * 2);
//            qrange[2] = (range[5] - range[4]) / (conf.absErrorBound * 2);
//
//            size_t remaining_length = 0;
//
//            read(numblockPointLimit, cmpData);
//            readVectorFromEncodedData(tree_nums, cmpData);
//            readVectorFromEncodedData(blkst, cmpData);
//            for(size_t i = 1; i < blkst.size(); i++) blkst[i] += blkst[i - 1];
//            readVectorFromEncodedData(blkcnt, cmpData);
//            readVectorFromEncodedData(repos, cmpData);
//
//            decompressTreeSplitting(datax, datay, dataz, n, qrange, conf.absErrorBound, cmpData);
//            delete[] cmpData;
//        }

    private:
        size_t numblockPointLimit = 1024;

        std::vector<int64_t> tree_nums;
        uint8_t maximum_depth;
        std::vector<std::vector<int64_t>> blkst, blkcnt, repos;

        Encoder encoder;
        Lossless lossless;

        size_t largestPowerOf2LessThan(size_t x) {
            if (x <= 1) return 0;
            --x;
            size_t highestBit = 63 - __builtin_clzll(x);
            size_t res = static_cast<size_t>(1) << highestBit;
            return res;
        }

        template<typename Type>
        void readVectorFromEncodedData(std::vector<Type> &vec, uchar const *&cmpData) {
            size_t remaining_length = 0;
            size_t vec_length = 0;
            read(vec_length, cmpData);
            encoder.load(cmpData, remaining_length);
            vec = std::move(encoder.decode(cmpData, vec_length));
        }

        void writePointsCSV(Point *l, Point *r, std::string filename) {
            static const std::string path = "/Users/longtaozhang/tem/";
            filename = path + filename + ".csv";
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Failed to open file:" + filename << std::endl;
                exit(1);
            }
            for (auto it = l; it < r; it++) {
                file << (*it)[0] << "," << (*it)[1] << "," << (*it)[2] << std::endl;
            }
        }
    };

}

#endif