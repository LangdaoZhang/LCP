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

        static const size_t numblockPointLimit = 64;

        uchar selectAxis(Point *l, Point *r, uchar last_axis) {
            return (last_axis + 1) % 3;
        }

        std::pair<uchar *, size_t> compressLCP(Point *l, Point *r) {
            return {nullptr, 0};
        }

        void compressTreeSplitting(Point *l, Point *r, const std::array<size_t, 3> &qrange) {

            class Status {
            public:

                explicit Status(Point *l, Point *r, uchar last_axis, std::array<size_t, 6> &&range) :
                    l(l), r(r), last_axis(last_axis), range(range) {
                }
                explicit Status(Point *l, Point *r, uchar last_axis, std::array<size_t, 6> &range) :
                        l(l), r(r), last_axis(last_axis), range(range) {
                }

                Point *l, *r;
                uchar last_axis;
                std::array<size_t, 6> range;
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

            std::stack<Status> stk;

            stk.push(Status(l, r, 3 - 1, {0, qrange[0] + 1, 0, qrange[1] + 1, 0, qrange[2] + 1}));

            while(!stk.empty()) {
                Status current_status = stk.top();
                stk.pop();

                l = current_status.l;
                r = current_status.r;
                size_t num_remaining_points = static_cast<size_t>(r - l);

                if (num_remaining_points < numblockPointLimit) {
                    clusters.push_back(compressLCP(l, r));
                    continue;
                }

                auto &current_axis = current_status.last_axis;
                auto &current_range = current_status.range;
                auto next_axis = selectAxis(l, r, current_axis);

                if (current_range[next_axis * 2 + 1] - current_range[next_axis * 2] <= 4) {
                    clusters.push_back(compressLCP(l, r));
                    continue;
                }

                size_t pivot = current_range[next_axis * 2] + (current_range[next_axis * 2 + 1] - current_range[next_axis * 2]) / 2;
                Point *mid = std::partition(l, r, Splitter(next_axis, pivot));
                int64_t weight_difference = static_cast<int64_t>(r - mid) - static_cast<int64_t>(mid - l);

                tree_nums.push_back(weight_difference);

                size_t current_range_next_axis_l = current_range[next_axis * 2];
                current_range[next_axis * 2] = pivot;
                stk.push(Status(mid, r, next_axis, current_range));
                current_range[next_axis * 2] = current_range_next_axis_l;
                current_range[next_axis * 2 + 1] = pivot;
                stk.push(Status(l, mid, next_axis, current_range));

            }
        }

        uchar *compress(const Config &conf, T *datax, T *datay, T *dataz, size_t &compressed_size,
                        size_t *ord = nullptr) {
            const size_t &n = conf.num;

            uchar *head = new uchar[n * 16], *tail = head;
            Point *p = new Point[n];

            std::array<T, 6> range = {0};
            std::array<size_t, 3> qrange = {0};
            getRangeQuantizePoints(conf, datax, datay, dataz, range, qrange, p);
            write(range.begin(), 6 * sizeof(T), tail);

            compressTreeSplitting(p, p + n, qrange);

            encoder.preprocess_encode(tree_nums, 0);
            encoder.encode(tree_nums, tail);
            encoder.postprocess_encode();
//            for (auto &num : tree_nums) {
//                printf("%lld\n", num);
//            }

            for(auto &cluster : clusters) {
                write(cluster.first, cluster.second, tail);
            }

            compressed_size = static_cast<size_t>(tail - head);

            printf("compressed_size = %zu\n", compressed_size);

            return head;
        }

    private:
        std::vector<int64_t> tree_nums;
//        std::vector<uchar> tree_axis;
        std::vector<std::pair<uchar *, size_t>> clusters;

        Encoder encoder;
        Lossless lossless;
    };

}

#endif