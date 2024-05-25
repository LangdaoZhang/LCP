#ifndef SZ3_SZDISCRETECOMPRESSOR_HPP
#define SZ3_SZDISCRETECOMPRESSOR_HPP

#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/frontend/Frontend.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Timer.hpp"
#include "SZ3/def.hpp"
#include "SZ3/utils/KDtree.hpp"
#include "SZ3/quantizer/IntegerQuantizer.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/utils/RadixSort.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include <cstring>
#include <algorithm>
#include <math.h>
#include <iostream>

#define __OUTPUT_INFO 0
#define __soft_eb 0
#define __batch_info 0

namespace SZ3 {
    template<class T, class Encoder, class Lossless>
    class SZDiscreteCompressor {
    public:

        SZDiscreteCompressor(Encoder encoder, Lossless lossless) :
                encoder(encoder), lossless(lossless) {
            static_assert(std::is_base_of<concepts::EncoderInterface<size_t>, Encoder>::value,
                          "must implement the encoder interface");
            static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                          "must implement the lossless interface");
        }

        class Node {
        public:
            size_t id;
            size_t reid;

            Node(size_t id = 0, size_t reid = 0) :
                    id(id), reid(reid) {
            }
        };

        class NodeWithOrder {
        public:
            size_t id;
            size_t reid;
            size_t ord;

            NodeWithOrder(size_t id = 0, size_t reid = 0, size_t ord = 0) :
                    id(id), reid(reid), ord(ord) {
            }
        };

//        T sq(T x){return x*x;}
        T sq(T x) { return fabs(x); }

        template<class Type>
        Type max(Type x, Type y, Type z) {
            return std::max(std::max(x, y), z);
        }

        class BlockSizeCache {
        public:

            BlockSizeCache() {
                init();
            }

            void init() {
                flag = 0x00;
                c = 0;
            }

            uchar isHit() {
                if (flag == 0x01) {
                    if (c > 0) {
                        --c;
                        return 0x01;
                    } else {
                        return 0x00;
                    }
                }
                return 0x00;
            }

            void write(std::vector<size_t> vec) {
                blockSize = vec;
                flag = 0x01;
                c = 64;
            }

            void write(size_t bx, size_t by, size_t bz) {
                write({bx, by, bz});
            }

            void write(size_t bx) {
                write({bx, bx, bx});
            }

            std::vector<size_t> get() {
                return blockSize;
            }

        private:

            uchar flag = 0x00;
            size_t c = 0;
            std::vector<size_t> blockSize = {0, 0, 0};
        };

        BlockSizeCache blockSizeCache;

        void
        getSample(T *datax, T *datay, T *dataz, T *samplex, T *sampley, T *samplez, T px, T py, T pz, T rx, T ry, T rz,
                  size_t n, size_t &_, size_t b = 3) {

            _ = 0;

            if (n <= (1llu << 12)) {
                memcpy(samplex, datax, n * sizeof(T));
                memcpy(sampley, datay, n * sizeof(T));
                memcpy(samplez, dataz, n * sizeof(T));
                _ = n;
                return;
            }

            size_t b2 = b * b, b3 = b2 * b;
            size_t cnt[b3];
            memset(cnt, 0x00, sizeof(cnt));
            size_t qx, qy, qz;

            for (size_t i = 0; i < 100; i++) {
                qx = (datax[i] - px) / rx * b;
                qy = (datay[i] - py) / ry * b;
                qz = (dataz[i] - pz) / rz * b;
                ++cnt[qx + qy * b + qz * b2];
            }
            size_t s = (b / 2) * (1 + b + b2);
            for (size_t i = 0; i < b3; i++) {
                if (cnt[i] > cnt[s] && cnt[i] <= n / std::min(b3, (size_t) 100)) {
                    s = i;
                }
            }
            for (size_t i = 0; i < n; i++) {
                qx = (datax[i] - px) / rx * b;
                qy = (datay[i] - py) / ry * b;
                qz = (dataz[i] - pz) / rz * b;
                if (qx + qy * b + qz * b2 == s) {
                    samplex[_] = datax[i];
                    sampley[_] = datay[i];
                    samplez[_] = dataz[i];
                    ++_;
                }
            }
//            printf("_ = %zu\n", _);
        }

        std::vector<size_t>
        getBlockSize(T *datax, T *datay, T *dataz, T px, T py, T pz, T rx, T ry, T rz, const Config &conf,
                     uchar blkflag) {

            if (blockSizeCache.isHit()) {
                return blockSizeCache.get();
            }

//            return {16, 16, 16};

            if (rx > 1e6 || ry > 1e6 || rz > 1e6) return {0, 0, 0};
            if (conf.absErrorBound < 1e-6) return {0, 0, 0};
            if (rx / conf.absErrorBound >= (1llu << 32) || ry / conf.absErrorBound >= (1llu << 32) ||
                rz / conf.absErrorBound >= (1llu << 32))
                return {0, 0, 0};
            if (rx / conf.absErrorBound <= (1 << 04) && ry / conf.absErrorBound <= (1 << 04) &&
                rz / conf.absErrorBound <= (1 << 04))
                return {1, 1, 1};

            /*
             * block size estimation
             * 0x01 : use the curve for HACC
             * 0x02 : use an online estimation method based on formula
             * 0x03 : use an online estimation method based on sampling and testing
             * 0x04 : use an offline estimation method which tries every possible blk sz
             */

            size_t la = 0, ra = ceil(log2(rx / conf.absErrorBound));
            while (ceil(rx / (2. * (1llu << la) * conf.absErrorBound)) > 1e6) ++la;
            size_t lb = 0, rb = ceil(log2(ry / conf.absErrorBound));
            while (ceil(ry / (2. * (1llu << lb) * conf.absErrorBound)) > 1e6) ++lb;
            size_t lc = 0, rc = ceil(log2(rz / conf.absErrorBound));
            while (ceil(rz / (2. * (1llu << lc) * conf.absErrorBound)) > 1e6) ++lc;


//            printf("l = %zu, r = %zu\n", 1llu << la, 1llu << ra);
//            printf("l = %zu, r = %zu\n", 1llu << lb, 1llu << rb);
//            printf("l = %zu, r = %zu\n", 1llu << lc, 1llu << rc);

            switch (blkflag) {

                case 0x00: {

                }
                case 0x01: {

                    double nlg = -log10(conf.absErrorBound);
                    static const double a[5] = {1,
                                                -14.75,
                                                27.9583333333333333333333,
                                                -16.25,
                                                3.04166666666666666666667};
                    double f = a[0];
                    for (int i = 1; i <= 4; i++) {
                        f += a[i] * pow(nlg, i);
                    }
                    f += 0.1;
                    size_t s = (size_t) f;
                    s = std::max(s, (size_t) (1llu << la));
                    s = std::max(s, (size_t) (1llu << lb));
                    s = std::max(s, (size_t) (1llu << lc));
                    s = std::min(s, (size_t) (1llu << ra));
                    s = std::min(s, (size_t) (1llu << rb));
                    s = std::min(s, (size_t) (1llu << rc));

                    blockSizeCache.write(s);
                    return {s, s, s};
                }
                case 0x02: {

                    size_t bx = -1, by = -1, bz = -1;
                    double est = 1. / 0.;

                    T *sample = new T[conf.num * 3];
                    T *samplex = sample, *sampley = samplex + conf.num, *samplez = sampley + conf.num;
                    size_t *blkid = new size_t[conf.num];
                    uchar *bytes = new uchar[conf.num * 12], *tail_bytes = bytes;
                    size_t n = 0;

                    getSample(datax, datay, dataz, samplex, sampley, samplez, px, py, pz, rx, ry, rz, conf.num, n);

//                for(size_t a = la; a <= ra; a++){
//                    for(size_t b = lb; b <= rb; b++){
//                        for(size_t c = lc; c <= rc; c++){
                    for (size_t a = std::max(la, std::max(lb, lc)); a <= std::min(ra, std::min(rb, rc)); a++) {
                        size_t b = a, c = a;
                        {
                            {
                                if (a + b + c > 48) {
                                    continue;
                                }
                                size_t nx = ceil(rx / (2. * (1llu << a) * conf.absErrorBound));
                                size_t ny = ceil(ry / (2. * (1llu << b) * conf.absErrorBound));
                                size_t nz = ceil(rz / (2. * (1llu << c) * conf.absErrorBound));
                                if (1. * nx * ny * nz > 1e18) {
                                    continue;
                                }

                                size_t snx = nx / 7 + 1;
                                size_t sny = ny / 7 + 1;
                                size_t snz = nz / 7 + 1;

                                for (size_t i = 0; i < n; i++) {
                                    blkid[i] = (samplex[i] / (2. * conf.absErrorBound * (1llu << a))) +
                                               ((sampley[i] / (2. * conf.absErrorBound * (1llu << b))) +
                                                (samplez[i] / (2. * conf.absErrorBound * (1llu << c))) * sny) * snx;
                                }
                                std::sort(blkid, blkid + n);
                                for (size_t i = n - 1; i > 0; i--) {
                                    blkid[i] -= blkid[i - 1];
                                }

                                tail_bytes = bytes;
                                encoder.preprocess_encode(blkid, n, 0, 0x00);
                                encoder.save(tail_bytes);
                                encoder.encode(blkid, n, tail_bytes);

//                                size_t blkid_compressed_size;
//                                delete[] lossless.compress(bytes, tail_bytes - bytes, blkid_compressed_size);
                                double f = 1. * (tail_bytes - bytes) / n * 8;

//                                printf("%zu %.12lf\n", a, f);

                                double len = f + a + b + c;
//                        printf("%zu %zu %zu %lf\n", 1llu << a, 1llu << b, 1llu << c, len);
                                if (len <= est) {
                                    bx = a;
                                    by = b;
                                    bz = c;
                                    est = len;
                                }
                            }
                        }
                    }

                    delete[] blkid;

                    bx = 1llu << bx;
                    by = 1llu << by;
                    bz = 1llu << bz;

//                printf("bx = %zu, by = %zu, bz = %zu\n", bx, by, bz);

//            size_t nx = ceil(rx / (2. * bx * conf.absErrorBound));
//            size_t ny = ceil(ry / (2. * by * conf.absErrorBound));
//            size_t nz = ceil(rz / (2. * bz * conf.absErrorBound));
//
//            printf("nx = %zu, ny = %zu, nz = %zu\n", nx, ny, nz);

                    blockSizeCache.write(bx, by, bz);
                    return {bx, by, bz};
                }
                case 0x03: {

                    T *sample = new T[conf.num * 3];
                    T *samplex = sample, *sampley = samplex + conf.num, *samplez = sampley + conf.num;
                    size_t n = 0;

//                    Timer timer(true);
                    getSample(datax, datay, dataz, samplex, sampley, samplez, px, py, pz, rx, ry, rz, conf.num, n);
//                    double t = timer.stop();
//                    printf("sample time = %lf\n", t);

                    Config sampleConfig = Config(n);
                    sampleConfig.absErrorBound = conf.absErrorBound;

                    size_t L = std::max(la, std::max(lb, lc));
                    size_t R = std::min(ra, std::min(rb, rc));
                    R = std::min(R, (size_t) 16);

                    size_t s = 0, sz = -1;

//                printf("sample n = %zu, l = %zu, r = %zu\n", n, 1llu<<l, 1llu<<r);

                    size_t l = L, r = R, midl = 0, midr = 0;

                    for (size_t i = L; i <= R; i++) {
//                for(size_t i=r;i>=l&&i<=r;i--){
                        size_t b = 1llu << i;

//                    memcpy(sample_tem, sample, conf.num * 3 * sizeof(T));

                        size_t compressed_size = 0;

                        const uchar *bytes = compressSimpleBlocking(sampleConfig, samplex, sampley, samplez,
                                                                    compressed_size, nullptr, 0x00, b, b, b);
                        delete[] bytes;
                        if (compressed_size < sz) {
                            s = b;
                            sz = compressed_size;
                        }
//                    printf("b = %5zu, compressed_size = %12zu, CR = %.6lf\n", b, compressed_size, 12. * n / compressed_size);
                    }

//                printf("s = %zu\n", s);

                    delete[] sample;

                    blockSizeCache.write(s);
                    return {s, s, s};

                }
                case 0x04: {

                    size_t bx = -1, by = -1, bz = -1;
                    size_t est = -1;
                    for (size_t a = la; a <= ra; a++) {
                        for (size_t b = lb; b <= rb; b++) {
                            for (size_t c = lc; c <= rc; c++) {
//                for(size_t a = std::max(la, std::max(lb, lc)); a <= std::min(ra, std::min(rb, rc)); a++){
//                    size_t b = a, c = a; {{
                                if (a + b + c >= 64) {
                                    continue;
                                }
                                size_t compressed_size = 0;
                                const uchar *bytes = compressSimpleBlocking(conf, datax, datay, dataz, compressed_size,
                                                                            nullptr, 0x00, 1llu << a, 1llu << b,
                                                                            1llu << c);
                                delete[] bytes;
//                            printf("%zu %zu %zu %zu %f\n", 1llu << a, 1llu << b, 1llu << c, compressed_size, 12. * conf.num / compressed_size);
                                if (compressed_size < est) {
                                    bx = a;
                                    by = b;
                                    bz = c;
                                    est = compressed_size;
                                }
                            }
                        }
                    }

                    bx = 1llu << bx;
                    by = 1llu << by;
                    bz = 1llu << bz;

//                printf("bx = %zu, by = %zu, bz = %zu\n", bx, by, bz);

                    blockSizeCache.write(bx, by, bz);
                    return {bx, by, bz};
                }
            }

            blockSizeCache.write(0);
            return {0, 0, 0};

        }

        /*
         * To compress the data from datax, datay and dataz using configure conf
         * Store the results in the return pointer, and store the size compressed data in compressed_data
         * Maybe store the order in ord if ord != nullptr
         *
         * after compression, datax, datay, dataz will not increase
         */

        uchar *compressSimpleBlocking(const Config &conf, T *datax, T *datay, T *dataz, size_t &compressed_size,
                                      size_t *ord = nullptr, uchar blkflag = 0x03, size_t bx_ = 0, size_t by_ = 0,
                                      size_t bz_ = 0) {

            T rx = datax[0], ry = datay[0], rz = dataz[0], px = datax[0], py = datay[0], pz = dataz[0];
//            T rx=0, ry=0, rz=0, px=0, py=0, pz=0;

            for (size_t i = 0; i < conf.num; i++) {
                rx = std::max(rx, datax[i]);
                px = std::min(px, datax[i]);
            }
            rx -= px;
//            if(px!=0) for(size_t i=0;i<conf.num;i++){
//                datax[i] -= px;
//            }

            for (size_t i = 0; i < conf.num; i++) {
                ry = std::max(ry, datay[i]);
                py = std::min(py, datay[i]);
            }
            ry -= py;
//            if(py!=0) for(size_t i=0;i<conf.num;i++){
//                datay[i] -= py;
//            }

            for (size_t i = 0; i < conf.num; i++) {
                rz = std::max(rz, dataz[i]);
                pz = std::min(pz, dataz[i]);
            }
            rz -= pz;
//            if(pz!=0) for(size_t i=0;i<conf.num;i++){
//                dataz[i] -= pz;
//            }

            std::vector<size_t> b = {0, 0, 0};

            if (blkflag != 0x00) {
//                Timer timer(true);
                b = getBlockSize(datax, datay, dataz, px, py, pz, rx, ry, rz, conf, blkflag);
//                double t = timer.stop();
//                printf("est time = %lf\n", t);
            } else {
                b = {bx_, by_, bz_};
            }

            size_t &bx = b[0], &by = b[1], &bz = b[2];
#if __OUTPUT_INFO
            for(int i=0;i<3;i++) printf("b%c = %zu\n",'x'+i,b[i]);
#endif

            // if eb is too small
            if (bx == 0 || by == 0 || bz == 0) {

                uchar *bytes_data = new uchar[conf.num * sizeof(T) * 3 + 1024];
                uchar *tail_data = bytes_data;

                SZ3::Config __conf = conf;
                __conf.save(tail_data);
                px = py = pz = 0;
                write(px, tail_data);
                write(py, tail_data);
                write(pz, tail_data);
                write(bx, tail_data);
                write(by, tail_data);
                write(bz, tail_data);

                memcpy(tail_data, datax, conf.num * sizeof(T));
                tail_data += conf.num * sizeof(T);
                memcpy(tail_data, datay, conf.num * sizeof(T));
                tail_data += conf.num * sizeof(T);
                memcpy(tail_data, dataz, conf.num * sizeof(T));
                tail_data += conf.num * sizeof(T);

                if (ord != nullptr) std::iota(ord, ord + conf.num, 0);

                uchar *lossless_data = lossless.compress(bytes_data, tail_data - bytes_data, compressed_size);
                delete[] bytes_data;

                return lossless_data;
            }

            size_t nx = ceil((rx) / (2. * bx * conf.absErrorBound)) + 1;
            size_t ny = ceil((ry) / (2. * by * conf.absErrorBound)) + 1;
            size_t nz = ceil((rz) / (2. * bz * conf.absErrorBound)) + 1;

            nx = (nx + 1) / 2;
            ny = (ny + 1) / 2;
            nz = (nz + 1) / 2;

#if __OUTPUT_INFO

            printf("nx = %zu\n",nx);
            printf("ny = %zu\n",ny);
            printf("nz = %zu\n",nz);

            printf("blknum with empty block = %zu\n",nx*ny*nz);

#endif

            // maybe use some tree data structure or sorting method to boost this process
            // 1. hashmap, then use faster sorting algos like Radix Sort
            // 2. multi tree?
//            std::map<size_t, std::vector<Node>> mp;
//            Node *vec = new Node[conf.num]{};
//            ska::unordered_map<size_t, std::vector<Node>> mp;

#if __OUTPUT_INFO

            printf("begin blocking\n");

#endif

            size_t blknum = 1;

            size_t *blkst = nullptr;
            size_t *blkcnt = nullptr;
            size_t *quads = new size_t[conf.num];
            size_t *repos = new size_t[conf.num];
//            size_t *reposs = new size_t[conf.num * 3], *reposx = reposs, *reposy = reposx + conf.num, *reposz = reposy + conf.num;
#if !__soft_eb
            size_t unid = nx * ny * nz + 1;
            std::vector<T> unx, uny, unz; // unqiantized data
#endif

            if (ord == nullptr) {

                Node *vec = new Node[conf.num]{};

                for (size_t i = 0; i < conf.num; i++) {

                    size_t x = (datax[i] - px) / (conf.absErrorBound);
                    x = (x + 0) / 2;
                    size_t y = (datay[i] - py) / (conf.absErrorBound);
                    y = (y + 0) / 2;
                    size_t z = (dataz[i] - pz) / (conf.absErrorBound);
                    z = (z + 0) / 2;

#if !__soft_eb

                    T decx = (x << 1 | 1) * conf.absErrorBound + px, decy =
                            (y << 1 | 1) * conf.absErrorBound + py, decz = (z << 1 | 1) * conf.absErrorBound + pz;

                    if (fabs(decx - datax[i]) > conf.absErrorBound || fabs(decy - datay[i]) > conf.absErrorBound ||
                        fabs(decz - dataz[i]) > conf.absErrorBound) {
                        unx.push_back(datax[i]);
                        uny.push_back(datay[i]);
                        unz.push_back(dataz[i]);
                        Node tem(unid, unx.size() - 1);
                        vec[i] = tem;
                        continue;
                    }
#endif

                    size_t cx = x / bx;
                    size_t dx = x % bx;
                    size_t cy = y / by;
                    size_t dy = y % by;
                    size_t cz = z / bz;
                    size_t dz = z % bz;

                    Node tem(cx / 2 + cy / 2 * nx + cz / 2 * nx * ny,
                             (dx + dy * bx + dz * bx * by) | ((cx & 1) << 60) | ((cy & 1) << 61) | ((cz & 1) << 62));
//                    Node tem(arr3(x / bx, y / by, z / bz), dx + dy * bx + dz * bx * by);
                    vec[i] = tem;
                }

//                Timer timer(true);

//                std::sort(vec,vec+conf.num,[&](Node& u, Node& v){return u.id==v.id?u.reid<v.reid:u.id<v.id;});
                radix_sort<Node>(vec, vec + conf.num);

//                double sort_time = timer.stop();

//                printf("sort time = %fs\n", sort_time);

                for (size_t i = 1; i < conf.num; i++) {
                    if (vec[i].id != vec[i - 1].id) ++blknum;
                }

                blkst = new size_t[blknum];
                blkcnt = new size_t[blknum]{};

                size_t i = -1;
                size_t j = 0;
                size_t pre = -1;
                size_t prequad = 0;
                size_t prereid = 0;
                for (; j < conf.num; j++) {
                    Node &node = vec[j];
                    size_t &id = node.id;
                    size_t quad = node.reid >> 60;
//                                              ++++----++++----
                    size_t reid = node.reid & 0x0fffffffffffffff;

                    if (id != pre) {
                        blkst[++i] = pre = id;
                        prequad = 0;
                        prereid = 0;
                    } else if (quad != prequad) {
                        prereid = 0;
                    }
                    ++blkcnt[i];

                    quads[j] = quad - prequad;
                    repos[j] = reid - prereid;
//                    reposx[j] = reid % bx;
//                    reposy[j] = reid / bx % by;
//                    reposz[j] = reid / bx / by;
                    prequad = quad;
                    prereid = reid;
                }

                delete[] vec;
            } else {

                NodeWithOrder *vec = new NodeWithOrder[conf.num]{};

                for (size_t i = 0; i < conf.num; i++) {

                    size_t x = (datax[i] - px) / (conf.absErrorBound);
                    x = (x + 0) / 2;
                    size_t y = (datay[i] - py) / (conf.absErrorBound);
                    y = (y + 0) / 2;
                    size_t z = (dataz[i] - pz) / (conf.absErrorBound);
                    z = (z + 0) / 2;

#if !__soft_eb
                    T decx = (x << 1 | 1) * conf.absErrorBound + px, decy =
                            (y << 1 | 1) * conf.absErrorBound + py, decz = (z << 1 | 1) * conf.absErrorBound + pz;

                    if (fabs(decx - datax[i]) > conf.absErrorBound || fabs(decy - datay[i]) > conf.absErrorBound ||
                        fabs(decz - dataz[i]) > conf.absErrorBound) {
                        unx.push_back(datax[i]);
                        uny.push_back(datay[i]);
                        unz.push_back(dataz[i]);
                        NodeWithOrder tem(unid, unx.size() - 1, i);
                        vec[i] = tem;
                        continue;
                    }
#endif

                    size_t cx = x / bx;
                    size_t dx = x % bx;
                    size_t cy = y / by;
                    size_t dy = y % by;
                    size_t cz = z / bz;
                    size_t dz = z % bz;

//                    if(datax[i] >= 0.229 && datax[i] <= 0.230)
//                        if(datay[i] >= 0.624 && datay[i] <= 0.625)
//                            if(dataz[i] >= 0.109 && dataz[i] <= 0.110){
//                                printf("i = %zu\n", i);
//                                printf("b %zu %zu %zu | n %zu %zu %zu\n", bx, by, bz, nx, ny, nz);
//                                printf("c %zu %zu %zu %zu | q %zu | d %zu %zu %zu %zu\n", cx / 2 + cy / 2 * nx + cz / 2 * nx * ny, cx / 2, cy / 2, cz / 2, ((cx & 1) << 0) | ((cy & 1) << 1) | ((cz & 1) << 2), (dx + dy * bx + dz * bx * by), dx, dy, dz);
//                            }

                    NodeWithOrder tem(cx / 2 + cy / 2 * nx + cz / 2 * nx * ny,
                                      (dx + dy * bx + dz * bx * by) | ((cx & 1) << 60) | ((cy & 1) << 61) |
                                      ((cz & 1) << 62), i);
                    vec[i] = tem;
                }

//                Timer timer(true);

//                std::sort(vec,vec+conf.num,[&](NodeWithOrder& u, NodeWithOrder& v){return u.id==v.id?u.reid<v.reid:u.id<v.id;});

//                size_t *reposOut = new size_t[conf.num];
//                for(size_t i=0;i<conf.num;i++){
//                    reposOut[i] = vec[i].reid & 0x0fffffffffffffff;
//                }
//                char buffer[1024];
//                sprintf(buffer, "/Users/longtaozhang/compress/repos/exaalt-83x1077290-eb=%.0e-blksz=%zux%zux%zu.txt", conf.absErrorBound, bx, by, bz);
//                writeTextFile(buffer, reposOut, conf.num);
//                delete[] reposOut;

                radix_sort<NodeWithOrder>(vec, vec + conf.num);

//                double sort_time = timer.stop();

//                printf("sort time = %fs\n", sort_time);

                for (size_t i = 0; i < conf.num; i++) {
                    ord[i] = vec[i].ord;
                }

                for (size_t i = 1; i < conf.num; i++) {
                    if (vec[i].id != vec[i - 1].id) ++blknum;
                }

                blkst = new size_t[blknum];
                blkcnt = new size_t[blknum]{};

                size_t i = -1;
                size_t j = 0;
                size_t pre = -1;
                size_t prequad = 0;
                size_t prereid = 0;
                for (; j < conf.num; j++) {
                    NodeWithOrder &node = vec[j];
                    size_t id = node.id;
                    size_t quad = node.reid >> 60;
//                                              ++++----++++----
                    size_t reid = node.reid & 0x0fffffffffffffff;

                    if (id != pre) {
                        blkst[++i] = pre = id;
                        prequad = 0;
                        prereid = 0;
                    } else if (quad != prequad) {
                        prereid = 0;
                    }
                    ++blkcnt[i];

                    quads[j] = quad - prequad;
                    repos[j] = reid - prereid;

//                    if(ord[j]==1){
//                        printf("j = %zu\n", j);
//                        printf("p %f %f %f | n %zu %zu %zu\n", px, py, pz, nx, ny, nz);
//                        printf("data %f %f %f | blkst %zu %zu %zu %zu | quad %zu | repos %zu\n", datax[ord[j]], datay[ord[j]], dataz[ord[j]], id, id%nx, id/nx%ny, id/nx/ny, quads[j], reid);
//                        printf("pre | %zu %zu\n", prequad, prereid);
//                    }

                    prequad = quad;
                    prereid = reid;
                }

                delete[] vec;
            }

//            printf("blknum = %zu, conf.num = %zu, ratio = %lf\n", blknum, conf.num, 1. * conf.num / blknum);

//            std::vector<std::vector<size_t>> reposlist(bx*by*bz);
//            for(size_t i=0;i<conf.num;i++){
//                reposlist[repos[i]].push_back(i);
//            }
//            for(size_t b=0;b<bx*by*bz;b++){
//                if(reposlist[b].size()>0)
//                for(size_t i=reposlist[b].size()-1;i>0;i--){
//                    reposlist[b][i] -= reposlist[b][i-1];
//                }
//            }

            // record the difference array

            for (size_t i = blknum - 1; i > 0; i--) {
                blkst[i] -= blkst[i - 1];
            }

            // use huffman encoder to compress the block bases
            uchar *bytes_blkst = new uchar[std::max(blknum * 8, (size_t) 1024)], *tail_blkst = bytes_blkst;
//            printf("+++flag = %.2lf\n",1.*nx*ny*nz/blknum);
            encoder.preprocess_encode(blkst, blknum, 0, (1. * nx * ny * nz / blknum > 1e6 ? 1 : 0));
            encoder.save(tail_blkst);
            encoder.encode(blkst, blknum, tail_blkst);

#if __OUTPUT_INFO

            printf("size of blkst = %.2lf MB, %zu bytes\n", 1. * (tail_blkst - bytes_blkst) / 1024 / 1024, tail_blkst - bytes_blkst);
            size_t cmpblkstSize;
            delete[] lossless.compress(bytes_blkst, tail_blkst - bytes_blkst, cmpblkstSize);
            printf("size of compressed blkst = %.2lf MB, %zu bytes\n", 1. * cmpblkstSize / 1024 / 1024, cmpblkstSize);

#endif

            delete[] blkst;

            // use huffman encoder to compress the number of points in each block

//            if(blkflag != 0x00){
//                size_t tem = 0;
//                for(size_t i=0;i<blknum;i++){
//                    tem = std::max(tem, *(blkcnt + i));
//                }
//                printf("%zu\n", tem);
//            }


            uchar *bytes_blkcnt = new uchar[std::max(blknum * 8, (size_t) 1024)], *tail_blkcnt = bytes_blkcnt;
            encoder.preprocess_encode(blkcnt, blknum, 0, 0xc0);
            encoder.save(tail_blkcnt);
            encoder.encode(blkcnt, blknum, tail_blkcnt);

#if __OUTPUT_INFO

            printf("size of blkcnt = %.2lf MB, %zu bytes\n", 1. * (tail_blkcnt - bytes_blkcnt) / 1024 / 1024, tail_blkcnt - bytes_blkcnt);
            size_t cmpblkcntSize;
            delete[] lossless.compress(bytes_blkcnt, tail_blkcnt - bytes_blkcnt, cmpblkcntSize);
            printf("size of compressed blkcnt = %.2lf MB, %zu bytes\n", 1. * cmpblkcntSize / 1024 / 1024, cmpblkcntSize);

#endif

            delete[] blkcnt;

//            for(size_t i=0;i<conf.num;i++){
//                printf("%zu", quads[i]);
//            }
//            printf("\n");

            uchar *bytes_quads = new uchar[std::max((size_t) ceil(conf.num * 0.4),
                                                    (size_t) 1024)], *tail_quads = bytes_quads;
            encoder.preprocess_encode(quads, conf.num, 8, 0xc1);
            encoder.save(tail_quads);
            encoder.encode(quads, conf.num, tail_quads);

#if __OUTPUT_INFO

            //            printf("size of quads = %.2lf MB, %zu bytes\n", 1. * (tail_quads - bytes_quads) / 1024 / 1024, tail_quads - bytes_quads);
                        size_t cmpQuadsSize;
                        delete[] lossless.compress(bytes_quads, tail_quads - bytes_quads, cmpQuadsSize);
            //            printf("size of compressed blkcnt = %.2lf MB, %zu bytes\n", 1. * cmpQuadsSize / 1024 / 1024, cmpQuadsSize);

#endif

            delete[] quads;

            // use huffman encoder to encode the relative error of each point

            uchar *bytes_repos = new uchar[std::max(
                    conf.num * std::max((size_t) 4, (size_t) ceil(log2(1. * bx * by * bz))),
                    (size_t) 1024)], *tail_repos = bytes_repos;
            encoder.preprocess_encode(repos, conf.num, 0, 0xc1);
            encoder.save(tail_repos);
            encoder.encode(repos, conf.num, tail_repos);

//            uchar *bytes_repos = new uchar[std::max(conf.num * std::max((size_t)16, (size_t)ceil(log2(1. * bx * by * bz))), (size_t)1024)], *tail_repos = bytes_repos;
////            for(size_t i=0;i<conf.num;i++) printf("%zu", reposx[i]);
//            encoder.preprocess_encode(reposx, conf.num, 0, 0xc1);
//            encoder.save(tail_repos);
//            encoder.encode(reposx, conf.num, tail_repos);
//            encoder.preprocess_encode(reposy, conf.num, 0, 0xc1);
//            encoder.save(tail_repos);
//            encoder.encode(reposy, conf.num, tail_repos);
//            encoder.preprocess_encode(reposz, conf.num, 0, 0xc1);
//            encoder.save(tail_repos);
//            encoder.encode(reposz, conf.num, tail_repos);

//            size_t *poslist = new size_t[conf.num];
//
//            for(int b=0;b<nx*ny*nz;b++){
//                int64ToBytes_bigEndian(tail_repos, reposlist[b].size());
//                tail_repos += 8;
//            }
//
//            pos=0;
//            for(int b=0;b<nx*ny*nz;b++){
//                for(size_t it:reposlist[b]) poslist[pos++]=it;
//            }
//
//            encoder.preprocess_encode(poslist, conf.num, 0);
//            encoder.save(tail_repos);
//            encoder.encode(poslist, conf.num, tail_repos);
//
//            delete[] poslist;

#if __OUTPUT_INFO

            printf("size of repos = %.2lf MB, %zu bytes\n", 1. * (tail_repos - bytes_repos + (tail_quads - bytes_quads)) / 1024 / 1024, tail_repos - bytes_repos + (tail_quads - bytes_quads));
            size_t cmpreposSize;
            delete[] lossless.compress(bytes_repos, tail_repos - bytes_repos, cmpreposSize);
            printf("size of compressed repos = %.2lf MB, %zu bytes\n", 1. * (cmpreposSize + cmpQuadsSize) / 1024 / 1024, cmpreposSize + cmpQuadsSize);

#endif

//            writefile("/Users/longtaozhang/compress/repos_file/hacc-33554432-eb=1e-3-combine.dat", repos, conf.num);
//            writefile("/Users/longtaozhang/compress/repos_file/hacc-33554432-eb=1e-3-separate.dat", reposs, 3 * conf.num);

            delete[] repos;

#if __OUTPUT_INFO

            printf("begin merge\n");

#endif

#if !__soft_eb
            uchar *bytes_data = new uchar[std::max(conf.num * 16, (size_t) 1024) +
                                          unx.size() * 3 * sizeof(T)], *tail_data = bytes_data;
#endif

#if __soft_eb
            uchar *bytes_data = new uchar[std::max(conf.num*16, (size_t)1024)], *tail_data = bytes_data;
#endif

            // write the basic info

            SZ3::Config __conf = conf;
            __conf.save(tail_data);
            write(px, tail_data);
            write(py, tail_data);
            write(pz, tail_data);
            write(bx, tail_data);
            write(by, tail_data);
            write(bz, tail_data);
            write(nx, tail_data);
            write(ny, tail_data);
            write(nz, tail_data);
            write(blknum, tail_data);

            // write the codes of the above 3 arrays

            write(bytes_blkst, tail_blkst - bytes_blkst, tail_data);
            delete[] bytes_blkst;
            write(bytes_blkcnt, tail_blkcnt - bytes_blkcnt, tail_data);
            delete[] bytes_blkcnt;
            write(bytes_quads, tail_quads - bytes_quads, tail_data);
            delete[] bytes_quads;
            write(bytes_repos, tail_repos - bytes_repos, tail_data);
            delete[] bytes_repos;

#if __OUTPUT_INFO

            printf("end merge\n");

#endif

#if !__soft_eb
//            printf("unpred = %zu\n", unx.size());
            write(unx.size(), tail_data);
            write(unx.data(), unx.size(), tail_data);
            write(uny.data(), uny.size(), tail_data);
            write(unz.data(), unz.size(), tail_data);
#endif

            uchar *lossless_data = lossless.compress(bytes_data, tail_data - bytes_data, compressed_size);
            delete[] bytes_data;

#if __OUTPUT_INFO

            printf("total bytes = %zu\n",compressed_size);

#endif

            return lossless_data;
        }

        /*
         * To arrange the data by using the ord,
         *
         * Make sure ord is a permutation starting from 0
         * After arrangement, the pointer won't change
         */

        void arrageByOrder(T *data, size_t n, size_t *ord) {

            static T *a = nullptr;
            if (a == nullptr) a = new T[n];

            for (size_t i = 0; i < n; i++) {
                a[i] = data[ord[i]];
//                if(ord[i] < 0 || ord[i] >= n) printf("ord[%zu] = %zu\n", i, ord[i]), exit(-1);
            }

            memcpy(data, a, n * sizeof(T));
        }

        class ErrorCodingLengthCache {
        public:

            ErrorCodingLengthCache() {
                f = -1;
                c = 0;
                lossless = Lossless_zstd();
            }

            uchar isHit() {
                return false;
                if (f > 0) {
                    if (c > 0) {
                        --c;
                        return 0x01;
                    } else {
                        return 0x00;
                    }
                }
                return 0x00;
            }

            double get() {
                return f;
            }

            // return the number of non-quantized values
            size_t
            write(const Config &conf, LinearQuantizer<T> &quantizer, const T *nowpx, const T *nowpy, const T *nowpz,
                  const T *prepx, const T *prepy, const T *prepz) {
                const size_t &n = conf.dims[1];
                size_t fail = 0, total = 3 * n;

                size_t *err = new size_t[3 * n], *errx = err, *erry = errx + n, *errz = erry + n;
                uchar *bytes = new uchar[12 * n], *tail = bytes;

                for (size_t i = 0; i < n; i++) {
                    errx[i] = quantizer.quantize(nowpx[i], prepx[i]);
                    if (errx[i] == 0) ++fail;
                }
                for (size_t i = 0; i < n; i++) {
                    erry[i] = quantizer.quantize(nowpy[i], prepy[i]);
                    if (erry[i] == 0) ++fail;
                }
                for (size_t i = 0; i < n; i++) {
                    errz[i] = quantizer.quantize(nowpz[i], prepz[i]);
                    if (errz[i] == 0) ++fail;
                }

                static HuffmanEncoder<size_t> encoder;

                encoder.preprocess_encode(err, 3 * n, 0);
                encoder.save(tail);
                encoder.encode(err, 3 * n, tail);

                size_t cmpSize;
                delete[] lossless.compress(bytes, tail - bytes, cmpSize);

                f = 1. * cmpSize / (total - fail);
//                f = 1. * (tail - bytes) / (total - fail);

//                printf("f = %lf\n", f);

                delete[] err;
                delete[] bytes;

                c = 16;

                return fail;
            }

        private:
            double f;
            size_t c;
            Lossless lossless;
        };

        ErrorCodingLengthCache errorCodingLengthCache;

        class IsTPCache {
        public:
            IsTPCache() {
                cnt = 0;
                nxtcnt = 0;
                flag = 0x01;
            }

            uchar isHit() {
                return false;
                if (cnt > 0) {
                    --cnt;
                    return 0x01;
                } else {
                    return 0x00;
                }
            }

            uchar get() {
                return flag;
            }

            void write(uchar f) {
                flag = f;
                cnt = nxtcnt;
                nxtcnt <<= 1;
                if (nxtcnt == 0) nxtcnt = 1;
            }

            void clear() {
                IsTPCache();
            }

        private:
            int64_t cnt;
            int64_t nxtcnt;
            uchar flag;
        };

        IsTPCache isTpCache;

        uchar isTP(const Config &conf, LinearQuantizer<T> &quantizer, const T *nowpx, const T *nowpy, const T *nowpz,
                   const T *prepx, const T *prepy, const T *prepz, size_t cmpSize, uchar newBatchFlag) {

//            return 0x01;

            if (newBatchFlag == 0x01) {
                isTpCache.clear();
            }

            if (isTpCache.isHit()) {
                return isTpCache.get();
            }

            if (prepx == nullptr || prepy == nullptr || prepz == nullptr) {
                return false;
            }

            const size_t &n = conf.dims[1];
            size_t fail = 0, total = n * 3;

            if (!errorCodingLengthCache.isHit()) {
                fail = errorCodingLengthCache.write(conf, quantizer, nowpx, nowpy, nowpz, prepx, prepy, prepz);
            } else {
                for (size_t i = 0; i < n; i++) {
                    if (quantizer.quantize(nowpx[i], prepx[i]) == 0) ++fail;
                }
                for (size_t i = 0; i < n; i++) {
                    if (quantizer.quantize(nowpy[i], prepy[i]) == 0) ++fail;
                }
                for (size_t i = 0; i < n; i++) {
                    if (quantizer.quantize(nowpz[i], prepz[i]) == 0) ++fail;
                }
            }

//            printf("total = %zu\n", total);
//            printf("fail ratio = %.2lf%%, fail = %zu\n", 100. * fail / total, fail);
//            printf("%.0f %zu\n", (total - fail) * errorCodingLengthCache.get() + fail * 4, cmpSize);
            uchar res = (total - fail) * errorCodingLengthCache.get() + fail * 4 < cmpSize ? 0x01 : 0x00;
            isTpCache.write(res);
            return res;
        }

        /*
        * To compress the data from datax, datay and dataz using configure conf and buffer size bt
        * Store the result in the return pointer, and store the size compressed data in compressed_data
        *
        * after decompress, datax, datay, dataz will not increase
        */

        uchar *compressSimpleBlockingWithTemporalPredictionOnSlice(const Config &conf, T *datax, T *datay, T *dataz,
                                                                   size_t &compressed_size, size_t *ord, uchar blkflag,
                                                                   size_t bx, size_t by, size_t bz, uchar *&bytes1,
                                                                   size_t &compressed_size1, size_t *ord1,
                                                                   T fflag) {

            static const int64_t radius = (1 << 15);

            LinearQuantizer<T> quantizer(conf.absErrorBound, radius);

            size_t nt = conf.dims[0];
            size_t n = conf.dims[1];

            uchar *bytes = new uchar[std::max(conf.num * 16, (size_t) 65536)], *tailp = bytes, *tail = bytes + nt;

            Config conf1(n);
            conf1.absErrorBound = conf.absErrorBound / fflag;
            T fflag_stride = std::min(std::max((fflag - 1) / (nt / (T) 4), (T) 0), (T) 2);
//            printf("fflag stride = %.2f\n", fflag_stride);

            size_t *p = new size_t[n];

//            printf("nt = %zu, n = %zu\n", nt, n);

//            const uchar *bytes1 = compressSimpleBlocking(conf1, datax, datay, dataz, compressed_size, p, blkflag, bx, by, bz);
//            if(ord != nullptr){
//                memcpy(ord, p, n * sizeof(size_t));
//            }
//            write(compressed_size, tail);
//            memcpy(tail, bytes1, compressed_size);
//            tail += compressed_size;
//
//            T *decmpdata1 = new T[n * 3];
//            T *d1x = decmpdata1, *d1y = d1x + n, *d1z = d1y + n;
//            size_t outSize;
//            decompressSimpleBlocking(bytes1, d1x, d1y, d1z, outSize, compressed_size);
//            delete[] bytes1;

            T *pdx = new T[nt * n];
            size_t *errx = new size_t[nt * n];
            T *pdy = new T[nt * n];
            size_t *erry = new size_t[nt * n];
            T *pdz = new T[nt * n];
            size_t *errz = new size_t[nt * n];
            size_t errlen = 0;

            T *decmpdata1 = new T[n * 3];
            T *d1x = decmpdata1, *d1y = d1x + n, *d1z = d1y + n;
            size_t outSize;

            // to check if to create a new time frame or not

            if (bytes1 == nullptr) {

#if __batch_info
                printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", 0);
#endif

                writeBytesByte(tailp, 0x00);

                bytes1 = compressSimpleBlocking(conf1, datax, datay, dataz, compressed_size1, p, blkflag, bx, by, bz);
                if (ord != nullptr) {
                    memcpy(ord, p, n * sizeof(size_t));
                }
                memcpy(ord1, p, n * sizeof(size_t));
                const uchar *bytes1c = bytes1;
                decompressSimpleBlocking(bytes1c, d1x, d1y, d1z, outSize, compressed_size1);

                memcpy(pdx, d1x, n * sizeof(T));
                memcpy(pdy, d1y, n * sizeof(T));
                memcpy(pdz, d1z, n * sizeof(T));
            } else {

                const uchar *bytes1c = bytes1;
                decompressSimpleBlocking(bytes1c, d1x, d1y, d1z, outSize, compressed_size1);

//                memcpy(pdx, datax, n * sizeof(T));
//                memcpy(pdy, datay, n * sizeof(T));
//                memcpy(pdz, dataz, n * sizeof(T));

                for (size_t i = 0; i < n; i++) pdx[i] = datax[ord1[i]];
                for (size_t i = 0; i < n; i++) pdy[i] = datay[ord1[i]];
                for (size_t i = 0; i < n; i++) pdz[i] = dataz[ord1[i]];

                if (isTP(conf, quantizer, pdx, pdy, pdz, d1x, d1y, d1z, compressed_size1, 0x01)) {

#if __batch_info
                    printf("\e[32m\e[1mold batch, t = %zu\n\e[0m", 0);
#endif

                    writeBytesByte(tailp, 0x01);

                    for (size_t i = 0; i < n; i++) {
                        errx[i] = quantizer.quantize_and_overwrite(pdx[i], d1x[i]);
                    }
                    for (size_t i = 0; i < n; i++) {
                        erry[i] = quantizer.quantize_and_overwrite(pdy[i], d1y[i]);
                    }
                    for (size_t i = 0; i < n; i++) {
                        errz[i] = quantizer.quantize_and_overwrite(pdz[i], d1z[i]);
                    }
                    errlen = n;

                    if (ord != nullptr) {
                        memcpy(ord, ord1, n * sizeof(size_t));
                    }
                    memcpy(p, ord1, n * sizeof(size_t));
                } else {

                    if(fflag > 1){
                        fflag -= fflag_stride;
                        if(fflag < 1) fflag = 1;
                        conf1.absErrorBound = conf.absErrorBound / fflag;
                    }

#if __batch_info
                    printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", 0);
#endif

                    writeBytesByte(tailp, 0x00);

                    bytes1 = compressSimpleBlocking(conf1, datax, datay, dataz, compressed_size1, p, blkflag, bx, by,
                                                    bz);
                    if (ord != nullptr) {
                        memcpy(ord, p, n * sizeof(size_t));
                    }
                    memcpy(ord1, p, n * sizeof(size_t));
                    const uchar *bytes1c = bytes1;
                    decompressSimpleBlocking(bytes1c, pdx, pdy, pdz, outSize, compressed_size1);

//                    for(size_t i=0;i<n;i++) pdx[i] = d1x[i];
//                    for(size_t i=0;i<n;i++) pdy[i] = d1y[i];
//                    for(size_t i=0;i<n;i++) pdz[i] = d1z[i];
                }
            }

            compressed_size = compressed_size1;

//            for(size_t i=0;i<n;i++) pdx[i] = datax[ord1[i]];


//            memcpy(pdx, d1x, n * sizeof(T));
//            memcpy(pdy, d1y, n * sizeof(T));
//            memcpy(pdz, d1z, n * sizeof(T));

            for (size_t t = 1; t < nt; t++) {

                T *nowpx = pdx + t * n;
                T *nowpy = pdy + t * n;
                T *nowpz = pdz + t * n;
                T *prepx = nowpx - n;
                T *prepy = nowpy - n;
                T *prepz = nowpz - n;

//                for(size_t i=0;i<n;i++){
//                    nowpx[i] = datax[t * n + i];
//                    nowpy[i] = datay[t * n + i];
//                    nowpz[i] = dataz[t * n + i];
//                }
//                arrageByOrder(nowpx, n, p);
//                arrageByOrder(nowpy, n, p);
//                arrageByOrder(nowpz, n, p);
                for (size_t i = 0; i < n; i++) nowpx[i] = datax[t * n + p[i]];
                for (size_t i = 0; i < n; i++) nowpy[i] = datay[t * n + p[i]];
                for (size_t i = 0; i < n; i++) nowpz[i] = dataz[t * n + p[i]];

                if (isTP(conf, quantizer, nowpx, nowpy, nowpz, prepx, prepy, prepz, compressed_size,
                         t == 1 ? 0x01 : 0x00)) {

#if __batch_info
                    printf("\e[32m\e[1mold batch, t = %zu\n\e[0m", t);
#endif

                    writeBytesByte(tailp, 0x01);

                    size_t *errpx = errx + errlen;
                    size_t *errpy = erry + errlen;
                    size_t *errpz = errz + errlen;

                    for (size_t i = 0; i < n; i++) {
                        T &nowx = nowpx[i];
                        T &prex = prepx[i];
                        errpx[i] = quantizer.quantize_and_overwrite(nowx, prex);
                    }
                    for (size_t i = 0; i < n; i++) {
                        T &nowy = nowpy[i];
                        T &prey = prepy[i];
                        errpy[i] = quantizer.quantize_and_overwrite(nowy, prey);
                    }
                    for (size_t i = 0; i < n; i++) {
                        T &nowz = nowpz[i];
                        T &prez = prepz[i];
                        errpz[i] = quantizer.quantize_and_overwrite(nowz, prez);
                    }

                    errlen += n;
                } else {

                    if(fflag > 1){
                        fflag -= fflag_stride;
                        if(fflag < 1) fflag = 1;
                        if(nt - t <= 2) fflag = 1;
                        conf1.absErrorBound = conf.absErrorBound / fflag;
                    }

#if __batch_info
                    printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", t);
#endif

                    writeBytesByte(tailp, 0x00);

                    const uchar *bytes1 = compressSimpleBlocking(conf1, datax + t * n, datay + t * n, dataz + t * n,
                                                                 compressed_size, p, blkflag, bx, by, bz);
                    write(compressed_size, tail);
                    memcpy(tail, bytes1, compressed_size);
                    tail += compressed_size;

                    decompressSimpleBlocking(bytes1, d1x, d1y, d1z, outSize, compressed_size);
                    delete[] bytes1;

                    if (t + 1 < nt) {
                        memcpy(pdx + t * n, d1x, n * sizeof(T));
                        memcpy(pdy + t * n, d1y, n * sizeof(T));
                        memcpy(pdz + t * n, d1z, n * sizeof(T));
                    }
                }

                if (ord != nullptr) {
                    for (size_t i = 0; i < n; i++) {
                        ord[t * n + i] = t * n + p[i];
                    }
                }
            }

            size_t *err = new size_t[errlen * 3];
            memcpy(err, errx, errlen * sizeof(size_t));
            memcpy(err + errlen, erry, errlen * sizeof(size_t));
            memcpy(err + errlen + errlen, errz, errlen * sizeof(size_t));
            encoder.preprocess_encode(err, errlen * 3, 0);
            encoder.save(tail);
            encoder.encode(err, errlen * 3, tail);
            delete[] err;

//            encoder.preprocess_encode(errx, errlen, 0);
//            encoder.save(tail);
//            encoder.encode(errx, errlen, tail);
//
////            encoder.preprocess_encode(erry, errlen, 0);
////            encoder.save(tail);
////            encoder.encode(erry, errlen, tail);
////
////            encoder.preprocess_encode(errz, errlen, 0);
////            encoder.save(tail);
////            encoder.encode(errz, errlen, tail);

            quantizer.save(tail);

            delete[] decmpdata1;
            delete[] p;
            delete[] pdx;
            delete[] errx;
            delete[] pdy;
            delete[] erry;
            delete[] pdz;
            delete[] errz;

            uchar *lossless_data = lossless.compress(bytes, tail - bytes, compressed_size);
            delete[] bytes;

            return lossless_data;
        }

        uchar isSpatialWorse(const Config conf, T *datax, T *datay, T *dataz){

            const size_t& n = conf.dims[1];
            LinearQuantizer<T> quantizer(conf.absErrorBound, (1 << 15));

            size_t *err = new size_t[3 * n], *errx = err, *erry = errx + n, *errz = erry + n;
            uchar *bytes = new uchar[12 * n], *tail = bytes;

            T *nowpx = datax, *nowpy = datay, *nowpz = dataz;
            T *prepx = datax - n, *prepy = datay - n, *prepz = dataz - n;

            for (size_t i = 0; i < n; i++) {
                errx[i] = quantizer.quantize(nowpx[i], prepx[i]);
            }
            for (size_t i = 0; i < n; i++) {
                erry[i] = quantizer.quantize(nowpy[i], prepy[i]);
            }
            for (size_t i = 0; i < n; i++) {
                errz[i] = quantizer.quantize(nowpz[i], prepz[i]);
            }

            HuffmanEncoder<size_t> encoder;

            encoder.preprocess_encode(err, 3 * n, 0);
            encoder.save(tail);
            encoder.encode(err, 3 * n, tail);

            size_t cmpSizeTemporal;
            delete[] lossless.compress(bytes, tail - bytes, cmpSizeTemporal);

            delete[] err;
            delete[] bytes;

            Config conf1 = Config(n);
            conf1.absErrorBound = conf.absErrorBound;
            size_t cmpSizeSpatial;

            compressSimpleBlocking(conf1, datax, datay, dataz, cmpSizeSpatial);

            return cmpSizeTemporal < cmpSizeSpatial;
        }

        /*
         * To compress the data from datax, datay and dataz using configure conf and buffer size bt
         * Store the result in the return pointer, and store the size compressed data in compressed_data
         *
         * after decompress, datax, datay, dataz will not increase
         */

        uchar *compressSimpleBlockingWithTemporalPrediction(const Config &conf, size_t bt, T *datax, T *datay, T *dataz,
                                                            size_t &compressed_size, size_t *ord = nullptr,
                                                            uchar blkflag = 0x00, size_t bx = 0, size_t by = 0,
                                                            size_t bz = 0) {

            assert(conf.N == 2);

            size_t nt = conf.dims[0];
            size_t n = conf.dims[1];

            T fflag = 1;

            if (nt >= 64 && n >= (1 << 16)) {
                size_t fail = 0, total = 0;
                T radius = (1 << 16) * conf.absErrorBound;
                for (size_t i = 0; i < n; i += n / 100) {
                    uchar flag = 0x00;
                    size_t j = n + i, k = i;
                    for (size_t t = 1; t < nt; t++) {
                        if (std::abs(datax[j] - datax[k]) > radius || std::abs(datay[j] - datay[k]) > radius ||
                            std::abs(dataz[j] - dataz[k]) > radius) {
                            flag = 0x01;
                        }
                    }
                    fail += flag;
                    ++total;
                }
                if (fail == 0) {
                    size_t t = (nt - 1) / 2;
                    if (isSpatialWorse(conf, datax + t * n, datay + t * n, dataz + t * n)){
                        fflag = 6;
                    }
                    blockSizeCache.init();
                }
            }

            compressed_size = (nt / bt + (nt % bt > 0 ? 1 : 0) + 4) * sizeof(size_t);

            uchar *bytes = new uchar[std::max(conf.num * 16, (size_t) 65536)], *tailpos = bytes, *tail =
                    bytes + compressed_size;
            uchar *bytes1s = new uchar[std::max(conf.num * 16, (size_t) 65536)], *tail1s =
                    bytes1s + sizeof(int); // sizeof(cnt1)

            write(nt, tailpos);
            write(n, tailpos);
            write(bt, tailpos);

            Config confSlice = Config(bt, n);
            confSlice.absErrorBound = conf.absErrorBound;

            uchar *bytes1 = nullptr;
            size_t compressed_size1 = 0;
            size_t *ord1 = new size_t[n];
            int cnt1 = -1;

            for (size_t l = 0; l < nt; l += bt) {

                size_t r = l + bt;
                if (r > nt) r = nt, confSlice = Config(r - l, n), confSlice.absErrorBound = conf.absErrorBound;

//                printf("%zu %zu\n", l, r);

                size_t slice_compressed_size = 0;

                uchar *bytes1p = bytes1;

                if (nt - l < 64) fflag = 0x01;

                uchar *sliceBytes = compressSimpleBlockingWithTemporalPredictionOnSlice(confSlice, datax + l * n,
                                                                                        datay + l * n, dataz + l * n,
                                                                                        slice_compressed_size,
                                                                                        ord == nullptr ? nullptr : ord +
                                                                                                                   l *
                                                                                                                   n,
                                                                                        blkflag, bx, by, bz, bytes1,
                                                                                        compressed_size1, ord1, fflag);

                if (bytes1 != bytes1p) {

                    ++cnt1;
                    write(compressed_size1, tail1s);
                    write(bytes1, compressed_size1, tail1s);
                    delete[] bytes1p;
                }

                if (ord != nullptr) {
                    for (size_t t = l; t < r; t++) {
                        for (size_t i = 0; i < n; i++) {
                            ord[t * n + i] += l * n;
                        }
                    }
                }


                write(cnt1, tail);
                write(sliceBytes, slice_compressed_size, tail);
                write(compressed_size + sizeof(cnt1), tailpos);

                compressed_size += sizeof(cnt1) + slice_compressed_size;

                delete[] sliceBytes;
            }

            write(cnt1 + 1, bytes1s);
            bytes1s -= sizeof(cnt1);

            write(compressed_size, tailpos);
            write(bytes, tail - bytes, tail1s);
            delete[] bytes;

            if (bytes1 != nullptr) delete[] bytes1;
            delete[] ord1;

            compressed_size = tail1s - bytes1s;

            return bytes1s;
        }

        uchar *compressSimpleBlocking(const Config &conf, T *data, size_t &compressed_size, size_t *ord = nullptr,
                                      uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

            return compressSimpleBlocking(conf, data, data + conf.num, data + conf.num + conf.num, compressed_size, ord,
                                          blkflag, bx, by, bz);
        }

        //std::vector<std::vector<T>>& data
        /*
        uchar *compressKDtreeBlocking(const Config& conf, std::vector<T> *data, size_t& compressed_size) {

            printf("begin to prequantize\n");

            std::vector<std::vector<int64_t>> prequantized_data(conf.num,std::vector<int64_t>(3));
            for(int i=0;i<conf.num;i++){

                int64_t dx = data[i][0] / (2. * conf.absErrorBound);
                dx = (dx + 1) / 2;
                int64_t dy = data[i][1] / (2. * conf.absErrorBound);
                dy = (dy + 1) / 2;
                int64_t dz = data[i][2] / (2. * conf.absErrorBound);
                dz = (dz + 1) / 2;

                prequantized_data[i][0] = dx;
                prequantized_data[i][1] = dy;
                prequantized_data[i][2] = dz;
            }

            int64_t bx=8,by=8,bz=8;

            printf("begin allocate memory for KD tree\n");

            KDTree<int64_t> kdt(bx,by,bz);

            printf("begin to build KD tree\n");

            kdt.build(prequantized_data.data(),conf.num);

            printf("begin to decorate\n");

            kdt.decorate();

            printf("begin to save\n");

            uchar *bytes = new uchar[conf.num*16], *tail = bytes;
            kdt.save(tail);
            kdt.clear();

            printf("tail - bytes = %zu\n",tail-bytes);

            uchar *lossless_data = lossless.compress(bytes, tail-bytes, compressed_size);

            delete[] bytes;

            return lossless_data;
        }
        */

        uchar *compress(const Config &conf, T *data, size_t &compressed_size, size_t *ord = nullptr,
                        uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

            return compressSimpleBlocking(conf, data, compressed_size, ord, blkflag, bx, by, bz);
        }

        uchar *compress(const Config &conf, T *datax, T *datay, T *dataz, size_t &compressed_size,
                        size_t *ord = nullptr, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

            return compressSimpleBlocking(conf, datax, datay, dataz, compressed_size, ord, blkflag, bx, by, bz);
        }

        /*
         * To decompress the data from lossless_data
         * Store the results in datax, datay, dataz, if any of them is nullptr, automatically allocate memory
         * The number of elements will be stored in outSize
         *
         * after decompression, lossless will not increase
         * datax, datay, dataz will not increase
         */

        void decompressSimpleBlocking(const uchar *&lossless_data, T *&datax, T *&datay, T *&dataz, size_t &outSize,
                                      size_t cmpSize) {

            uchar const *cmpData;
//            if(cmpSize > 0){
            cmpData = lossless.decompress(lossless_data, cmpSize);
//            }
//            else{
//                cmpData = lossless_data;
//            }

            SZ3::Config conf;
            conf.load(cmpData);
            outSize = conf.num;

            if (datax == nullptr && datay == nullptr && dataz == nullptr) {
                datax = new T[3 * conf.num];
                datay = datax + conf.num;
                dataz = datay + conf.num;
            }

            if (datax == nullptr) datax = new T[conf.num];
            if (datay == nullptr) datay = new T[conf.num];
            if (dataz == nullptr) dataz = new T[conf.num];

            T px, py, pz;
            size_t bx, by, bz, nx, ny, nz, blknum;

            read(px, cmpData);
            read(py, cmpData);
            read(pz, cmpData);

            read(bx, cmpData);
            read(by, cmpData);
            read(bz, cmpData);

            if (bx == 0 || by == 0 || bz == 0) {

                memcpy(datax, cmpData, conf.num * sizeof(T));
                for (size_t i = 0; i < conf.num; i++) datax[i] += px;
                memcpy(datay, cmpData + conf.num * sizeof(T), conf.num * sizeof(T));
                for (size_t i = 0; i < conf.num; i++) datay[i] += py;
                memcpy(dataz, cmpData + (conf.num + conf.num) * sizeof(T), conf.num * sizeof(T));
                for (size_t i = 0; i < conf.num; i++) dataz[i] += pz;

                return;
            }

            read(nx, cmpData);
            read(ny, cmpData);
            read(nz, cmpData);
            read(blknum, cmpData);

            size_t remaining_length = 0;

            encoder.load(cmpData, remaining_length);
            auto blkst = encoder.decode(cmpData, blknum);

            encoder.load(cmpData, remaining_length);
            auto blkcnt = encoder.decode(cmpData, blknum);

            encoder.load(cmpData, remaining_length);
            auto quads = encoder.decode(cmpData, conf.num);

            encoder.load(cmpData, remaining_length);
            auto repos = encoder.decode(cmpData, conf.num);

#if !__soft_eb
            size_t cnt_unquants, unid = nx * ny * nz + 1;
            read(cnt_unquants, cmpData);
            std::vector<T> unx(cnt_unquants), uny(cnt_unquants), unz(cnt_unquants);
            read(unx.data(), cnt_unquants, cmpData);
            read(uny.data(), cnt_unquants, cmpData);
            read(unz.data(), cnt_unquants, cmpData);
#endif

            size_t i = 0, j = 0;
            for (; i < blknum; i++) {

                if (i) blkst[i] += blkst[i - 1];

#if !__soft_eb

                if (blkst[i] == unid) {

                    for (size_t j_ = 0; j_ < blkcnt[i]; j_++) {
                        datax[j] = unx[j_];
                        datay[j] = uny[j_];
                        dataz[j] = unz[j_];
                        ++j;
                    }
                    continue;
                }
#endif

                size_t pbx = (blkst[i] % nx * 2) * bx;
                size_t pby = (blkst[i] / nx % ny * 2) * by;
                size_t pbz = (blkst[i] / nx / ny * 2) * bz;

                size_t prequad = 0;
                size_t prerepos = 0;

                for (size_t j_ = 0; j_ < blkcnt[i]; j_++) {

                    if (quads[j] != 0) prerepos = 0;
                    size_t reposj = repos[j] + prerepos;
                    size_t quadj = quads[j] + prequad;
                    prerepos = reposj;
                    prequad = quadj;

                    size_t idx = (pbx + ((quadj & 0x01) >> 0) * bx + (reposj % bx));
                    datax[j] = (idx << 1 | 1) * conf.absErrorBound + px;
                    size_t idy = (pby + ((quadj & 0x02) >> 1) * by + (reposj / bx % by));
                    datay[j] = (idy << 1 | 1) * conf.absErrorBound + py;
                    size_t idz = (pbz + ((quadj & 0x04) >> 2) * bz + (reposj / bx / by));
                    dataz[j] = (idz << 1 | 1) * conf.absErrorBound + pz;

                    ++j;
                }
            }

            return;
        }

        void
        decompressSimpleBlockingWithTemporalPrediction(const uchar *&lossless_data, T *&datax, T *&datay, T *&dataz,
                                                       size_t &outSize, size_t cmpSize) {

            int cnt1;
            read(cnt1, lossless_data);

            std::vector<const uchar *> st(cnt1);
            std::vector<size_t> stsz(cnt1);

            for (int i = 0; i < cnt1; i++) {
                read(stsz[i], lossless_data);
                st[i] = lossless_data;
                lossless_data += stsz[i];
            }

            size_t nt = 0, n = 0, bt = 0;
            read(nt, lossless_data);
            read(n, lossless_data);
            read(bt, lossless_data);

            size_t nbt = nt / bt + (nt % bt > 0 ? 1 : 0);

            const uchar *bytes = nullptr, *tailp = nullptr, *tail = nullptr;
            size_t *pos = new size_t[nbt + 1];
            memcpy(pos, lossless_data, (nbt + 1) * sizeof(size_t));
            lossless_data -= 3 * sizeof(size_t);

//            for(size_t i=0;i<=nbt;i++){
//
//                printf("pos[%zu] = %zu\n", i, pos[i]);
//            }

            T *tailx = datax, *taily = datay, *tailz = dataz;

            LinearQuantizer<T> quantizer;

            for (size_t l = 0; l < nt; l += bt) {

                int stid;
                memcpy(&stid, lossless_data + pos[l / bt] - sizeof(stid), sizeof(stid));

                size_t r = std::min(l + bt, nt);
                size_t cmpSize = pos[l / bt + 1] - pos[l / bt];
                bytes = lossless.decompress(lossless_data + pos[l / bt], cmpSize);
                tailp = bytes;
                tail = tailp + r - l;

                size_t cnt0 = 0;

                {
                    const uchar *ttailp = tailp;
                    T *ttailx = tailx, *ttaily = taily, *ttailz = tailz;

                    {
                        uchar flag = *(ttailp++);
                        if (flag == 0x00) {
                            size_t outSize = 0;
                            decompressSimpleBlocking(st[stid], ttailx, ttaily, ttailz, outSize, stsz[stid]);
                            ++cnt0;
                        }
                        ttailx += n;
                        ttaily += n;
                        ttailz += n;
                    }

                    for (size_t t = l + 1; t < r; t++) {
                        uchar flag = *(ttailp++);
                        if (flag == 0x00) {
                            size_t compressed_size = 0, outSize = 0;
                            read(compressed_size, tail);
                            decompressSimpleBlocking(tail, ttailx, ttaily, ttailz, outSize, compressed_size);
                            tail += compressed_size;
                            ++cnt0;
                        } else {

                        }
                        ttailx += n;
                        ttaily += n;
                        ttailz += n;
                    }
                }

                {
                    const uchar *ttailp = tailp;
                    T *ttailx = tailx, *ttaily = taily, *ttailz = tailz;

                    size_t remaining_length = 0;
                    size_t target_length = 3 * (r - l - cnt0) * n;

                    encoder.load(tail, remaining_length);
                    auto err = encoder.decode(tail, target_length);
                    assert(err.size() % 3 == 0);
                    size_t *errpx = err.data(), *errpy = errpx + err.size() / 3, *errpz = errpy + err.size() / 3;

                    quantizer.load(tail, remaining_length);

                    {
                        uchar flag = *(ttailp++);
                        if (flag == 0x01) {
                            size_t outSize = 0;
                            decompressSimpleBlocking(st[stid], ttailx, ttaily, ttailz, outSize, stsz[stid]);
                            for (size_t i = 0; i < n; i++) {
                                *(ttailx + i) = quantizer.recover(*(ttailx + i), *(errpx++));
                            }
                            for (size_t i = 0; i < n; i++) {
                                *(ttaily + i) = quantizer.recover(*(ttaily + i), *(errpy++));
                            }
                            for (size_t i = 0; i < n; i++) {
                                *(ttailz + i) = quantizer.recover(*(ttailz + i), *(errpz++));
                            }
                        }
                        ttailx += n;
                        ttaily += n;
                        ttailz += n;
                    }

                    for (size_t t = l + 1; t < r; t++) {
                        uchar flag = *(ttailp++);
                        if (flag == 0x01) {
                            for (size_t i = 0; i < n; i++) {
                                *(ttailx + i) = quantizer.recover(*(ttailx - n + i), *(errpx++));
                            }
                            for (size_t i = 0; i < n; i++) {
                                *(ttaily + i) = quantizer.recover(*(ttaily - n + i), *(errpy++));
                            }
                            for (size_t i = 0; i < n; i++) {
                                *(ttailz + i) = quantizer.recover(*(ttailz - n + i), *(errpz++));
                            }
                        } else {

                        }
                        ttailx += n;
                        ttaily += n;
                        ttailz += n;
                    }
                }

                tailp += bt;
                tailx += bt * n;
                taily += bt * n;
                tailz += bt * n;

                delete[] bytes;
            }

            delete[] pos;
        }

        // do not use this function
//        T *decompress(uchar const *cmpData, const size_t &cmpSize, size_t num) {
//            return nullptr;
//            T *dec_data = new T[num];
//            return decompress(cmpData, cmpSize, dec_data);
//        }

//        void decompressSimpleBlocking(uchar const *lossless_data, T *datax, T *datay, T *dataz, size_t &outSize){
//
//
//        }

        void
        decompressWithoutAllocateMemory(const uchar *&lossless_data, T *&datax, T *&datay, T *&dataz, size_t &outSize,
                                        size_t cmpSize) {

            datax = datay = dataz = nullptr;
            outSize = 0;
            decompressSimpleBlocking(lossless_data, datax, datay, dataz, outSize, cmpSize);
        }

        void decompressWithAllocateMemory(const uchar *&lossless_data, T *&datax, T *&datay, T *&dataz, size_t &outSize,
                                          size_t cmpSize) {

            decompressSimpleBlocking(lossless_data, datax, datay, dataz, outSize, cmpSize);
        }


    private:
        Encoder encoder;
        Lossless lossless;
    };

}

#undef __OUTPUT_INFO

#endif

