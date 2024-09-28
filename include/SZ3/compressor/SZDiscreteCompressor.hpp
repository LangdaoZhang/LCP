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
#include <random>
#include <cmath>

#define __OUTPUT_INFO 0
#define __soft_eb 0
#define __batch_info 0

#define __fix 1
#define __huf 0

//#define __blockcnt_encoding_method __huf

//#define test_coding_methods 0

#ifdef test_coding_methods

#if test_coding_methods == 0
#define __blockst_encoding_method __fix
#define __repos_encoding_method __fix
#endif

#if test_coding_methods == 1
#define __blockst_encoding_method __fix
#define __repos_encoding_method __huf
#endif

#if test_coding_methods == 2
#define __blockst_encoding_method __huf
#define __repos_encoding_method __fix
#endif

#if test_coding_methods == 3
#define __blockst_encoding_method __huf
#define __repos_encoding_method __huf
#endif

#endif

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

//        T sq(T x){ return x*x; }
        T sq(T x) { return fabs(x); }

        template<class Type>
        Type max(Type x, Type y, Type z) {
            return std::max(std::max(x, y), z);
        }

//        template<class Type>
//        class StaticArrayManager {
//        public:
//            StaticArrayManager() {
//                mp = std::map<size_t, std::map<std::string, Type *>>();
//            }
//
//            Type *getArray(size_t n, std::string flag) {
//                if (mp.find(n) == mp.end()) {
//                    mp[n] = std::map<std::string, Type *>();
//                    mp[n][flag] = new Type[n];
//
//                } else {
//                    if (mp[n].find(flag) == mp[n].end()) {
//                        mp[n][flag] = new Type[n];
//                    }
//                }
//                return mp[n][flag];
//            }
//
//            void clear() {
//                for (const auto [key_n, mp_n] : mp) {
//                    for (const auto [key_flag, ptr]: mp_n) {
//                        delete[] ptr;
//                    }
//                    mp_n.clear();
//                }
//                mp.clear();
//            }
//
//            void erase(size_t n, std::string flag) {
//                if (mp.find(n) != mp.end()) {
//                    if (mp[n].find(flag) != mp[n].end()) {
//                        delete[] mp[n][flag];
//                        mp[n].erase(flag);
//                        if (mp[n].empty()) {
//                            mp.erase(n);
//                        }
//                    }
//                }
//            }
//
//        private:
//            std::map<size_t, std::map<std::string, Type *>> mp;
//        };
//
//        StaticArrayManager<T> staticArrayManagerT;
//        StaticArrayManager<size_t> staticArrayManageru64;
//        StaticArrayManager<uchar> staticArrayManageru8;

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

//            static std::unordered_set<size_t> selectedIndices;
////            std::random_device rd;
//            std::mt19937 gen(16707);
//            std::uniform_int_distribution<size_t> dis(0, n - 1);
//
//            size_t target_ = ceil(n * .05);
//            T shrink_ratio = std::pow(.05, 1./3.);
//            printf("shrink_ratio = %.6f\n", shrink_ratio);
//
//            for (_ = 0 ; _ < target_; _++) {
//                size_t i = dis(gen);
//                while (selectedIndices.find(i) != selectedIndices.end()) i = dis(gen);
//                selectedIndices.insert(i);
//                samplex[_] = datax[i] * shrink_ratio;
//                sampley[_] = datax[i] * shrink_ratio;
//                samplez[_] = dataz[i] * shrink_ratio;
//            }
//
//            selectedIndices.clear();
//
//            return;

            _ = 0;

            if (n <= (1llu << 20)) {
                memcpy(samplex, datax, n * sizeof(T));
                memcpy(sampley, datay, n * sizeof(T));
                memcpy(samplez, dataz, n * sizeof(T));
                _ = n;
                return;
            }

            size_t b2 = b * b, b3 = b2 * b;
            size_t cnt[b3 << 1];
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
            while (ceil(rx / (2. * (1llu << la) * conf.absErrorBound)) > 2e5) ++la;
            size_t lb = 0, rb = ceil(log2(ry / conf.absErrorBound));
            while (ceil(ry / (2. * (1llu << lb) * conf.absErrorBound)) > 2e5) ++lb;
            size_t lc = 0, rc = ceil(log2(rz / conf.absErrorBound));
            while (ceil(rz / (2. * (1llu << lc) * conf.absErrorBound)) > 2e5) ++lc;

//            if (std::max(std::max(la, lb), lc) <= 8 && 8 <= std::min(std::min(ra, rb), rc)) {
//                return {8, 8, 8};
//            }
//            else{
//                return {0, 0, 0};
//            }


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

        enum EncodingMethod {
            HUFFMAN,
            FIX_LENGTH,
            NOT_SPECIFIED,
            ENCODING_STATE_NUM
        };

        static const std::string encodingMethod2String (EncodingMethod e) {
            switch (e) {
                case HUFFMAN : return "HUFFMAN";
                case FIX_LENGTH : return "FIX_LENGTH";
                case NOT_SPECIFIED : return "NOT_SPECIFIED";
                default : return "ERROR";
            }
            return "ERROR";
        };

        class EncodingMethodsCache {
        public:

            EncodingMethodsCache() {
                init();
            }

            void init() {
                mapEncodingMethodCacheState.clear();
            }

            EncodingMethod *get(double eb, size_t bx, size_t by, size_t bz) {

                size_t h = getStateHash(eb, bx, by, bz);
                if (mapEncodingMethodCacheState.count(h) == 0) {
                    mapEncodingMethodCacheState[h] = EncodingMethodCacheState();
                }
                EncodingMethodCacheState &state = mapEncodingMethodCacheState[h];

                if (state.cnt == 0) {
                    state.update(NOT_SPECIFIED, NOT_SPECIFIED, NOT_SPECIFIED, NOT_SPECIFIED);
                }
                else {
                    --state.cnt;
                }

                return (EncodingMethod*) state.methods;
            }

            void write(double eb, size_t bx, size_t by, size_t bz, int64_t cnt_ = 1ll << 32) {

                size_t h = getStateHash(eb, bx, by, bz);
                if (mapEncodingMethodCacheState.count(h) == 0) {
                    mapEncodingMethodCacheState[h] = EncodingMethodCacheState();
                }
                EncodingMethodCacheState &state = mapEncodingMethodCacheState[h];

                state.cnt = cnt_;
            }

            void write(double eb, size_t bx, size_t by, size_t bz,
                       EncodingMethod blkstMethod_, EncodingMethod blkcntMethod_,
                       EncodingMethod quadsMethod_, EncodingMethod reposMethod_, int64_t cnt_ = 1ll << 32) {

                size_t h = getStateHash(eb, bx, by, bz);
                if (mapEncodingMethodCacheState.count(h) == 0) {
                    mapEncodingMethodCacheState[h] = EncodingMethodCacheState();
                }
                EncodingMethodCacheState &state = mapEncodingMethodCacheState[h];

                state.update(blkstMethod_, blkcntMethod_, quadsMethod_, reposMethod_);
                state.cnt = cnt_;
            }

        private:

            size_t getStateHash(double error_bound, size_t bx, size_t by, size_t bz) {
                size_t heb = std::hash<double>{}(error_bound);
                size_t hb = bx << 32 | by << 16 | bz;
                return heb ^ hb;
            }

            class EncodingMethodCacheState {
            public:

                EncodingMethodCacheState() {
                    cnt = 0;
//                    blkstMethod = NOT_SPECIFIED;
//                    blkcntMethod = NOT_SPECIFIED;
//                    quadsMethod = NOT_SPECIFIED;
//                    reposMethod = NOT_SPECIFIED;
//                    methods[0] = methods[1] = methods[2] = methods[3] = NOT_SPECIFIED;
                    update(NOT_SPECIFIED, NOT_SPECIFIED, NOT_SPECIFIED, NOT_SPECIFIED);
                }

                void update(EncodingMethod blkstMethod, EncodingMethod blkcntMethod, EncodingMethod quadsMethod, EncodingMethod reposMethod) {
                    methods[0] = blkstMethod;
                    methods[1] = blkcntMethod;
                    methods[2] = quadsMethod;
                    methods[3] = reposMethod;
                }

                int64_t cnt;
                EncodingMethod methods[4];
//                EncodingMethod &blkstMethod = methods[0];
//                EncodingMethod &blkcntMethod = methods[1];
//                EncodingMethod &quadsMethod = methods[2];
//                EncodingMethod &reposMethod = methods[3];
            };

            std::unordered_map<size_t, EncodingMethodCacheState> mapEncodingMethodCacheState;

        } encodingMethodsCache;

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

            for (size_t i = 0; i < conf.num; i++) {
                ry = std::max(ry, datay[i]);
                py = std::min(py, datay[i]);
            }
            ry -= py;

            for (size_t i = 0; i < conf.num; i++) {
                rz = std::max(rz, dataz[i]);
                pz = std::min(pz, dataz[i]);
            }
            rz -= pz;

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
            std::vector<size_t> unx, uny, unz;
#endif

            size_t *inord = nullptr;

            if (ord == nullptr) {
                inord = new size_t[conf.num];
                ord = inord;
            }

            {

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

                    if (fabs(decx - datax[i]) > conf.absErrorBound) {
                        unx.push_back(i << 1 | (decx > datax[i] ? 1 : 0));
                    }
                    if (fabs(decy - datay[i]) > conf.absErrorBound) {
                        uny.push_back(i << 1 | (decy > datay[i] ? 1 : 0));
                    }
                    if (fabs(decz - dataz[i]) > conf.absErrorBound) {
                        unz.push_back(i << 1 | (decz > dataz[i] ? 1 : 0));
                    }
#endif

                    size_t cx = x / bx;
                    size_t dx = x % bx;
                    size_t cy = y / by;
                    size_t dy = y % by;
                    size_t cz = z / bz;
                    size_t dz = z % bz;

                    NodeWithOrder tem(cx / 2 + cy / 2 * nx + cz / 2 * nx * ny,
                                      (dx + dy * bx + dz * bx * by) | ((cx & 1) << 60) | ((cy & 1) << 61) |
                                      ((cz & 1) << 62), i);
                    vec[i] = tem;
                }

                radix_sort<NodeWithOrder>(vec, vec + conf.num);

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

#if !__soft_eb
            uchar *bytes_data = new uchar[std::max(conf.num * 16, (size_t) (1 << 16)) +
                                          (unx.size() + uny.size() + unz.size()) * 8], *tail_data = bytes_data;
#endif

#if __soft_eb
            uchar *bytes_data = new uchar[std::max(conf.num * 16, (size_t) (1 << 16))], *tail_data = bytes_data;
#endif

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

            uchar *vtail_data = tail_data;

#if __OUTPUT_INFO
            uchar *ptail_data = tail_data;
#endif

            EncodingMethod *encodingMethods  = encodingMethodsCache.get(conf.absErrorBound, bx, by, bz);
            EncodingMethod *blkstMethod_ptr  = &encodingMethods[0];
            EncodingMethod *blkcntMethod_ptr = &encodingMethods[1];
            EncodingMethod *quadsMethod_ptr  = &encodingMethods[2];
            EncodingMethod *reposMethod_ptr  = &encodingMethods[3];
            uchar encodingMethodsCacheUpdate = 0x00;
            if (encodingMethods[0] == NOT_SPECIFIED ||
                encodingMethods[1] == NOT_SPECIFIED ||
                encodingMethods[2] == NOT_SPECIFIED ||
                encodingMethods[3] == NOT_SPECIFIED) {
                encodingMethodsCacheUpdate = 0x01;
            }

//                encodingMethods[0] = encodingMethods[1] = encodingMethods[2] = encodingMethods[3] = FIX_LENGTH; // NOT_SPECIFIED;

            uchar *bytes_blkst = nullptr, *tail_blkst = nullptr;

            switch (*blkstMethod_ptr) {

                case HUFFMAN: {

                    encoder.preprocess_encode(blkst, blknum, 0, 0x00);
                    encoder.save(tail_data);
                    encoder.encode(blkst, blknum, tail_data);

                    break;
                }
                case FIX_LENGTH: {

                    encoder.preprocess_encode(blkst, blknum, 0, 0x01);
                    encoder.save(tail_data);
                    encoder.encode(blkst, blknum, tail_data);

                    break;
                }
                case NOT_SPECIFIED: {

                    bytes_blkst = new uchar[std::max(blknum * 16, (size_t) (1 << 16))], tail_blkst = bytes_blkst;
                    vtail_data = tail_data;

                    encoder.preprocess_encode(blkst, blknum, 0, 0x00);
                    encoder.save(vtail_data);
                    encoder.encode(blkst, blknum, vtail_data);

                    encoder.preprocess_encode(blkst, blknum, 0, 0x01);
                    encoder.save(tail_blkst);
                    encoder.encode(blkst, blknum, tail_blkst);

                    size_t cmpSize0 = 0, cmpSize1 = 0;
                    delete[] lossless.compress(tail_data, vtail_data - tail_data, cmpSize0);
                    delete[] lossless.compress(bytes_blkst, tail_blkst - bytes_blkst, cmpSize1);

                    if (cmpSize0 < cmpSize1) {
                        tail_data = vtail_data;
                        *blkstMethod_ptr = HUFFMAN;
                    } else {
                        write(bytes_blkst, tail_blkst - bytes_blkst, tail_data);
                        *blkstMethod_ptr = FIX_LENGTH;
                    }

                    delete[] bytes_blkst;

                    break;
                }
                default: {
                    exit(-1);
                }
            };

            delete[] blkst;

#if __OUTPUT_INFO

            printf("size of blkst = %.2lf MB, %zu bytes\n", 1. * (tail_data - ptail_data) / 1024 / 1024, tail_data - ptail_data);
            size_t cmpblkstSize;
            delete[] lossless.compress(ptail_data, tail_data - ptail_data, cmpblkstSize);
            printf("size of compressed blkst = %.2lf MB, %zu bytes\n", 1. * cmpblkstSize / 1024 / 1024, cmpblkstSize);
            ptail_data = tail_data;

#endif

            switch (*blkcntMethod_ptr) {

                case HUFFMAN: {

                    encoder.preprocess_encode(blkcnt, blknum, 0, 0xc0);
                    encoder.save(tail_data);
                    encoder.encode(blkcnt, blknum, tail_data);

                    break;
                }

                case FIX_LENGTH: {

                    encoder.preprocess_encode(blkcnt, blknum, 0, 0xc1);
                    encoder.save(tail_data);
                    encoder.encode(blkcnt, blknum, tail_data);

                    break;
                }

                case NOT_SPECIFIED: {

                    uchar *bytes_blkcnt = new uchar[std::max(blknum * 8, (size_t) (1 << 16))], *tail_blkcnt = bytes_blkcnt;
                    vtail_data = tail_data;

                    encoder.preprocess_encode(blkcnt, blknum, 0, 0xc0);
                    encoder.save(vtail_data);
                    encoder.encode(blkcnt, blknum, vtail_data);

                    encoder.preprocess_encode(blkcnt, blknum, 0, 0xc1);
                    encoder.save(tail_blkcnt);
                    encoder.encode(blkcnt, blknum, tail_blkcnt);

                    size_t cmpSize0 = 0, cmpSize1 = 0;
                    delete[] lossless.compress(tail_data, vtail_data - tail_data, cmpSize0);
                    delete[] lossless.compress(bytes_blkcnt, tail_blkcnt - bytes_blkcnt, cmpSize1);

                    if (cmpSize0 < cmpSize1) {
                        tail_data = vtail_data;
                        *blkcntMethod_ptr = HUFFMAN;
                    } else {
                        write(bytes_blkcnt, tail_blkcnt - bytes_blkcnt, tail_data);
                        *blkcntMethod_ptr = FIX_LENGTH;
                    }

                    delete[] bytes_blkcnt;

                    break;
                }

                default: {
                    exit(-1);
                }

            }

            delete[] blkcnt;


#if __OUTPUT_INFO

            printf("size of blkcnt = %.2lf MB, %zu bytes\n", 1. * (tail_data - ptail_data) / 1024 / 1024, tail_data - ptail_data);
            size_t cmpblkcntSize;
            delete[] lossless.compress(ptail_data, tail_data - ptail_data, cmpblkcntSize);
            printf("size of compressed blkcnt = %.2lf MB, %zu bytes\n", 1. * cmpblkcntSize / 1024 / 1024, cmpblkcntSize);
            ptail_data = tail_data;

#endif

            switch (*quadsMethod_ptr) {

                case HUFFMAN: {

                    encoder.preprocess_encode(quads, conf.num, 8, 0xc0);
                    encoder.save(tail_data);
                    encoder.encode(quads, conf.num, tail_data);

                    break;
                }

                case FIX_LENGTH: {

                    encoder.preprocess_encode(quads, conf.num, 8, 0xc1);
                    encoder.save(tail_data);
                    encoder.encode(quads, conf.num, tail_data);

                    break;
                }

                case NOT_SPECIFIED: {

                    uchar *bytes_quads = new uchar[std::max((size_t) ceil(conf.num * 0.4), (size_t) 1024)],
                            *tail_quads = bytes_quads;
                    vtail_data = tail_data;

                    encoder.preprocess_encode(quads, conf.num, 8, 0xc0);
                    encoder.save(vtail_data);
                    encoder.encode(quads, conf.num, vtail_data);

                    encoder.preprocess_encode(quads, conf.num, 8, 0xc1);
                    encoder.save(tail_quads);
                    encoder.encode(quads, conf.num, tail_quads);

                    size_t cmpSize0 = 0, cmpSize1 = 0;
                    delete[] lossless.compress(tail_data, vtail_data - tail_data, cmpSize0);
                    delete[] lossless.compress(bytes_quads, tail_quads - bytes_quads, cmpSize1);

                    if (cmpSize0 < cmpSize1) {
                        tail_data = vtail_data;
                        *quadsMethod_ptr = HUFFMAN;
                    } else {
                        write(bytes_quads, tail_quads - bytes_quads, tail_data);
                        *quadsMethod_ptr = FIX_LENGTH;
                    }

                    delete[] bytes_quads;

                    break;
                }

                default: {
                    exit(-1);
                }
            }

            delete[] quads;

            switch (*reposMethod_ptr) {

                case HUFFMAN: {

                    encoder.preprocess_encode(repos, conf.num, 0, 0x00);
                    encoder.save(tail_data);
                    encoder.encode(repos, conf.num, tail_data);

                    break;
                }

                case FIX_LENGTH: {

                    encoder.preprocess_encode(repos, conf.num, 0, 0x01);
                    encoder.save(tail_data);
                    encoder.encode(repos, conf.num, tail_data);

                    break;
                }

                case NOT_SPECIFIED: {

                    if (bx * by * bz > (1llu << 48)) {

                        encoder.preprocess_encode(repos, conf.num, 0, 0x01);
                        encoder.save(tail_data);
                        encoder.encode(repos, conf.num, tail_data);

                        *reposMethod_ptr = FIX_LENGTH;

                        break;
                    }

                    uchar *bytes_repos = new uchar[std::max(conf.num *
                                                            std::max((size_t) 4,
                                                                     (size_t) ceil(log2(1. * bx * by * bz) / 4.)),
                                                            (size_t) (1 << 16))], *tail_repos = bytes_repos;
                    vtail_data = tail_data;

                    encoder.preprocess_encode(repos, conf.num, 0, 0x00);
                    encoder.save(vtail_data);
                    encoder.encode(repos, conf.num, vtail_data);

                    encoder.preprocess_encode(repos, conf.num, 0, 0x01);
                    encoder.save(tail_repos);
                    encoder.encode(repos, conf.num, tail_repos);

                    size_t cmpSize0 = 0, cmpSize1 = 0;
                    delete[] lossless.compress(tail_data, vtail_data - tail_data, cmpSize0);
                    delete[] lossless.compress(bytes_repos, tail_repos - bytes_repos, cmpSize1);

                    if (cmpSize0 < cmpSize1) {
                        tail_data = vtail_data;
                        *reposMethod_ptr = HUFFMAN;
                    } else {
                        write(bytes_repos, tail_repos - bytes_repos, tail_data);
                        *reposMethod_ptr = FIX_LENGTH;
                    }

                    delete[] bytes_repos;

                    break;
                }

                default: {
                    exit(-1);
                }
            };

            delete[] repos;

            if (encodingMethodsCacheUpdate) {

                encodingMethodsCache.write(conf.absErrorBound, bx, by, bz);
            }

#if !__soft_eb
            {
                size_t *ordv = new size_t[conf.num];
                for (size_t i = 0; i < conf.num; i++) {
                    ordv[ord[i]] = i;
                }

                {
                    for (size_t i = 0; i < unx.size(); i++) {
                        unx[i] = (ordv[unx[i] >> 1] << 1) | (unx[i] & 1);
                    }
                    std::sort(unx.begin(), unx.end());
                    if (unx.size() > 1)
                        for (size_t i = unx.size() - 1; i > 0; i--) {
                            unx[i] -= unx[i - 1];
                        }

                    write(unx.size(), tail_data);
                    if (unx.size() > 0) {
                        encoder.preprocess_encode(unx.data(), unx.size(), 0);
                        encoder.save(tail_data);
                        encoder.encode(unx.data(), unx.size(), tail_data);
                    }
                }

                {
                    for (size_t i = 0; i < uny.size(); i++) {
                        uny[i] = (ordv[uny[i] >> 1] << 1) | (uny[i] & 1);
                    }
                    std::sort(uny.begin(), uny.end());
                    if (uny.size() > 1)
                        for (size_t i = uny.size() - 1; i > 0; i--) {
                            uny[i] -= uny[i - 1];
                        }

                    write(uny.size(), tail_data);
                    if (uny.size() > 0) {
                        encoder.preprocess_encode(uny.data(), uny.size(), 0);
                        encoder.save(tail_data);
                        encoder.encode(uny.data(), uny.size(), tail_data);
                    }
                }

                {
                    for (size_t i = 0; i < unz.size(); i++) {
                        unz[i] = (ordv[unz[i] >> 1] << 1) | (unz[i] & 1);
                    }
                    std::sort(unz.begin(), unz.end());
                    if (unz.size() > 1)
                        for (size_t i = unz.size() - 1; i > 0; i--) {
                            unz[i] -= unz[i - 1];
                        }

                    write(unz.size(), tail_data);
                    if (unz.size() > 0) {
                        encoder.preprocess_encode(unz.data(), unz.size(), 0);
                        encoder.save(tail_data);
                        encoder.encode(unz.data(), unz.size(), tail_data);
                    }
                }

                delete[] ordv;
            }

//            size_t num_unquants = unx.size() + uny.size() + unz.size();
//            printf("%zu %.6lf%%\n", num_unquants, 100. / 3. * num_unquants / conf.num);
#endif

            if (inord != nullptr) {
                delete[] inord;
            }

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
            }

            memcpy(data, a, n * sizeof(T));
        }

        class IsTPCache {
        public:
            IsTPCache(){
                init();
            }

            void init() {
                isTP = 0;
                cnts = 0;
                limit = 1;
                sizeTP = numTP = 0;
            }

            uchar read() {
//                return 0x00;
                /*
                 * 0x00 test both LCP-S and TP
                 * 0x01 do LCP-S
                 * 0x02 do TP
                 */
                if (isTP == 1) {
                    return 0x00;
                }
                if (cnts == 0) {
                    return 0x00;
                }
                --cnts;
                return 0x01;
            }

            void writeS() {
                isTP = 0;
                // maximum = 32
                cnts = limit >> 1;
                if (limit < (1 << 6))  limit <<= 1;
            }

            void writeT() {
                isTP = 0;
                cnts = 0;
                limit = 1;
            }

            uchar writeTSizeFirst(size_t _) {
                printf("%zu %.2f %zu %zu\n", _, 1.2 * sizeTP / numTP, sizeTP, numTP);
                if (_ > 2 * sizeTP / numTP) {
                    return 0x01;
                }
                return 0x00;
            }

            void writeTSizeSecond(size_t _, uchar success) {
                // if T is indeed worse
                if (success == 0x01) {
                    isTP = 0;
                    cnts = 0;
                    limit = 1;
                } else {
                    sizeTP += _;
                    ++numTP;
                }
            }

//        private:
        public:
            size_t isTP;
            size_t cnts;
            size_t limit;
            size_t sizeTP, numTP;

        };

        IsTPCache isTpCacheFirst, isTpCacheBatch;

//        class TPSizeCache {
//        public:
//
//            TPSizeCache() {
//                n = 0;
//                prex = nullptr;
//                prey = nullptr;
//                prez = nullptr;
//                cmpSize = 0;
//            }
//
//            ~TPSizeCache() {
//                delete[] prex;
//                delete[] prey;
//                delete[] prez;
//            }
//
//            size_t n;
//            ska::unordered_set<size_t> piv;
//            size_t *prex, *prey, *prez;
//            size_t cmpSize;
//
//            void init(size_t n_) {
//                n = n_;
//                static std::mt19937 gen(16707);
//                std::uniform_int_distribution<size_t> dis(1, n);
//                while(piv.size() * 100 < n) {
//                    piv.insert(dis(gen));
//                }
//                prex = new size_t[n];
//                prey = new size_t[n];
//                prez = new size_t[n];
//                cmpSize = 0;
//                for (size_t it : piv) {
//                    printf("%zu ", it);
//                }
//                printf("\n");
//            }
//
//            void clear(size_t n_ = 0) {
//                if (n_ > 0) n = n_;
//                cmpSize = 0;
//            }
//
//            /*
//             * 0x00: use the last size
//             * 0x01: need to test
//             */
//
//            uchar test(size_t *pre, size_t *err) {
//                uchar fail = 0;
////                size_t max_diff = 0;
//                for (size_t it : piv) {
//                    if (pre[it] == 0) continue;
//                    if (err[it] == 0) {
//                        fail = 1;
//                        break;
//                    }
//                    size_t diff = err[it] > pre[it] ? err[it] - pre[it] : pre[it] - err[it];
////                    max_diff = std::max(max_diff, diff);
//                    if (diff >= 2) {
//                        fail = 1;
//                        break;
//                    }
//                }
////                printf("max_diff = %zu\n", max_diff);
//                return fail;
//            }
//
//            uchar test(size_t *errx, size_t *erry, size_t *errz) {
//                if (cmpSize == 0) return 0x01;
//                return test(prex, errx) | test(prey, erry) | test(prez, errz);
//            }
//
//            void write(size_t *errx, size_t *erry, size_t *errz, size_t cmpSize_) {
//                memcpy(prex, errx, n * sizeof(size_t));
//                memcpy(prey, erry, n * sizeof(size_t));
//                memcpy(prez, errz, n * sizeof(size_t));
//
//                cmpSize = cmpSize_;
//            }
//
//            size_t getSize() {
//                return cmpSize;
//            }
//
//        };

//        TPSizeCache tpSizeCache[2];

        void getTPArray1D(T *data, T *predData, size_t *err, LinearQuantizer<T> &quantizer, size_t n) {
            for (size_t i = 0; i < n; i++) {
                err[i] = quantizer.quantize(data[i], predData[i]);
                if (err[i] == 0) {
                    quantizer.insertUnpred(data[i]);
                }
            }
        }

        void overwriteTPData1D(T *data, T *predData, LinearQuantizer<T> &quantizer, size_t n) {
            for (size_t i = 0; i < n; i++) {
                quantizer.overwrite(data[i], predData[i]);
            }
        }

        void getTPArray1DOverwrite(T *data, T *predData, size_t *err, LinearQuantizer<T> &quantizer, size_t n) {
            for (size_t i = 0; i < n; i++) {
                err[i] = quantizer.quantize_and_overwrite(data[i], predData[i]);
            }
        }

        /*
         * return the predicted compressed size of the error array
         */

        size_t getTPArrayAndEstimateSize(T *datax, T *datay, T *dataz, T *predDatax, T *predDatay, T *predDataz,
                                         size_t *errx, size_t *erry, size_t *errz, LinearQuantizer<T> &quantizer,
                                         size_t n) {
            getTPArray1D(datax, predDatax, errx, quantizer, n);
            getTPArray1D(datay, predDatay, erry, quantizer, n);
            getTPArray1D(dataz, predDataz, errz, quantizer, n);

            uchar *bytes = new uchar[std::max(n * 12, (size_t) 1 << 16)], *tail = bytes;
//            size_t *err = new size_t[n * 3];
//            memcpy(err, errx, n * sizeof(size_t));
//            memcpy(err + n, erry, n * sizeof(size_t));
//            memcpy(err + n + n, errz, n * sizeof(size_t));

            static HuffmanEncoder<size_t> encoder;

            encoder.preprocess_encode(errx, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(errx, n, tail);
            encoder.preprocess_encode(erry, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(erry, n, tail);
            encoder.preprocess_encode(errz, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(errz, n, tail);

            quantizer.save(tail);

            size_t cmpSize;
            delete[] lossless.compress(bytes, tail - bytes, cmpSize);

            return cmpSize;
        }

        double est_time_encode;

        size_t getTPArrayAndEstimateSizeOverwrite(T *datax, T *datay, T *dataz, T *predDatax, T *predDatay, T *predDataz,
                                           size_t *errx, size_t *erry, size_t *errz, LinearQuantizer<T> &quantizer,
                                           size_t n) {
            getTPArray1DOverwrite(datax, predDatax, errx, quantizer, n);
            getTPArray1DOverwrite(datay, predDatay, erry, quantizer, n);
            getTPArray1DOverwrite(dataz, predDataz, errz, quantizer, n);

            uchar *bytes = new uchar[std::max(n * 12, (size_t) 1 << 16)], *tail = bytes;
//            size_t *err = new size_t[n * 3];
//            memcpy(err, errx, n * sizeof(size_t));
//            memcpy(err + n, erry, n * sizeof(size_t));
//            memcpy(err + n + n, errz, n * sizeof(size_t));

            Timer timer;
            timer.start();

            static HuffmanEncoder<size_t> encoder;

            encoder.preprocess_encode(errx, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(errx, n, tail);
            encoder.preprocess_encode(erry, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(erry, n, tail);
            encoder.preprocess_encode(errz, n, quantizer.get_radius() * 2);
            encoder.save(tail);
            encoder.encode(errz, n, tail);

            quantizer.save(tail);

            size_t cmpSize;
            delete[] lossless.compress(bytes, tail - bytes, cmpSize);

            est_time_encode = timer.stop();

            return cmpSize;
        }

        size_t getTPEstimateSize(T *datax, T *datay, T *dataz, T *predDatax, T *predDatay, T *predDataz,
                                 LinearQuantizer<T> &quantizer, size_t n) {
            size_t *err = new size_t[3 * n], *errx = err, *erry = errx + n, *errz = erry + n;
            uchar *bytes = new uchar[12 * n + (1 << 10)], *tail = bytes;

            std::vector<T> unpred;
            for (size_t i = 0; i < n; i++) {
                errx[i] = quantizer.quantize(datax[i], predDatax[i]);
                if (errx[i] == 0) unpred.push_back(datax[i]);
            }
            for (size_t i = 0; i < n; i++) {
                erry[i] = quantizer.quantize(datay[i], predDatay[i]);
                if (erry[i] == 0) unpred.push_back(datay[i]);
            }
            for (size_t i = 0; i < n; i++) {
                errz[i] = quantizer.quantize(dataz[i], predDataz[i]);
                if (errz[i] == 0) unpred.push_back(dataz[i]);
            }
            static HuffmanEncoder<size_t> encoder;
            encoder.preprocess_encode(err, 3 * n, 0);
            encoder.save(tail);
            encoder.encode(err, 3 * n, tail);

            delete[] err;

            write(unpred, tail);

            size_t cmpSize;
            delete[] lossless.compress(bytes, tail - bytes, cmpSize);
            delete[] bytes;

            return cmpSize;
        }

        void getTPArrayAndOverwrite(T *datax, T *datay, T *dataz, T *predDatax, T *predDatay, T *predDataz,
                                      size_t *errx, size_t *erry, size_t *errz, LinearQuantizer<T> &quantizer, size_t n) {
            getTPArray1DOverwrite(datax, predDatax, errx, quantizer, n);
            getTPArray1DOverwrite(datay, predDatay, erry, quantizer, n);
            getTPArray1DOverwrite(dataz, predDataz, errz, quantizer, n);
        }

        void overwriteTPData(T *datax, T *datay, T *dataz, T *predDatax, T *predDatay, T *predDataz,
                             LinearQuantizer<T> quantizer, size_t n) {
            overwriteTPData1D(datax, predDatax, quantizer, n);
            overwriteTPData1D(datay, predDatay, quantizer, n);
            overwriteTPData1D(dataz, predDataz, quantizer, n);
        }

        /*
        * To compress the data from datax, datay and dataz using configure conf and buffer size bt
        * Store the result in the return pointer, and store the size compressed data in compressed_data
        *
        * after decompress, datax, datay, dataz will not increase
        */

        Timer timer;
        bool est_lock = true;
        double est_time_s, est_time_t;
        double est_time = 0;

        uchar *compressTemporalPredictionOnSlice(const Config &conf, T *datax, T *datay, T *dataz,
                                                 size_t &compressed_size, size_t *ord, uchar blkflag,
                                                 size_t bx, size_t by, size_t bz, uchar *&bytes1,
                                                 size_t &compressed_size1, T *decmpdata1, size_t *ord1,
                                                 T fflag) {



            static const int64_t radius = (1 << 15);

            LinearQuantizer<T> quantizer(conf.absErrorBound, radius);
            LinearQuantizer<T> temQuantizer(conf.absErrorBound, radius);

            size_t nt = conf.dims[0];
            size_t n = conf.dims[1];

            uchar *bytes = new uchar[std::max(conf.num * 16, (size_t) 65536)], *tailp = bytes, *tail = bytes + nt;

            Config conf1(n);
            conf1.absErrorBound = conf.absErrorBound / fflag;
            T fflag_stride = std::min((fflag - 1) / (nt / (T) 4), fflag - 1);
//            printf("fflag stride = %.2f\n", fflag_stride);

            size_t *p = new size_t[n];
//            size_t *p = new size_t[n], *temp = new size_t[n];

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

            T *pda = new T[3 * nt * n];
            T *pdx = pda, *pdy = pdx + nt * n, *pdz = pdy + nt * n;
            T *nxa = new T[3 * n];
            T *nxx = nxa, *nxy = nxx + n, *nxz = nxy + n;
            size_t *erra = new size_t[3 * nt * n];
            size_t *errx = erra, *erry = errx + nt * n, *errz = erry + nt * n;
            size_t errlen = 0;

//            T *decmpdata1 = new T[n * 3];
            T *d1x = decmpdata1, *d1y = d1x + n, *d1z = d1y + n;
            size_t outSize;

            // to check if to create a new time frame or not

            if (bytes1 == nullptr) {

#if __batch_info
                printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", (size_t) 0);
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
//                decompressSimpleBlocking(bytes1c, d1x, d1y, d1z, outSize, compressed_size1);

//                memcpy(pdx, datax, n * sizeof(T));
//                memcpy(pdy, datay, n * sizeof(T));
//                memcpy(pdz, dataz, n * sizeof(T));

                for (size_t i = 0; i < n; i++) pdx[i] = datax[ord1[i]];
                for (size_t i = 0; i < n; i++) pdy[i] = datay[ord1[i]];
                for (size_t i = 0; i < n; i++) pdz[i] = dataz[ord1[i]];

//                size_t cmpSizeT = getTPEstimateSize(pdx, pdy, pdz, d1x, d1y, d1z, quantizer, n);
//                if (compressed_size1 < cmpSizeT) {
//                    if (fflag > 1) {
//                        fflag -= fflag_stride;
//                        if (fflag < 1) fflag = 1;
//                        conf1.absErrorBound = conf.absErrorBound / fflag;
//                    }
//#if __batch_info
//                    printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", (size_t) 0);
//#endif
//                    writeBytesByte(tailp, 0x00);
//
//                    bytes1 = compressSimpleBlocking(conf1, datax, datay, dataz, compressed_size1, p, blkflag, bx, by,
//                                                    bz);
//                    if (ord != nullptr) {
//                        memcpy(ord, p, n * sizeof(size_t));
//                    }
//                    memcpy(ord1, p, n * sizeof(size_t));
//                    const uchar *bytes1c = bytes1;
//                    decompressSimpleBlocking(bytes1c, d1x, d1y, d1z, outSize, compressed_size1);
//                    memcpy(pdx, d1x, n * sizeof(T));
//                    memcpy(pdy, d1y, n * sizeof(T));
//                    memcpy(pdz, d1z, n * sizeof(T));
//                } else {
//#if __batch_info
//                    printf("\e[32m\e[1mold batch, t = %zu\n\e[0m", (size_t) 0);
//#endif
//                    writeBytesByte(tailp, 0x01);
//                    getTPArrayAndOverwrite(pdx, pdy, pdz, d1x, d1y, d1z, errx, erry, errz, quantizer, n);
//                    errlen = n;
//                    if (ord != nullptr) {
//                        memcpy(ord, ord1, n * sizeof(size_t));
//                    }
//                    memcpy(p, ord1, n * sizeof(size_t));
//                }

                switch (isTpCacheFirst.read()) {
                    case 0x00 : {
                        size_t cmpSizeS = 0, cmpSizeT = 0;
                        timer.start();
                        uchar *bytesv1 = nullptr;
                        est_time_s = timer.stop();
                        timer.start();
                        memcpy(nxx, pdx, n * sizeof(T));
                        memcpy(nxy, pdy, n * sizeof(T));
                        memcpy(nxz, pdz, n * sizeof(T));
                        temQuantizer.clear();
                        cmpSizeT = getTPArrayAndEstimateSizeOverwrite(nxx, nxy, nxz, d1x, d1y, d1z, errx, erry, errz, temQuantizer, n);
//                        cmpSizeT = getTPEstimateSize(pdx, pdy, pdz, d1x, d1y, d1z, temQuantizer, n);
                        est_time_t = timer.stop();

                        if (compressed_size1 < cmpSizeT) {
                            bytesv1 = compressSimpleBlocking(conf1, datax, datay, dataz, cmpSizeS, p, blkflag, bx,
                                                   by, bz);
                            compressed_size1 = cmpSizeS;
                            if(!est_lock) est_time += est_time_t;
#if __batch_info
                            printf("[01]\e[34m\e[1mnew batch, t = %zu\n\e[0m", (size_t) 0);
#endif
                            isTpCacheFirst.writeS();
                            writeBytesByte(tailp, 0x00);
                            bytes1 = bytesv1;
                            if (fflag > 1) {
                                fflag -= fflag_stride;
                                if (fflag < 1) fflag = 1;
                                conf1.absErrorBound = conf.absErrorBound / fflag;
                            }
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
                            if(!est_lock) est_time += est_time_s;
                            isTpCacheFirst.writeTSizeSecond(cmpSizeT, 0x00);
#if __batch_info
                            printf("[02]\e[32m\e[1mold batch, t = %zu\n\e[0m", (size_t) 0);
#endif
                            isTpCacheFirst.writeT();
                            writeBytesByte(tailp, 0x01);

                            timer.start();

                            memcpy(pdx, nxx, n * sizeof(T));
                            memcpy(pdy, nxy, n * sizeof(T));
                            memcpy(pdz, nxz, n * sizeof(T));

                            quantizer.mergeUnpreds(temQuantizer);
                            if(!est_lock) est_time += timer.stop();
                            if(!est_lock) est_time += est_time_encode;
//                            getTPArrayAndOverwrite(pdx, pdy, pdz, d1x, d1y, d1z, errx, erry, errz, quantizer, n);
                            errlen = n;

                            if (ord != nullptr) {
                                memcpy(ord, ord1, n * sizeof(size_t));
                            }
                            memcpy(p, ord1, n * sizeof(size_t));
                        }
                        break;
                    }
                    case 0x01 : {
                        if (fflag > 1) {
                            fflag -= fflag_stride;
                            if (fflag < 1) fflag = 1;
                            conf1.absErrorBound = conf.absErrorBound / fflag;
                        }
#if __batch_info
                        printf("[03]\e[34m\e[1mnew batch, t = %zu\n\e[0m", (size_t) 0);
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
                        break;
                    }
//                    case 0x02 : {
//                        size_t cmpSizeT = 0;
//                        LinearQuantizer<T> temQuantizer(conf.absErrorBound, radius);
//                        cmpSizeT = getTPArrayAndEstimateSize(pdx, pdy, pdz, d1x, d1y, d1z, errx, erry, errz, temQuantizer, n);
//                        if (isTpCacheFirst.writeTSizeFirst(cmpSizeT) == 0x00) {
//                            isTpCacheFirst.writeTSizeSecond(cmpSizeT, 0x00);
//#if __batch_info
//                            printf("[04]\e[32m\e[1mold batch, t = %zu\n\e[0m", (size_t) 0);
//#endif
//                            writeBytesByte(tailp, 0x01);
//                            quantizer.mergeUnpreds(temQuantizer);
//                            errlen = n;
//                            if (ord != nullptr) {
//                                memcpy(ord, ord1, n * sizeof(size_t));
//                            }
//                            memcpy(p, ord1, n * sizeof(size_t));
//                        } else {
//                            size_t cmpSizeS = 0;
//                            uchar *bytesv1 = compressSimpleBlocking(conf1, datax, datay, dataz, cmpSizeS, temp, blkflag, bx, by, bz);
//                            if (cmpSizeS < cmpSizeT) {
//                                isTpCacheFirst.writeTSizeSecond(cmpSizeT, 0x01);
//#if __batch_info
//                                printf("[05]\e[34m\e[1mnew batch, t = %zu\n\e[0m", (size_t) 0);
//#endif
//                                if (fflag > 1) {
//                                    fflag -= fflag_stride;
//                                    if (fflag < 1) fflag = 1;
//                                    conf1.absErrorBound = conf.absErrorBound / fflag;
//                                }
//                                writeBytesByte(tailp, 0x00);
//                                bytes1 = bytesv1;
//                                memcpy(p, temp, n * sizeof(size_t));
//                                if (ord != nullptr) {
//                                    memcpy(ord, p, n * sizeof(size_t));
//                                }
//                                memcpy(ord1, p, n * sizeof(size_t));
//                                const uchar *bytes1c = bytes1;
//                                compressed_size1 = cmpSizeS;
//                                decompressSimpleBlocking(bytes1c, pdx, pdy, pdz, outSize, compressed_size1);
//
//                            } else {
//                                isTpCacheFirst.writeTSizeSecond(cmpSizeT, 0x00);
//#if __batch_info
//                                printf("[06]\e[32m\e[1mold batch, t = %zu\n\e[0m", (size_t) 0);
//#endif
//                                writeBytesByte(tailp, 0x01);
//                                delete[] bytesv1;
//                                quantizer.mergeUnpreds(temQuantizer);
//                                errlen = n;
//                                if (ord != nullptr) {
//                                    memcpy(ord, ord1, n * sizeof(size_t));
//                                }
//                                memcpy(p, ord1, n * sizeof(size_t));
//                            }
//                        }
//
//                        break;
//                    }
                    default: {
                        perror("Abnormal return value of isTPCache.\n");
                        exit(-1);
                    }
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

                size_t cmpSizeT = getTPEstimateSize(nowpx, nowpy, nowpz, prepx, prepy, prepz, quantizer, n);
//                if (compressed_size < cmpSizeT) {
////                if (true) {
//
//                    if (fflag > 1) {
//                        fflag -= fflag_stride;
//                        if (fflag < 1) fflag = 1;
//                        if (nt - t <= 2) fflag = 1;
//                        conf1.absErrorBound = conf.absErrorBound / fflag;
//                    }
//#if __batch_info
//                    printf("\e[34m\e[1mnew batch, t = %zu\n\e[0m", t);
//#endif
//                    writeBytesByte(tailp, 0x00);
//
//                    const uchar *bytes1 = compressSimpleBlocking(conf1, datax + t * n, datay + t * n, dataz + t * n,
//                                                                 compressed_size, p, blkflag, bx, by, bz);
//                    write(compressed_size, tail);
//                    write(bytes1, compressed_size, tail);
//
//                    if (t + 1 < nt) {
//                        decompressSimpleBlocking(bytes1, nowpx, nowpy, nowpz, outSize, compressed_size);
//                    }
//
//                    delete[] bytes1;
//
//                } else {
//#if __batch_info
//                    printf("\e[32m\e[1mold batch, t = %zu\n\e[0m", t);
//#endif
//                    writeBytesByte(tailp, 0x01);
//                    getTPArrayAndOverwrite(nowpx, nowpy, nowpz, prepx, prepy, prepz, errx + errlen, erry + errlen,
//                                           errz + errlen, quantizer, n);
////                    size_t *errpx = errx + errlen;
////                    size_t *errpy = erry + errlen;
////                    size_t *errpz = errz + errlen;
////
////                    for (size_t i = 0; i < n; i++) {
////                        T &nowx = nowpx[i];
////                        T &prex = prepx[i];
////                        errpx[i] = quantizer.quantize_and_overwrite(nowx, prex);
////                    }
////                    for (size_t i = 0; i < n; i++) {
////                        T &nowy = nowpy[i];
////                        T &prey = prepy[i];
////                        errpy[i] = quantizer.quantize_and_overwrite(nowy, prey);
////                    }
////                    for (size_t i = 0; i < n; i++) {
////                        T &nowz = nowpz[i];
////                        T &prez = prepz[i];
////                        errpz[i] = quantizer.quantize_and_overwrite(nowz, prez);
////                    }
//
//
//                    errlen += n;
//                }


                switch (isTpCacheBatch.read()) {
                    case 0x00 : {
                        size_t cmpSizeS = 0, cmpSizeT = 0;
                        timer.start();
                        uchar *bytesv1 = nullptr;
                        est_time_s = timer.stop();
                        timer.start();
                        memcpy(nxx, nowpx, n * sizeof(T));
                        memcpy(nxy, nowpy, n * sizeof(T));
                        memcpy(nxz, nowpz, n * sizeof(T));
                        temQuantizer.clear();
                        cmpSizeT = getTPArrayAndEstimateSizeOverwrite(nxx, nxy, nxz, prepx, prepy, prepz,
                                                                      errx + errlen, erry + errlen, errz + errlen,
                                                                      temQuantizer, n);
//                        cmpSizeT = getTPEstimateSize(nowpx, nowpy, nowpz, prepx, prepy, prepz, temQuantizer, n);
                        est_time_t = timer.stop();
                        if (compressed_size < cmpSizeT) {
                            bytesv1 = compressSimpleBlocking(conf1, datax + t * n, datay + t * n, dataz + t * n,
                                                     cmpSizeS, p, blkflag, bx, by, bz);
                            compressed_size = cmpSizeS;
                            if(!est_lock) est_time += est_time_t;
#if __batch_info
                            printf("[11]\e[34m\e[1mnew batch, t = %zu\n\e[0m", t);
#endif
                            isTpCacheBatch.writeS();
                            if (fflag > 1) {
                                fflag -= fflag_stride;
                                if (fflag < 1) fflag = 1;
                                if (nt - t <= 2) fflag = 1;
                                conf1.absErrorBound = conf.absErrorBound / fflag;
                            }
                            writeBytesByte(tailp, 0x00);
                            write(cmpSizeS, tail);
                            write(bytesv1, cmpSizeS, tail);
                            const uchar *bytesv1c = bytesv1;
                            decompressSimpleBlocking(bytesv1c, nowpx, nowpy, nowpz, outSize, cmpSizeS);
                        } else {
                            if(!est_lock) est_time += est_time_s;
                            isTpCacheBatch.writeTSizeSecond(cmpSizeT, 0x00);
#if __batch_info
                            printf("[12]\e[32m\e[1mold batch, t = %zu\n\e[0m", t);
#endif
                            isTpCacheBatch.writeT();
                            writeBytesByte(tailp, 0x01);

                            timer.start();

                            memcpy(nowpx, nxx, n * sizeof(T));
                            memcpy(nowpy, nxy, n * sizeof(T));
                            memcpy(nowpz, nxz, n * sizeof(T));

                            quantizer.mergeUnpreds(temQuantizer);
                            if(!est_lock) est_time += timer.stop();
                            if(!est_lock) est_time += est_time_encode;
//                            getTPArrayAndOverwrite(nowpx, nowpy, nowpz, prepx, prepy, prepz, errx + errlen, erry + errlen,
//                                           errz + errlen, quantizer, n);
                            errlen += n;
                        }
                        delete[] bytesv1;
                        break;
                    }
                    case 0x01 : {
                        size_t cmpSizeS = 0;
                        uchar *bytesv1 = compressSimpleBlocking(conf1, datax + t * n, datay + t * n, dataz + t * n,
                                                                cmpSizeS, p, blkflag, bx, by, bz);
                        compressed_size = cmpSizeS;
#if __batch_info
                        printf("[13]\e[34m\e[1mnew batch, t = %zu\n\e[0m", t);
#endif
                        if (fflag > 1) {
                            fflag -= fflag_stride;
                            if (fflag < 1) fflag = 1;
                            if (nt - t <= 2) fflag = 1;
                            conf1.absErrorBound = conf.absErrorBound / fflag;
                        }
                        writeBytesByte(tailp, 0x00);
                        write(cmpSizeS, tail);
                        write(bytesv1, cmpSizeS, tail);
                        const uchar *bytesv1c = bytesv1;
                        decompressSimpleBlocking(bytesv1c, nowpx, nowpy, nowpz, outSize, cmpSizeS);
                        delete[] bytesv1;
                        break;
                    }
//                    case 0x02 : {
//                        size_t cmpSizeT = 0;
//                        LinearQuantizer<T> temQuantizer(conf.absErrorBound, radius);
//                        cmpSizeT = getTPArrayAndEstimateSize(nowpx, nowpy, nowpz, prepx, prepy, prepz, errx + errlen,
//                                                             erry + errlen, errz + errlen, temQuantizer, n);
//                        if (isTpCacheBatch.writeTSizeFirst(cmpSizeT) == 0x00) {
//                            isTpCacheBatch.writeTSizeSecond(cmpSizeT, 0x00);
//#if __batch_info
//                            printf("[14]\e[32m\e[1mold batch, t = %zu\n\e[0m", t);
//#endif
//                            writeBytesByte(tailp, 0x01);
//                            quantizer.mergeUnpreds(temQuantizer);
//                            errlen += n;
//                        } else {
//                            size_t cmpSizeS = 0;
//                            uchar *bytesv1 = compressSimpleBlocking(conf1, datax + t * n, datay + t * n, dataz + t * n,
//                                                                    cmpSizeS, temp, blkflag, bx, by, bz);
//                            if (cmpSizeS < cmpSizeT) {
//                                isTpCacheBatch.writeTSizeSecond(cmpSizeT, 0x01);
//#if __batch_info
//                                printf("[15]\e[34m\e[1mnew batch, t = %zu\n\e[0m", t);
//#endif
//                                if (fflag > 1) {
//                                    fflag -= fflag_stride;
//                                    if (fflag < 1) fflag = 1;
//                                    if (nt - t <= 2) fflag = 1;
//                                    conf1.absErrorBound = conf.absErrorBound / fflag;
//                                }
//                                writeBytesByte(tailp, 0x00);
//                                write(cmpSizeS, tail);
//                                write(bytesv1, cmpSizeS, tail);
//                                memcpy(p, temp, n * sizeof(size_t));
//                                const uchar *bytesv1c = bytesv1;
//                                decompressSimpleBlocking(bytesv1c, d1x, d1y, d1z, outSize, cmpSizeS);
//                                if (t + 1 < nt) {
//                                    memcpy(pdx + t * n, d1x, n * sizeof(T));
//                                    memcpy(pdy + t * n, d1y, n * sizeof(T));
//                                    memcpy(pdz + t * n, d1z, n * sizeof(T));
//                                }
//                            } else {
//                                isTpCacheBatch.writeTSizeSecond(cmpSizeT, 0x00);
//#if __batch_info
//                                printf("[16]\e[32m\e[1mold batch, t = %zu\n\e[0m", t);
//#endif
//                                writeBytesByte(tailp, 0x01);
//                                quantizer.mergeUnpreds(temQuantizer);
//                                errlen += n;
//                            }
//                            delete[] bytesv1;
//                        }
//                        break;
//                    }
                    default: {
                        perror("Abnormal return value of isTPCache.\n");
                        exit(-1);
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

//            delete[] decmpdata1;
            delete[] p;
            delete[] pda;
            delete[] erra;
//            delete[] nxa;

            uchar *lossless_data = lossless.compress(bytes, tail - bytes, compressed_size);
            delete[] bytes;

            return lossless_data;
        }

        uchar isSpatialWorse(const Config conf, T *datax, T *datay, T *dataz) {

            const size_t &n = conf.dims[1];
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
                                                            size_t bz = 0, double fflagInit = 0) {

            assert(conf.N == 2);

            size_t nt = conf.dims[0];
            size_t n = conf.dims[1];

            Timer timer;
            timer.start();

            T fflag = 1;

            if (fflagInit >= 1) {
                fflag = fflagInit;
            } else if (nt >= 64) {
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
                    size_t cnt = 0;
                    size_t t = (nt - 1) / 2;
                    cnt += isSpatialWorse(conf, datax + t * n, datay + t * n, dataz + t * n);
                    blockSizeCache.init();
                    t = (nt - 1) / 4;
                    cnt += isSpatialWorse(conf, datax + t * n, datay + t * n, dataz + t * n);
                    blockSizeCache.init();
                    t = (nt - 1) / 4 * 3;
                    cnt += isSpatialWorse(conf, datax + t * n, datay + t * n, dataz + t * n);
                    blockSizeCache.init();
                    if (cnt > 1) {
                        fflag = 6.4;
                    }
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
            T *decmpdata1 = new T[3 * n];
            size_t compressed_size1 = 0;
            size_t *ord1 = new size_t[n];
            int cnt1 = -1;

            if (fflagInit > 1) {

            } else if (fflag > 1) {
                double l = 1, r = 24, midl, midr;
                size_t size_midl, size_midr;
                Config confSliceTest = Config(std::min(std::min(bt * 4, n > (1 << 20) ? (size_t) 16 : (size_t) 64), nt),
                                              n);
                confSliceTest.absErrorBound = conf.absErrorBound;
                while (r - l > 1) {
                    midl = (l + l + r) / 3.;
                    midr = (l + r + r) / 3.;

                    compressTemporalPredictionOnSlice(confSliceTest, datax, datay, dataz, size_midl, nullptr,
                                                      blkflag, bx, by, bz, bytes1, compressed_size1, decmpdata1, ord1, midl);
                    delete[] bytes1;
                    bytes1 = nullptr;
                    compressed_size1 = 0;
                    size_midl += compressed_size1;
                    blockSizeCache.init();
                    isTpCacheBatch.init();

                    compressTemporalPredictionOnSlice(confSliceTest, datax, datay, dataz, size_midr, nullptr,
                                                      blkflag, bx, by, bz, bytes1, compressed_size1, decmpdata1, ord1, midr);
                    delete[] bytes1;
                    bytes1 = nullptr;
                    compressed_size1 = 0;
                    size_midr += compressed_size1;
                    blockSizeCache.init();
                    isTpCacheBatch.init();
//                    printf("l:[%.2f, %zu], r:[%.2f, %zu]\n", midl, size_midl, midr, size_midr);

                    if (size_midl < size_midr) {
                        r = midr;
                    } else {
                        l = midl;
                    }
                }
                fflag = (l + r) / 2;
            }

//            printf("fflag = %.6lf\n", fflag);

//            printf("fflag time = %.6lfs\n", timer.stop());
            est_lock = false;

            for (size_t l = 0; l < nt; l += bt) {

                size_t r = l + bt;
                if (r > nt) r = nt, confSlice = Config(r - l, n), confSlice.absErrorBound = conf.absErrorBound;

//                printf("%zu %zu\n", l, r);

                size_t slice_compressed_size = 0;

                uchar *bytes1p = bytes1;

                if (nt - l < 64) fflag = 0x01;

                uchar *sliceBytes = compressTemporalPredictionOnSlice(confSlice, datax + l * n,
                                                                      datay + l * n, dataz + l * n,
                                                                      slice_compressed_size,
                                                                      ord == nullptr ? nullptr : ord +
                                                                                                 l *
                                                                                                 n,
                                                                      blkflag, bx, by, bz, bytes1,
                                                                      compressed_size1, decmpdata1, ord1, fflag);

//                uchar *sliceBytes = compressTemporalPredictionOnSliceIntegers(confSlice, datax + l * n,
//                                                                      datay + l * n, dataz + l * n,
//                                                                      slice_compressed_size,
//                                                                      ord == nullptr ? nullptr : ord +
//                                                                                                 l *
//                                                                                                 n,
//                                                                      blkflag, bx, by, bz, bytes1,
//                                                                      compressed_size1, ord1);

//                size_t test_size = slice_compressed_size;

                if (bytes1 != bytes1p) {

                    ++cnt1;
                    write(compressed_size1, tail1s);
                    write(bytes1, compressed_size1, tail1s);
                    delete[] bytes1p;

//                    test_size += compressed_size1;
                }

//                printf("%zu\n", test_size);

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

//            printf("est time = %.6lfs\n", est_time);

            delete[] decmpdata1;

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
            size_t unxsize, unysize, unzsize;
            std::vector<size_t> unx, uny, unz;

            read(unxsize, cmpData);
            if (unxsize > 0) {
                encoder.load(cmpData, remaining_length);
                unx = encoder.decode(cmpData, unxsize);
                for (size_t i = 1; i < unx.size(); i++) unx[i] += unx[i - 1];
            }

            read(unysize, cmpData);
            if (unysize > 0) {
                encoder.load(cmpData, remaining_length);
                uny = encoder.decode(cmpData, unysize);
                for (size_t i = 1; i < uny.size(); i++) uny[i] += uny[i - 1];
            }

            read(unzsize, cmpData);
            if (unzsize > 0) {
                encoder.load(cmpData, remaining_length);
                unz = encoder.decode(cmpData, unzsize);
                for (size_t i = 1; i < unz.size(); i++) unz[i] += unz[i - 1];
            }
#endif

            size_t i = 0, j = 0;
            for (; i < blknum; i++) {

                if (i) blkst[i] += blkst[i - 1];

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

#if !__soft_eb
            for (size_t it: unx) {
                if (it & 1) {
                    datax[it >> 1] -= conf.absErrorBound;
                } else {
                    datax[it >> 1] += conf.absErrorBound;
                }
            }
            for (size_t it: uny) {
                if (it & 1) {
                    datay[it >> 1] -= conf.absErrorBound;
                } else {
                    datay[it >> 1] += conf.absErrorBound;
                }
            }
            for (size_t it: unz) {
                if (it & 1) {
                    dataz[it >> 1] -= conf.absErrorBound;
                } else {
                    dataz[it >> 1] += conf.absErrorBound;
                }
            }

#endif

            return;
        }

//        void decompressSimpleBlockingIntegers(const uchar *&lossless_data, T &px, T &py, T &pz, size_t *&datax, size_t *&datay, size_t *&dataz, size_t &outSize,
//                                               size_t cmpSize) {
//            uchar const *cmpData;
////            if(cmpSize > 0){
//            cmpData = lossless.decompress(lossless_data, cmpSize);
////            }
////            else{
////                cmpData = lossless_data;
////            }
//
//            SZ3::Config conf;
//            conf.load(cmpData);
//            outSize = conf.num;
//
//            if (datax == nullptr && datay == nullptr && dataz == nullptr) {
//                datax = new size_t[3 * conf.num];
//                datay = datax + conf.num;
//                dataz = datay + conf.num;
//            }
//
//            if (datax == nullptr) datax = new size_t[conf.num];
//            if (datay == nullptr) datay = new size_t[conf.num];
//            if (dataz == nullptr) dataz = new size_t[conf.num];
//
//            size_t bx, by, bz, nx, ny, nz, blknum;
//
//            read(px, cmpData);
//            read(py, cmpData);
//            read(pz, cmpData);
//
//            read(bx, cmpData);
//            read(by, cmpData);
//            read(bz, cmpData);
//
//            if (bx == 0 || by == 0 || bz == 0) {
//
////                memcpy(datax, cmpData, conf.num * sizeof(T));
////                for (size_t i = 0; i < conf.num; i++) datax[i] += px;
////                memcpy(datay, cmpData + conf.num * sizeof(T), conf.num * sizeof(T));
////                for (size_t i = 0; i < conf.num; i++) datay[i] += py;
////                memcpy(dataz, cmpData + (conf.num + conf.num) * sizeof(T), conf.num * sizeof(T));
////                for (size_t i = 0; i < conf.num; i++) dataz[i] += pz;
//
//
//                return;
//            }
//
//            read(nx, cmpData);
//            read(ny, cmpData);
//            read(nz, cmpData);
//            read(blknum, cmpData);
//
//            size_t remaining_length = 0;
//
//            encoder.load(cmpData, remaining_length);
//            auto blkst = encoder.decode(cmpData, blknum);
//
//            encoder.load(cmpData, remaining_length);
//            auto blkcnt = encoder.decode(cmpData, blknum);
//
//            encoder.load(cmpData, remaining_length);
//            auto quads = encoder.decode(cmpData, conf.num);
//
//            encoder.load(cmpData, remaining_length);
//            auto repos = encoder.decode(cmpData, conf.num);
//
//#if !__soft_eb
//            size_t cnt_unquants, unid = nx * ny * nz + 1;
//            read(cnt_unquants, cmpData);
//            std::vector<T> unx(cnt_unquants), uny(cnt_unquants), unz(cnt_unquants);
//            read(unx.data(), cnt_unquants, cmpData);
//            read(uny.data(), cnt_unquants, cmpData);
//            read(unz.data(), cnt_unquants, cmpData);
//#endif
//
//            size_t i = 0, j = 0;
//            for (; i < blknum; i++) {
//
//                if (i) blkst[i] += blkst[i - 1];
//
//#if !__soft_eb
//
//                if (blkst[i] == unid) {
//
//                    for (size_t j_ = 0; j_ < blkcnt[i]; j_++) {
//                        datax[j] = unx[j_];
//                        datay[j] = uny[j_];
//                        dataz[j] = unz[j_];
//                        ++j;
//                    }
//                    continue;
//                }
//#endif
//
//                size_t pbx = (blkst[i] % nx * 2) * bx;
//                size_t pby = (blkst[i] / nx % ny * 2) * by;
//                size_t pbz = (blkst[i] / nx / ny * 2) * bz;
//
//                size_t prequad = 0;
//                size_t prerepos = 0;
//
//                for (size_t j_ = 0; j_ < blkcnt[i]; j_++) {
//
//                    if (quads[j] != 0) prerepos = 0;
//                    size_t reposj = repos[j] + prerepos;
//                    size_t quadj = quads[j] + prequad;
//                    prerepos = reposj;
//                    prequad = quadj;
//
//                    size_t idx = (pbx + ((quadj & 0x01) >> 0) * bx + (reposj % bx));
//                    datax[j] = idx;
//                    size_t idy = (pby + ((quadj & 0x02) >> 1) * by + (reposj / bx % by));
//                    datay[j] = idy;
//                    size_t idz = (pbz + ((quadj & 0x04) >> 2) * bz + (reposj / bx / by));
//                    dataz[j] = idz;
//
//                    ++j;
//                }
//            }
//
//            return;
//        }

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

