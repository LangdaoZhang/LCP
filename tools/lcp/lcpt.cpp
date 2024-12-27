#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include "SZ3/api/sz.hpp"
#include "SZ3/compressor/SZLCPTreeCompressor.hpp"
#include "SZ3/frontend/SZGeneralFrontend.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/def.hpp"
#include "SZ3/utils/Statistic.hpp"

void usage() {

}

typedef SZ3::uchar uchar;

template<class T>
uchar *compressWithoutAllocateMemory(T *datax, T *datay, T *dataz, const SZ3::Config &conf, size_t &outSize,
                                     size_t *ord = nullptr) {
    const size_t &n = conf.num;

    SZ3::SZLCPTreeCompressor<T, SZ3::HuffmanEncoder<int64_t>, SZ3::Lossless_zstd> compressor =
            SZ3::SZLCPTreeCompressor<T, SZ3::HuffmanEncoder<int64_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<int64_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compress(conf, datax, datay, dataz, outSize, ord);

    return bytes;
}

template<class T>
T *readFile(char *inPath[], size_t num_inPath, size_t n) {

    T *data = new T[n * num_inPath];
    for (size_t i = 0; i < num_inPath; i++) {
        SZ3::readfile<T>(inPath[i], n, data + i * n);
    }

    return data;
}

template<typename T>
void compress(char *inPath[], char *cmpPath, const SZ3::Config &conf, T *oridata = nullptr, size_t *ord = nullptr) {

    const size_t &n = conf.num;

    T *data = readFile<T>(inPath, 3, n);

    if (oridata != nullptr) memcpy(oridata, data, n * 3 * sizeof(T));

    size_t outSize = 0;

    SZ3::Timer timer(true);

    T *datax = data, *datay = datax +n, *dataz = datay + n;

    uchar *bytes = compressWithoutAllocateMemory(datax, datay, dataz, conf, outSize, ord);
    delete[] data;

    double compress_time = timer.stop();

    SZ3::writefile(cmpPath, bytes, outSize);
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * 3 / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

signed main(int argc, char *argv[]) {

    char **inPath = new char *[3];
    char **outPath = new char *[3];
    for (int i = 0; i < 3; i++) inPath[i] = new char[1024];
    for (int i = 0; i < 3; i++) outPath[i] = new char[1024];
    char cmpPath[1024];
    char ordPath[1024];

    uchar cmp = 0x00, decmp = 0x00, output_ord = 0x00;
    size_t ordBits = 64;

    size_t n;

    double eb = 1e-3;

    uchar _a = 0x00;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            assert(i + 3 < argc);
            snprintf(inPath[0], 1024, "%s", argv[i + 1]);
            snprintf(inPath[1], 1024, "%s", argv[i + 2]);
            snprintf(inPath[2], 1024, "%s", argv[i + 3]);
            cmp = 0x01;
            i += 3;
        } else if (strcmp(argv[i], "-z") == 0) {
            assert(i + 1 < argc);
            snprintf(cmpPath, 1024, "%s", argv[i + 1]);
            i += 1;
        } else if (strcmp(argv[i], "-o") == 0) {
            assert(i + 3 < argc);
            snprintf(outPath[0], 1024, "%s", argv[i + 1]);
            snprintf(outPath[1], 1024, "%s", argv[i + 2]);
            snprintf(outPath[2], 1024, "%s", argv[i + 3]);
            decmp = 0x01;
            i += 3;
        } else if (strcmp(argv[i], "-osn") == 0) {
            // output same name
            snprintf(outPath[0], 1024, "%s.lcp.out", inPath[0]);
            snprintf(outPath[1], 1024, "%s.lcp.out", inPath[1]);
            snprintf(outPath[2], 1024, "%s.lcp.out", inPath[2]);
            decmp = 0x01;
        } else if (strcmp(argv[i], "-eb") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%lf", &eb);
            i += 1;
        } else if (strcmp(argv[i], "-1") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%zu", &n);
            i += 1;
        } else if (strcmp(argv[i], "-a") == 0) {
            _a = 0x01;
        } else if (strcmp(argv[i], "-ord") == 0) {
            assert(i + 2 < argc);
            sscanf(argv[i + 1], "%zu", &ordBits);
//            assert(ordBits == 32 || ordBits == 64);
            if (ordBits != 32 && ordBits != 64) {
                printf("ordBits must be 32 or 64.\n");
                exit(-1);
            }
            sscanf(argv[i + 2], "%s", ordPath);
            output_ord = 0x01;
            i += 2;
        } else {
            usage();
        }

    }

    SZ3::Config conf(n);
    conf.absErrorBound = eb;

    float *oridata = nullptr;
    size_t *ord = nullptr;

    if (_a || output_ord) {
        if (cmp == 0x00) {
            printf("Must contain input while using -a or -ord\n");
            exit(-1);
        }
        oridata = new float[conf.num * 3];
        ord = new size_t[conf.num];
    }

    if (cmp == 0x01) {
        compress<float>(inPath, cmpPath, conf, oridata, ord);
    }

    if (output_ord) {
        if (ordBits == 32) {
            uint32_t *ordu32 = new uint32_t[conf.num];
            for (size_t i = 0; i < conf.num; i++) {
                ordu32[i] = ord[i];
            }
            SZ3::writefile(ordPath, ordu32, conf.num);
            delete[] ordu32;
        }
        else {
            SZ3::writefile(ordPath, ord, conf.num);
        }
    }

    if (decmp == 0x01) {
        //
    }

    return 0;
}