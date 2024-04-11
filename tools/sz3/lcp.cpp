#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <iostream>
#include "SZ3/api/sz.hpp"
#include "SZ3/compressor/SZDiscreteCompressor.hpp"
#include "SZ3/frontend/SZGeneralFrontend.hpp"
#include "SZ3/encoder/HuffmanEncoder.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/def.hpp"
#include "SZ3/utils/Statistic.hpp"

void usage() {

    exit(0);
}

size_t max(size_t a, size_t b) {
    return a > b ? a : b;
}

template<class T>
void sortAndVerify(T *a, T *b, size_t *ord, size_t n, std::string str) {

//    std::set<size_t> st;
//    for(size_t i=0;i<n;i++) st.insert(ord[i]);
//    if(*st.begin() == 0 && *st.rbegin() == n-1 && st.size() == n){
//        printf("is a permutation\n");
//    }
//    else{
//        printf("not a permutation\n");
//        printf("min = %zu, max = %zu\n", *st.begin(), *st.rbegin());
//    }

    T *c = new T[n];

    for (size_t i = 0; i < n; i++) {
        c[ord[i]] = b[i];
    }

//    for(size_t i=0;i<n;i++){
//        if(fabs(a[i] - c[i]) > 1e-4){
//            printf("%zu %.12f %.12f %.12f\n", i, c[i], a[i], fabs(a[i] - c[i]));
//            exit(-1);
//        }
//    }

    printf("statistics of %s\n", str.data());

    double psnr, nrmse, max_diff;
    SZ3::verify<T>(a, c, n, psnr, nrmse, max_diff);

    delete[] c;
}

template<class T>
uchar *
compressWithoutAllocateMemory(T *data, SZ3::Config &conf, size_t &outSize, size_t *ord = nullptr, uchar blkflag = 0x00,
                              size_t bx = 0, size_t by = 0, size_t bz = 0) {

    /*
     * the length of data is 3 * conf.num
     */

    const size_t &n = conf.num;

    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compress(conf, data, outSize, ord, blkflag, bx, by, bz);

    return bytes;
}

template<class T>
uchar *
compressWithoutAllocateMemory(T *datax, T *datay, T *dataz, SZ3::Config &conf, size_t &outSize, size_t *ord = nullptr,
                              uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    size_t &n = conf.num;

    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compress(conf, datax, datay, dataz, outSize, ord, blkflag, bx, by, bz);

    return bytes;
}

/*
 * blkflagfore use this function, make sure you have at least least 12 * sizeof(T) allocated.
 */

template<class T>
uchar *compressWithAllocatedMemory(T *data, SZ3::Config conf, size_t &outSize, uchar *&dest, size_t *ord = nullptr,
                                   uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    uchar *bytes = compressWithoutAllocateMemory(data, conf, outSize, ord, blkflag, bx, by, bz);

    memcpy(dest, bytes, outSize);

    delete[] bytes;

    dest += outSize;

    return dest;
}

template<class T>
uchar *compressWithAllocatedMemory(T *datax, T *datay, T *dataz, SZ3::Config conf, size_t &outSize, uchar *&dest,
                                   size_t *ord = nullptr, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0,
                                   size_t bz = 0) {

    uchar *bytes = compressWithoutAllocateMemory(datax, datay, dataz, conf, outSize, ord, blkflag, bx, by, bz);

    memcpy(dest, bytes, outSize);

    delete[] bytes;

    dest += outSize;

    return dest;
}

template<class T>
void decompressWithoutAllocatedMemory(const uchar *cmpData, T *&datax, T *&datay, T *&dataz, size_t &outSize,
                                      size_t cmpSize) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressWithoutAllocateMemory(cmpData, datax, datay, dataz, outSize, cmpSize);
}

template<class T>
void
decompressWithAllocatedMemory(const uchar *cmpData, T *datax, T *datay, T *dataz, size_t &outSize, size_t cmpSize) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressWithAllocateMemory(cmpData, datax, datay, dataz, outSize, cmpSize);
}

template<class T>
void
decompressWithAllocatedMemoryWithTemporalPrediction(const uchar *cmpData, T *datax, T *datay, T *dataz, size_t &outSize,
                                                    size_t cmpSize) {

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    compressor.decompressSimpleBlockingWithTemporalPrediction(cmpData, datax, datay, dataz, outSize, cmpSize);
}

template<class T>
void deleteData(T *datax, T *datay, T *dataz, size_t n) {

    if (datay + n != dataz) delete[] dataz;
    if (datax + n != datay) delete[] datay;
    delete[] datax;
}

template<class T>
T *readFile(char *inPath[], size_t num_inPath, size_t n) {

    T *data = new T[n * num_inPath];
    for (size_t i = 0; i < num_inPath; i++) {
        SZ3::readfile<T>(inPath[i], n, data + i * n);
    }

    return data;
}

template<class T>
void compress1(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata = nullptr,
               size_t *ord = nullptr, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t outSize = 0;

    SZ3::Timer timer(true);

    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;

    uchar *bytes = compressWithoutAllocateMemory<T>(datax, datay, dataz, conf, outSize, ord, blkflag, bx, by, bz);
    delete[] data;

    double compress_time = timer.stop();

    SZ3::writefile(cmpPath, bytes, outSize);
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

template<class T>
void compress2(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata, size_t *ord,
               uchar blkflag = 0x00, size_t bx = 0, size_t by = 0, size_t bz = 0) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);
    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;
    T *tailx = datax, *taily = datay, *tailz = dataz;
    size_t *ordt = ord;

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t &nt = conf.dims[0], &n = conf.dims[1];

    uchar *bytes = new uchar[max(conf.num * 12, 65536)], *tail = bytes + (nt + 2) * sizeof(size_t), *tailp = bytes;
    SZ3::write(nt, tailp);
    SZ3::write(n, tailp);

    size_t outSize = 0, sumOutSize = 0;
    SZ3::Config confSlice(n);
    confSlice.absErrorBound = conf.absErrorBound;

    SZ3::Timer timer(true);

    if (ord == nullptr) {

        for (size_t t = 0; t < nt; t++) {

//            printf("t = %zu\n", t);

            compressWithAllocatedMemory(tailx, taily, tailz, confSlice, outSize, tail, nullptr, blkflag, bx, by, bz);

//            printf("t = %zu outSize = %zu\n", t, outSize);

            sumOutSize += outSize;
            SZ3::write(outSize, tailp);

            tailx += n;
            taily += n;
            tailz += n;
        }
    } else {
        for (size_t t = 0; t < nt; t++) {

//            printf("t = %zu\n", t);

            compressWithAllocatedMemory(tailx, taily, tailz, confSlice, outSize, tail, ordt, blkflag, bx, by, bz);

//            printf("t = %zu outSize = %zu\n", t, outSize);

            for (size_t i = 0; i < n; i++) ord[t * n + i] += t * n;

            sumOutSize += outSize;
            SZ3::write(outSize, tailp);

            tailx += n;
            taily += n;
            tailz += n;
            ordt += n;
        }
    }


    double compress_time = timer.stop();

    deleteData(datax, datay, dataz, conf.num);

    SZ3::writefile(cmpPath, bytes, (size_t) (tail - bytes));
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / (size_t) (tail - bytes));
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

template<class T>
void compress2WithTemporalPrediction(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf, T *oridata,
                                     size_t *ord, size_t bt, uchar blkflag = 0x00, size_t bx = 0, size_t by = 0,
                                     size_t bz = 0) {

    T *data = readFile<T>(inPath, num_inPath, conf.num);
    T *datax = data, *datay = datax + conf.num, *dataz = datay + conf.num;
    T *tailx = datax, *taily = datay, *tailz = dataz;
    size_t *ordt = ord;

    if (oridata != nullptr) memcpy(oridata, data, conf.num * 3 * sizeof(T));

    size_t &nt = conf.dims[0], &n = conf.dims[1];

    size_t outSize = 0;

    SZ3::Timer timer(true);

    static SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(
            SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());

    uchar *bytes = compressor.compressSimpleBlockingWithTemporalPrediction(conf, bt, datax, datay, dataz, outSize, ord,
                                                                           blkflag, bx, by, bz);

    double compress_time = timer.stop();

    deleteData(datax, datay, dataz, conf.num);

    SZ3::writefile(cmpPath, bytes, outSize);
    delete[] bytes;

    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / outSize);
    printf("compression time = %f\n", compress_time);
    printf("compressed data file = %s\n", cmpPath);
}

#include <random>

template<class T>
void shuffle3(T *&data, size_t n) {

    SZ3::Timer timer(true);

//    std::mt19937 rng(std::random_device{}());
    std::mt19937 rng(16707);

    size_t *p = new size_t[n];
    for (int i = 0; i < n; i++) p[i] = i;
    std::shuffle(p, p + n, rng);

    T *shuffled_data = new T[n * 3];
    for (int i = 0; i < n; i++) shuffled_data[i] = data[p[i]];
    for (int i = 0; i < n; i++) shuffled_data[n + i] = data[n + p[i]];
    for (int i = 0; i < n; i++) shuffled_data[2 * n + i] = data[2 * n + p[i]];

    delete[] p;
    delete[] data;

    data = shuffled_data;

    double shuffle_time = timer.stop();
    printf("shuffle time = %.2lf\n", shuffle_time);
}

//template <class T>
//void blockingCompress(char *inPath[], size_t num_inPath, char *cmpPath, SZ3::Config conf){
//
//    SZ3::Timer timer(true);
//
//    T *data = new T[conf.num*3];
//    for(int i=0;i<num_inPath;i++){
//        SZ3::readfile<T>(inPath[i], conf.num, data + i * conf.num);
//    }
//
////    shuffle3(data, conf.num);
//
//    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());
//
//    std::vector<size_t> szvec;
//    size_t n = conf.num;
//    while(n){
//        szvec.push_back(n-n/2);
//        n=n/2;
//    }
//    std::reverse(szvec.begin(), szvec.end());
//
////    for(size_t it:sz){
////        std::cout<<it<<' ';
////    }
////    std::cout<<std::endl;
//
//    uchar *bytes = new uchar[conf.num * 12], *tail = bytes;
//    size_t totalSize = 0, ssz = 0;
//
//    tail += 4 * (szvec.size() + 2);
//    uchar *tailp = bytes;
//    SZ3::write(conf.num, tailp);
//
//    for(size_t bsz:szvec){
//        SZ3::Config bconf = conf;
//        bconf.num = bsz;
////        printf("bsz = %zu\n",bsz);
//        size_t bSize;
//        T *bdata = new T[bsz * 3];
//        memcpy(bdata, data + ssz, bsz * sizeof(T));
//        memcpy(bdata + bsz, data + (conf.num + ssz), bsz * sizeof(T));
//        memcpy(bdata + 2 * bsz, data + (2 * conf.num + ssz), bsz * sizeof(T));
//
//        uchar *bbytes = compressor.compress(bconf, bdata, bSize);
//        memcpy(tail, bbytes, bSize);
//        delete[] bbytes;
//        SZ3::write((size_t)(tail - bytes), tailp);
//        tail += bSize;
//
//        ssz += bsz;
//    }
//
//    SZ3::write((size_t)(tail - bytes), tailp);
//
//    totalSize = tail - bytes;
//
//    SZ3::writefile(cmpPath, bytes, totalSize);
//    delete[] data;
//    delete[] bytes;
//
//    double compress_time = timer.stop();
//
//    printf("compression ratio = %.2f \n", conf.num * 1.0 * sizeof(T) * num_inPath / totalSize);
//    printf("compression time = %f\n", compress_time);
//    printf("compressed data file = %s\n", cmpPath);
//}

template<class T>
void decompress1(char *outPath[], size_t num_outPath, char *cmpPath, T *oridata = nullptr, size_t *ord = nullptr) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailData = cmpData.get();

    T *datax = nullptr, *datay = nullptr, *dataz = nullptr;

    size_t n;

    SZ3::Timer timer(true);

    decompressWithoutAllocatedMemory(tailData, datax, datay, dataz, n, cmpSize);

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, n, "x");
        sortAndVerify(oridata + n, datay, ord, n, "y");
        sortAndVerify(oridata + n + n, dataz, ord, n, "z");
    }

    SZ3::writefile(outPath[0], datax, n);
    SZ3::writefile(outPath[1], datay, n);
    SZ3::writefile(outPath[2], dataz, n);

    deleteData(datax, datay, dataz, n);

    printf("decompression time = %f\n", compress_time);
}

template<class T>
void decompress2(char *outPath[], int num_outPath, char *cmpPath, T *oridata = nullptr, size_t *ord = nullptr) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailPos = cmpData.get();
    size_t nt, n;
    SZ3::read(nt, tailPos);
    SZ3::read(n, tailPos);
    const uchar *tailData = tailPos + nt * sizeof(size_t);

    T *datax = new T[nt * n * 3], *datay = datax + nt * n, *dataz = datay + nt * n;
    T *tailx = datax, *taily = datay, *tailz = dataz;

    SZ3::Timer timer(true);

    for (size_t t = 0; t < nt; t++) {

        size_t n;
        size_t cmpSize;
        SZ3::read(cmpSize, tailPos);

        decompressWithAllocatedMemory(tailData, tailx, taily, tailz, n, cmpSize);
        tailData += cmpSize;
        tailx += n;
        taily += n;
        tailz += n;
    }

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, nt * n, "x");
        sortAndVerify(oridata + nt * n, datay, ord, nt * n, "y");
        sortAndVerify(oridata + 2 * nt * n, dataz, ord, nt * n, "z");
    }

    SZ3::writefile(outPath[0], datax, nt * n);
    SZ3::writefile(outPath[1], datay, nt * n);
    SZ3::writefile(outPath[2], dataz, nt * n);

    deleteData(datax, datay, dataz, nt * n);

    printf("decompression time = %f\n", compress_time);
}

template<class T>
void decompress2WithTemporalPrediction(char *outPath[], int num_outPath, char *cmpPath, T *oridata = nullptr,
                                       size_t *ord = nullptr) {

    size_t cmpSize;
    const auto cmpData = SZ3::readfile<uchar>(cmpPath, cmpSize);
    const uchar *tailPos = cmpData.get();
    size_t nt, n, bt;

    //
    {
        int cnt1;
        SZ3::read(cnt1, tailPos);
        while (cnt1--) {
            size_t tem;
            SZ3::read(tem, tailPos);
            tailPos += tem;
        }
    }

    SZ3::read(nt, tailPos);
    SZ3::read(n, tailPos);
    SZ3::read(bt, tailPos);

    T *datax = new T[nt * n * 3], *datay = datax + nt * n, *dataz = datay + nt * n;

    SZ3::Timer timer(true);

    size_t outSize;

    decompressWithAllocatedMemoryWithTemporalPrediction(cmpData.get(), datax, datay, dataz, outSize, cmpSize);

    double compress_time = timer.stop();

    if (oridata != nullptr && ord != nullptr) {

        sortAndVerify(oridata, datax, ord, nt * n, "x");
        sortAndVerify(oridata + nt * n, datay, ord, nt * n, "y");
        sortAndVerify(oridata + 2 * nt * n, dataz, ord, nt * n, "z");
    }

    SZ3::writefile(outPath[0], datax, nt * n);
    SZ3::writefile(outPath[1], datay, nt * n);
    SZ3::writefile(outPath[2], dataz, nt * n);

    deleteData(datax, datay, dataz, nt * n);

    printf("decompression time = %f\n", compress_time);
}

//template <class T>
//void blockingDecompress(char *outPath[], int num_outPath, char *cmpPath, double ratio){

//    SZ3::Timer timer(true);
//
//    size_t cmpSize;
//    const auto blockingData = SZ3::readfile<uchar>(cmpPath, cmpSize);
//    const uchar *blockingData_tail = blockingData.get();
//    SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd> compressor = SZ3::SZDiscreteCompressor<T, SZ3::HuffmanEncoder<size_t>, SZ3::Lossless_zstd>(SZ3::HuffmanEncoder<size_t>(), SZ3::Lossless_zstd());
//
//    std::vector<size_t> szvec, posvec;
//    size_t n = 0;
//    SZ3::read(n, blockingData_tail);
//    {
//        size_t tem = n;
//        while (tem) {
//            szvec.push_back(tem - tem / 2);
//            tem = tem / 2;
//        }
//    }
//    std::reverse(szvec.begin(), szvec.end());
//    posvec.resize(szvec.size() + 1);
//    for(size_t &it:posvec){
//        SZ3::read(it, blockingData_tail);
//        printf("%zu\n",it);
//    }
//    size_t ssz = 0;
//    std::vector<size_t> bno;
//    for(int i=szvec.size()-1;i>=0;i--){
//        if(1. * (ssz + szvec[i]) / n <= ratio){
//            bno.push_back(i);
//            ssz += szvec[i];
////            printf("(ssz + szvec[i]) / n = (%zu + %zu) / %zu = %f <= %f\n",ssz,szvec[i],n,1. * (ssz + szvec[i]) / n,ratio);
//        }
////        else{
////            printf("(ssz + szvec[i]) / n = (%zu + %zu) / %zu = %f > %f\n",ssz,szvec[i],n,1. * (ssz + szvec[i]) / n,ratio);
////        }
//    }
//
//    printf("ssz = %zu\n",ssz);
//
//    T *decmpData = new T[ssz*3];
//
//    // need to be done
//
//    size_t p = 0;
//
//    for(size_t it:bno){
//
//        printf("bid = %zu, sz = %zu\n", it, szvec[it]);
//
//
//    }
//
//    for(int i=0;i<num_outPath;i++){
//        SZ3::writefile(outPath[i], decmpData + i * ssz, ssz);
//    }
//
//    delete[] decmpData;
//
//    double compress_time = timer.stop();
//
//    printf("decompression time = %f\n", compress_time);
//}

signed main(int argc, char *argv[]) {

//    static SZ3::LinearQuantizer<float> quantizer(0.1, 65536);
//
//    float a = 141.595322, b = 141.595322;
//    int e = quantizer.quantize(a,b) - 65536;
//
//    printf("%d\n", e);
//    printf("%f %f\n", a, b);
//
//    return 0;

    char **inPath = new char *[3];
    char **outPath = new char *[3];
    for (int i = 0; i < 3; i++) inPath[i] = new char[1024];
    for (int i = 0; i < 3; i++) outPath[i] = new char[1024];
    char cmpPath[1024];

    uchar cmp = 0x00, decmp = 0x00, flag = 0x00;
    /*
     * flag = 1, compress without time dimension
     * flag = 2, compress with time dimension
     * flag = 3, blocking compression without time dimension
     */
    size_t nt, n;
    double eb = 0;
    size_t bt = 0;

    uchar blkflag = 0x01;
    size_t bx = 0, by = 0, bz = 0;

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
            snprintf(outPath[0], 1024, "%s.sz.out", inPath[0]);
            snprintf(outPath[1], 1024, "%s.sz.out", inPath[1]);
            snprintf(outPath[2], 1024, "%s.sz.out", inPath[2]);
            decmp = 0x01;
        } else if (strcmp(argv[i], "-eb") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%lf", &eb);
            i += 1;
        } else if (strcmp(argv[i], "-1") == 0) {
            assert(i + 1 < argc);
            flag = 1;
            sscanf(argv[i + 1], "%zu", &n);
            i += 1;
        } else if (strcmp(argv[i], "-2") == 0) {
            assert(i + 2 < argc);
            flag = 2;
            sscanf(argv[i + 1], "%zu", &nt);
            sscanf(argv[i + 2], "%zu", &n);
            i += 2;
        } else if (strcmp(argv[i], "-b") == 0) {

        } else if (strcmp(argv[i], "-a") == 0) {
            _a = 0x01;
        } else if (strcmp(argv[i], "-bt") == 0) {
            assert(i + 1 < argc);
            sscanf(argv[i + 1], "%zu", &bt);
            i += 1;
        } else if (strcmp(argv[i], "-be") == 0) {
            assert(i + 1 < argc);
            size_t tem;
            sscanf(argv[i + 1], "%zu", &tem);
            blkflag = tem;
            i += 1;
            if (blkflag == 0x00) {
                assert(i + 3 < argc);
                sscanf(argv[i + 1], "%zu", &bx);
                sscanf(argv[i + 2], "%zu", &by);
                sscanf(argv[i + 3], "%zu", &bz);
                i += 3;
            }
        } else {
            usage();
        }
    }

    SZ3::Config conf(n);
    if (flag == 2) conf = SZ3::Config(nt, n);

    conf.absErrorBound = eb;

    float *oridata = nullptr;
    size_t *ord = nullptr;

    if (_a) {
        if (cmp == 0x00) {
            printf("Must contain input while using -a\n");
            exit(-1);
        }
        oridata = new float[conf.num * 3];
        ord = new size_t[conf.num];
    }

    if (cmp == 0x01) {
        switch (flag) {
            case 1: {
                compress1<float>(inPath, 3, cmpPath, conf, oridata, ord, blkflag, bx, by, bz);
                break;
            }
            case 2: {
                if (bt > 0) {
                    compress2WithTemporalPrediction<float>(inPath, 3, cmpPath, conf, oridata, ord, bt, blkflag, bx, by,
                                                           bz);
                } else {
                    compress2<float>(inPath, 3, cmpPath, conf, oridata, ord, blkflag, bx, by, bz);
                }

                break;
            }
            case 3: {
//                blockingCompress<float>(inPath, 3, cmpPath, conf);
                break;
            }
            default: {
                exit(0);
            }
        }
    }

    if (decmp == 0x01) {
        switch (flag) {
            case 1: {
                decompress1<float>(outPath, 3, cmpPath, oridata, ord);
                break;
            }
            case 2: {
                if (bt > 1) {
                    decompress2WithTemporalPrediction<float>(outPath, 3, cmpPath, oridata, ord);
                } else {
                    decompress2<float>(outPath, 3, cmpPath, oridata, ord);
                }
                break;
            }
            case 3: {
//                blockingDecompress<float>(outPath, 3, cmpPath, 0.8);
                break;
            }
            default: {
                exit(0);
            }
        }
    }

    return 0;
}
