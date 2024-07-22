SIGMOD 25 Paper 561 Artifact README
=====

LCP: Enhancing Scientific Data Management with Lossy Compression for Particles
=====

# Important functions of LCP:

* /tools/sz3/lcp.cpp
* /include/SZ3/compressor/SZDiscreteCompressor.hpp

# How to build LCP:

[//]: # (## Installation)

* mkdir build && cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
* make lcp
* make install

# How to download the data:

use the following link to download the test data (Copper):

https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-exaalt-copper.tar.gz

The file names of the test data are dataset1-5423x3137.x.f32.dat, dataset1-5423x3137.y.f32.dat, dataset1-5423x3137.z.f32.dat.

## Usage

The default datatype is 32 bit float numbers. 

### Arguments

| Arguments                                          | Explanation                                                                |
|----------------------------------------------------|----------------------------------------------------------------------------|
| -i [input_file_x] [input_file_y] [input_file_z]    | The three input files need to be compressed                                |
| -z [compressed_file]                               | The compressed file name                                                   |
| -o [output_file_x] [output_file_y] [output_file_z] | The three decompressed files                                               |
| -osn                                               | Decompress with the name + '.lcp.out' corresponding to the input files     |
| -eb [error_bound]                                  | The user set absolute error bound                                          |
| -1 [number_of_particles]                           | The number of particles                                                    |
| -2 [number_of_frames] [number_of_particles]        | The numbers of time frames and the number of particles                     |
| -bt [bath_size]                                    | The maximum number of frames within one batch                              |
| -a                                                 | Output the compression results like the maximum absolute error, PSNR, etc. |

[//]: # (|                                                    |                                                                            |)

[//]: # (|                                                    |                                                                            |)

[//]: # (|                                                    |                                                                            |)

[//]: # (|                                                    |                                                                            |)

# How to test:

For example, you can try the following command for the test:

### Download the data:
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-exaalt-copper.tar.gz

tar -xzf SDRBENCH-exaalt-copper.tar.gz

cd SDRBENCH-exaalt-copper/

### Compress and decompress at the same time with verification:
lcp -i dataset1-5423x3137.x.f32.dat dataset1-5423x3137.y.f32.dat dataset1-5423x3137.z.f32.dat -z cmp.lcp -osn -2 5423 3137 -eb 1e-3 -bt 16 -a

### Compress only:
lcp -i dataset1-5423x3137.x.f32.dat dataset1-5423x3137.y.f32.dat dataset1-5423x3137.z.f32.dat -z cmp.lcp -2 5423 3137 -eb 1e-3 -bt 16

### Decompress:
lcp -z cmp.lcp -o dataset1-5423x3137.x.f32.dat.lcp.out dataset1-5423x3137.y.f32.dat.lcp.out dataset1-5423x3137.z.f32.dat.lcp.out -2 5423 3137



# Note: LCP is built up on the SZ3 framework, as explained in Section 5 of the paper. The following is the README from SZ3


## SZ3: A Modular Error-bounded Lossy Compression Framework for Scientific Datasets

(C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory. See COPYRIGHT in top-level directory.

* Major Authors: Sheng Di, Kai Zhao, Xin Liang
* Supervisor: Franck Cappello
* Other Contributors: Robert Underwood, Sihuan Li, Ali M. Gok

## Citations

**Kindly note**: If you mention SZ in your paper, the most appropriate citation is including these three references (**TBD22, ICDE21, Bigdata18**), because they cover the design and implementation of the latest version of SZ.

* SZ3 Framework: Xin Liang, Kai Zhao, Sheng Di, Sihuan Li, Robert Underwood, Ali M Gok, Jiannan Tian, Junjing Deng, Jon C Calhoun, Dingwen Tao, Zizhong Chen, and Franck Cappello.
  "[SZ3: A modular framework for composing prediction-based error-bounded lossy compressors](https://ieeexplore.ieee.org/abstract/document/9866018)",
  IEEE Transactions on Big Data (TBD 22).

* SZ3 Algorithm: Kai Zhao, Sheng Di, Maxim Dmitriev, Thierry-Laurent D. Tonellot, Zizhong Chen, and Franck
  Cappello. "[Optimizing Error-Bounded Lossy Compression for Scientiﬁc Data by Dynamic Spline Interpolation](https://ieeexplore.ieee.org/document/9458791)"
  , Proceeding of the 37th IEEE International Conference on Data Engineering (ICDE 21), Chania, Crete, Greece, Apr 19 -
  22, 2021.

* SZauto: Kai Zhao, Sheng Di, Xin Liang, Sihuan Li, Dingwen Tao, Zizhong Chen, and Franck
  Cappello. "[Significantly Improving Lossy Compression for HPC Datasets with Second-Order Prediction and Parameter Optimization](https://dl.acm.org/doi/10.1145/3369583.3392688)"
  , Proceedings of the 29th International Symposium on High-Performance Parallel and Distributed Computing (HPDC 20),
  Stockholm, Sweden, 2020. (code: https://github.com/szcompressor/SZauto/)

* SZ 2.0+: Xin Liang, Sheng Di, Dingwen Tao, Zizhong Chen, Franck
  Cappello, "[Error-Controlled Lossy Compression Optimized for High Compression Ratios of Scientific Datasets](https://ieeexplore.ieee.org/document/8622520)"
  , in IEEE International Conference on Big Data (Bigdata 2018), Seattle, WA, USA, 2018.

* SZ 1.4.0-1.4.13: Dingwen Tao, Sheng Di, Franck
  Cappello. "[Significantly Improving Lossy Compression for Scientific Data Sets Based on Multidimensional Prediction and Error-Controlled Quantization](https://ieeexplore.ieee.org/document/7967203)"
  , in IEEE International Parallel and Distributed Processing Symposium (IPDPS 2017), Orlando, Florida, USA, 2017.

* SZ 0.1-1.0: Sheng Di, Franck
  Cappello. "[Fast Error-bounded Lossy HPC Data Compression with SZ](https://ieeexplore.ieee.org/document/7516069)", in
  IEEE International Parallel and Distributed Processing Symposium (IPDPS 2016), Chicago, IL, USA, 2016.

* Point-wise relative error bound mode (i.e., PW_REL): Xin Liang, Sheng Di, Dingwen Tao, Zizhong Chen, Franck
  Cappello, "[An Efficient Transformation Scheme for Lossy Data Compression with Point-wise Relative Error Bound](https://ieeexplore.ieee.org/document/8514879)"
  , in IEEE International Conference on Clustering Computing (CLUSTER 2018), Belfast, UK, 2018. (Best Paper)

## 3rd party libraries/tools

* Zstandard (https://facebook.github.io/zstd/). Zstandard v1.4.5 is included and will be used if libzstd can not be found by
  pkg-config.

## Installation

* mkdir build && cd build
* cmake -DCMAKE_INSTALL_PREFIX:PATH=[INSTALL_DIR] ..
* make
* make install

Then, you'll find all the executables in [INSTALL_DIR]/bin and header files in [INSTALL_DIR]/include

## Testing Examples

You can use the executable 'sz3' command to do the compression/decompression.

SZ3 simplifies command line arguments in the previous version. If you are a new user, please follow the instructions
given by the executable.


## Backward Compatibility with SZ2
For backward compatibility, most of the SZ2 command line parameters are supported in SZ3. **Exceptions are listed below**.
Scripts without parameters below should work fine by replacing SZ2 with SZ3.

| Parameter | Explanation                     | SZ3 roadmap                              |
|-----------|---------------------------------|------------------------------------------|
| -c        | Config file                     | SZ3 has different config format with SZ2 |
| -p        | Print configuration info        | Will be supported soon                   |
| -T        | Tucker Tensor Decomposition     | Will be supported later                  |
| -P        | Point-wise relative error bound | Will be supported later                  |


## API

#### SZ3 C++ API
* Located in 'include/SZ3/api/sz.hpp'. 
* Requiring a modern C++ compiler.  
* Different with SZ2 API.

#### SZ3 C API
* Located in 'tools/sz3c/include/sz3c.h'
* Compatible with SZ2 API

#### Python API
* Located in 'tools/pysz/pysz.py'
* Test file provided ('tools/pysz/test.py')
* Compatible with both SZ3 and SZ2
* Requiring SZ2/3 dynamic library

## H5Z-SZ3

Use examples/print_h5repack_args.c to construct the cd_values parameters based on the specified error configuration. 

* Example: 

Compression: 

h5repack -f UD=32024,0,5,0,981668463,0,0,0 -i ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.h5 -o ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.sz3.h5

Decompression:

h5repack -f NONE -i ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.sz3.h5 -o ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.sz3.out.h5


Alternatively, the error bound information can also be given through sz3.config (when there are no cd_values for h5repack)

* Example (You need to put sz3.config in the current local directory so that it will read sz3.config to get error bounds):

h5repack -f UD=32024,0 -i ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.h5 -o ~/Data/CESM-ATM-tylor/1800x3600/CLDLOW_1_1800_3600.dat.sz3.h5

## Version history

Version New features

* SZ 3.0.0 SZ3 is the C++ version of SZ with modular and composable design.
* SZ 3.0.1 Improve the build process.
* SZ 3.1.0 The default algorithm is now interpolation+Lorenzo.
* SZ 3.1.1 Add OpenMP support. Works for all algorithms. Please enable it using the config file. 
* SZ 3.1.2 Support configuration file (INI format). Example can be found in 'tools/sz3/sz3.config'.
* SZ 3.1.3 Support more error control mode: PSNR, L2Norm, ABS_AND_REL, ABS_OR_REL. Support INT32 and INT64 datatype.
* SZ 3.1.4 Support running on Windows. Please refer to https://github.com/szcompressor/SZ3/issues/5#issuecomment-1094039224 for instructions.
* SZ 3.1.5 Support HDF5 by H5Z-SZ3. Please add "-DBUILD_H5Z_FILTER=ON" to enable this function for CMake.
* SZ 3.1.6 Support C API and Python API.
* SZ 3.1.7 Initial MDZ(https://github.com/szcompressor/SZ3/tree/master/tools/mdz) support.
* SZ 3.1.8 namespace changed from SZ to SZ3. H5Z-SZ3 supports configuration file now.
