# cmake min version
cmake_minimum_required(VERSION 3.5)

# project's name
PROJECT(TEST C)

# set var
SET(OPENBLAS_DIR /home/jin/OpenBLAS-develop/install/)
SET(OPENBLAS_INC /home/jin/OpenBLAS-develop/install/include/)
SET(OPENBLAS_LIB /home/jin/OpenBLAS-develop/install/lib/)

link_directories(${OPENBLAS_LIB})
include_directories(${OPENBLAS_INC})

add_executable(openblas_gemm "src/openblas_gemm.c")
target_link_libraries(openblas_gemm -L${OPENBLAS_LIB} libopenblas.a -lm -lpthread)

