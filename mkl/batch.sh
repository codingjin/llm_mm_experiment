#!/bin/bash
mkdir -p llama3/ gemma27b/ gemma9b/ gemma7b/ gemma2b/

# Llama3
echo "Llama3"
./mkl_gemm 4096 128 4096 | tee llama3/llama3_4096_128_4096

./mkl_gemm 128 8192 4096 | tee llama3/llama3_128_8192_4096

./mkl_gemm 128 4096 8192 | tee llama3/llama3_128_4096_8192

./mkl_gemm 4096 4096 4096 | tee llama3/llama3_4096_4096_4096

################################################################################
# Gemma27B
echo "Gemma27B"
./mkl_gemm 4608 256 4096 | tee gemma27b/gemma27b_4608_256_4096

./mkl_gemm 256 8192 4608 | tee gemma27b/gemma27b_256_8192_4608

./mkl_gemm 256 4608 8192 | tee gemma27b/gemma27b_256_4608_8192

./mkl_gemm 4608 4608 36864 | tee gemma27b/gemma27b_4608_4608_36864

################################################################################
# Gemma9B
echo "Gemma9B"
./mkl_gemm 3584 256 4096 | tee gemma9b/gemma9b_3584_256_4096

./mkl_gemm 256 8192 3584 | tee gemma9b/gemma9b_256_8192_3584

./mkl_gemm 256 3584 8192 | tee gemma9b/gemma9b_256_3584_8192

./mkl_gemm 3584 3584 14336 | tee gemma9b/gemma9b_3584_3584_14336

################################################################################
# Gemma7B
echo "Gemma7B"
./mkl_gemm 3072 256 4096 | tee gemma7b/gemma7b_3072_256_4096

./mkl_gemm 256 8192 3072 | tee gemma7b/gemma7b_256_8192_3072

./mkl_gemm 256 3072 8192 | tee gemma7b/gemma7b_256_3072_8192

./mkl_gemm 3072 3072 24576 | tee gemma7b/gemma7b_3072_3072_24576

################################################################################
# Gemma2B
echo "Gemma2B"
./mkl_gemm 2048 256 4096 | tee gemma2b/gemma2b_2048_256_4096

./mkl_gemm 256 8192 2048 | tee gemma2b/gemma2b_256_8192_2048

./mkl_gemm 256 2048 8192 | tee gemma2b/gemma2b_256_2048_8192

./mkl_gemm 2048 2048 16384 | tee gemma2b/gemma2b_2048_2048_16384

