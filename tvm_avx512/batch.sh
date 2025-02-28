#!/bin/bash

mkdir -p outlog/llama3/ out/llama3/ outlog/gemma27b/ out/gemma27b/ outlog/gemma9b/ out/gemma9b/ outlog/gemma7b/ out/gemma7b/ outlog/gemma2b/ out/gemma2b/

# ThreadNum 32
# Llama3
python matmul_gencode.py 4096 128 4096 32 2>&1 | tee outlog/llama3/llama3_4096_128_4096_32.log
python matmul_timer.py 4096 128 4096 32 2>&1 | tee out/llama3/llama3_4096_128_4096_32

python matmul_gencode.py 128 8192 4096 32 2>&1 | tee outlog/llama3/llama3_128_8192_4096_32.log
python matmul_timer.py 128 8192 4096 32 2>&1 | tee out/llama3/llama3_128_8192_4096_32

python matmul_gencode.py 128 4096 8192 32 2>&1 | tee outlog/llama3/llama3_128_4096_8192_32.log
python matmul_timer.py 128 4096 8192 32 2>&1 | tee out/llama3/llama3_128_4096_8192_32

python matmul_gencode.py 4096 4096 4096 32 2>&1 | tee outlog/llama3/llama3_4096_4096_4096_32.log
python matmul_timer.py 4096 4096 4096 32 2>&1 | tee out/llama3/llama3_4096_4096_4096_32

################################################################################
# Gemma27B
python matmul_gencode.py 4608 256 4096 32 2>&1 | tee outlog/gemma27b/gemma27b_4608_256_4096_32.log
python matmul_timer.py 4608 256 4096 32 2>&1 | tee out/gemma27b/gemma27b_4608_256_4096_32

python matmul_gencode.py 256 8192 4608 32 2>&1 | tee outlog/gemma27b/gemma27b_256_8192_4608_32.log
python matmul_timer.py 256 8192 4608 32 2>&1 | tee out/gemma27b/gemma27b_256_8192_4608_32

python matmul_gencode.py 256 4608 8192 32 2>&1 | tee outlog/gemma27b/gemma27b_256_4608_8192_32.log
python matmul_timer.py 256 4608 8192 32 2>&1 | tee out/gemma27b/gemma27b_256_4608_8192_32

python matmul_gencode.py 4608 4608 36864 32 2>&1 | tee outlog/gemma27b/gemma27b_4608_4608_36864_32.log
python matmul_timer.py 4608 4608 36864 32 2>&1 | tee out/gemma27b/gemma27b_4608_4608_36864_32

################################################################################
# Gemma9B
python matmul_gencode.py 3584 256 4096 32 2>&1 | tee outlog/gemma9b/gemma9b_3584_256_4096_32.log
python matmul_timer.py 3584 256 4096 32 2>&1 | tee out/gemma9b/gemma9b_3584_256_4096_32

python matmul_gencode.py 256 8192 3584 32 2>&1 | tee outlog/gemma9b/gemma9b_256_8192_3584_32.log
python matmul_timer.py 256 8192 3584 32 2>&1 | tee out/gemma9b/gemma9b_256_8192_3584_32

python matmul_gencode.py 256 3584 8192 32 2>&1 | tee outlog/gemma9b/gemma9b_256_3584_8192_32.log
python matmul_timer.py 256 3584 8192 32 2>&1 | tee out/gemma9b/gemma9b_256_3584_8192_32

python matmul_gencode.py 3584 3584 14336 32 2>&1 | tee outlog/gemma9b/gemma9b_3584_3584_14336_32.log
python matmul_timer.py 3584 3584 14336 32 2>&1 | tee out/gemma9b/gemma9b_3584_3584_14336_32

################################################################################
# Gemma7B
python matmul_gencode.py 3072 256 4096 32 2>&1 | tee outlog/gemma7b/gemma7b_3072_256_4096_32.log
python matmul_timer.py 3072 256 4096 32 2>&1 | tee out/gemma7b/gemma7b_3072_256_4096_32

python matmul_gencode.py 256 8192 3072 32 2>&1 | tee outlog/gemma7b/gemma7b_256_8192_3072_32.log
python matmul_timer.py 256 8192 3072 32 2>&1 | tee out/gemma7b/gemma7b_256_8192_3072_32

python matmul_gencode.py 256 3072 8192 32 2>&1 | tee outlog/gemma7b/gemma7b_256_3072_8192_32.log
python matmul_timer.py 256 3072 8192 32 2>&1 | tee out/gemma7b/gemma7b_256_3072_8192_32

python matmul_gencode.py 3072 3072 24576 32 2>&1 | tee outlog/gemma7b/gemma7b_3072_3072_24576_32.log
python matmul_timer.py 3072 3072 24576 32 2>&1 | tee out/gemma7b/gemma7b_3072_3072_24576_32

################################################################################
# Gemma2B
python matmul_gencode.py 2048 256 4096 32 2>&1 | tee outlog/gemma2b/gemma2b_2048_256_4096_32.log
python matmul_timer.py 2048 256 4096 32 2>&1 | tee out/gemma2b/gemma2b_2048_256_4096_32

python matmul_gencode.py 256 8192 2048 32 2>&1 | tee outlog/gemma2b/gemma2b_256_8192_2048_32.log
python matmul_timer.py 256 8192 2048 32 2>&1 | tee out/gemma2b/gemma2b_256_8192_2048_32

python matmul_gencode.py 256 2048 8192 32 2>&1 | tee outlog/gemma2b/gemma2b_256_2048_8192_32.log
python matmul_timer.py 256 2048 8192 32 2>&1 | tee out/gemma2b/gemma2b_256_2048_8192_32

python matmul_gencode.py 2048 2048 16384 32 2>&1 | tee outlog/gemma2b/gemma2b_2048_2048_16384_32.log
python matmul_timer.py 2048 2048 16384 32 2>&1 | tee out/gemma2b/gemma2b_2048_2048_16384_32




# ThreadNum 16

# Llama3
python matmul_gencode.py 4096 128 4096 16 2>&1 | tee outlog/llama3/llama3_4096_128_4096_16.log
python matmul_timer.py 4096 128 4096 16 2>&1 | tee out/llama3/llama3_4096_128_4096_16

python matmul_gencode.py 128 8192 4096 16 2>&1 | tee outlog/llama3/llama3_128_8192_4096_16.log
python matmul_timer.py 128 8192 4096 16 2>&1 | tee out/llama3/llama3_128_8192_4096_16

python matmul_gencode.py 128 4096 8192 16 2>&1 | tee outlog/llama3/llama3_128_4096_8192_16.log
python matmul_timer.py 128 4096 8192 16 2>&1 | tee out/llama3/llama3_128_4096_8192_16

python matmul_gencode.py 4096 4096 4096 16 2>&1 | tee outlog/llama3/llama3_4096_4096_4096_16.log
python matmul_timer.py 4096 4096 4096 16 2>&1 | tee out/llama3/llama3_4096_4096_4096_16

################################################################################
# Gemma27B
python matmul_gencode.py 4608 256 4096 16 2>&1 | tee outlog/gemma27b/gemma27b_4608_256_4096_16.log
python matmul_timer.py 4608 256 4096 16 2>&1 | tee out/gemma27b/gemma27b_4608_256_4096_16

python matmul_gencode.py 256 8192 4608 16 2>&1 | tee outlog/gemma27b/gemma27b_256_8192_4608_16.log
python matmul_timer.py 256 8192 4608 16 2>&1 | tee out/gemma27b/gemma27b_256_8192_4608_16

python matmul_gencode.py 256 4608 8192 16 2>&1 | tee outlog/gemma27b/gemma27b_256_4608_8192_16.log
python matmul_timer.py 256 4608 8192 16 2>&1 | tee out/gemma27b/gemma27b_256_4608_8192_16

python matmul_gencode.py 4608 4608 36864 16 2>&1 | tee outlog/gemma27b/gemma27b_4608_4608_36864_16.log
python matmul_timer.py 4608 4608 36864 16 2>&1 | tee out/gemma27b/gemma27b_4608_4608_36864_16

################################################################################
# Gemma9B
python matmul_gencode.py 3584 256 4096 16 2>&1 | tee outlog/gemma9b/gemma9b_3584_256_4096_16.log
python matmul_timer.py 3584 256 4096 16 2>&1 | tee out/gemma9b/gemma9b_3584_256_4096_16

python matmul_gencode.py 256 8192 3584 16 2>&1 | tee outlog/gemma9b/gemma9b_256_8192_3584_16.log
python matmul_timer.py 256 8192 3584 16 2>&1 | tee out/gemma9b/gemma9b_256_8192_3584_16

python matmul_gencode.py 256 3584 8192 16 2>&1 | tee outlog/gemma9b/gemma9b_256_3584_8192_16.log
python matmul_timer.py 256 3584 8192 16 2>&1 | tee out/gemma9b/gemma9b_256_3584_8192_16

python matmul_gencode.py 3584 3584 14336 16 2>&1 | tee outlog/gemma9b/gemma9b_3584_3584_14336_16.log
python matmul_timer.py 3584 3584 14336 16 2>&1 | tee out/gemma9b/gemma9b_3584_3584_14336_16

################################################################################
# Gemma7B
python matmul_gencode.py 3072 256 4096 16 2>&1 | tee outlog/gemma7b/gemma7b_3072_256_4096_16.log
python matmul_timer.py 3072 256 4096 16 2>&1 | tee out/gemma7b/gemma7b_3072_256_4096_16

python matmul_gencode.py 256 8192 3072 16 2>&1 | tee outlog/gemma7b/gemma7b_256_8192_3072_16.log
python matmul_timer.py 256 8192 3072 16 2>&1 | tee out/gemma7b/gemma7b_256_8192_3072_16

python matmul_gencode.py 256 3072 8192 16 2>&1 | tee outlog/gemma7b/gemma7b_256_3072_8192_16.log
python matmul_timer.py 256 3072 8192 16 2>&1 | tee out/gemma7b/gemma7b_256_3072_8192_16

python matmul_gencode.py 3072 3072 24576 16 2>&1 | tee outlog/gemma7b/gemma7b_3072_3072_24576_16.log
python matmul_timer.py 3072 3072 24576 16 2>&1 | tee out/gemma7b/gemma7b_3072_3072_24576_16

################################################################################
# Gemma2B
python matmul_gencode.py 2048 256 4096 16 2>&1 | tee outlog/gemma2b/gemma2b_2048_256_4096_16.log
python matmul_timer.py 2048 256 4096 16 2>&1 | tee out/gemma2b/gemma2b_2048_256_4096_16

python matmul_gencode.py 256 8192 2048 16 2>&1 | tee outlog/gemma2b/gemma2b_256_8192_2048_16.log
python matmul_timer.py 256 8192 2048 16 2>&1 | tee out/gemma2b/gemma2b_256_8192_2048_16

python matmul_gencode.py 256 2048 8192 16 2>&1 | tee outlog/gemma2b/gemma2b_256_2048_8192_16.log
python matmul_timer.py 256 2048 8192 16 2>&1 | tee out/gemma2b/gemma2b_256_2048_8192_16

python matmul_gencode.py 2048 2048 16384 16 2>&1 | tee outlog/gemma2b/gemma2b_2048_2048_16384_16.log
python matmul_timer.py 2048 2048 16384 16 2>&1 | tee out/gemma2b/gemma2b_2048_2048_16384_16




# ThreadNum 0

# Llama3
python matmul_gencode.py 4096 128 4096 0 2>&1 | tee outlog/llama3/llama3_4096_128_4096_0.log
python matmul_timer.py 4096 128 4096 0 2>&1 | tee out/llama3/llama3_4096_128_4096_0

python matmul_gencode.py 128 8192 4096 0 2>&1 | tee outlog/llama3/llama3_128_8192_4096_0.log
python matmul_timer.py 128 8192 4096 0 2>&1 | tee out/llama3/llama3_128_8192_4096_0

python matmul_gencode.py 128 4096 8192 0 2>&1 | tee outlog/llama3/llama3_128_4096_8192_0.log
python matmul_timer.py 128 4096 8192 0 2>&1 | tee out/llama3/llama3_128_4096_8192_0

python matmul_gencode.py 4096 4096 4096 0 2>&1 | tee outlog/llama3/llama3_4096_4096_4096_0.log
python matmul_timer.py 4096 4096 4096 0 2>&1 | tee out/llama3/llama3_4096_4096_4096_0

################################################################################
# Gemma27B
python matmul_gencode.py 4608 256 4096 0 2>&1 | tee outlog/gemma27b/gemma27b_4608_256_4096_0.log
python matmul_timer.py 4608 256 4096 0 2>&1 | tee out/gemma27b/gemma27b_4608_256_4096_0

python matmul_gencode.py 256 8192 4608 0 2>&1 | tee outlog/gemma27b/gemma27b_256_8192_4608_0.log
python matmul_timer.py 256 8192 4608 0 2>&1 | tee out/gemma27b/gemma27b_256_8192_4608_0

python matmul_gencode.py 256 4608 8192 0 2>&1 | tee outlog/gemma27b/gemma27b_256_4608_8192_0.log
python matmul_timer.py 256 4608 8192 0 2>&1 | tee out/gemma27b/gemma27b_256_4608_8192_0

python matmul_gencode.py 4608 4608 36864 0 2>&1 | tee outlog/gemma27b/gemma27b_4608_4608_36864_0.log
python matmul_timer.py 4608 4608 36864 0 2>&1 | tee out/gemma27b/gemma27b_4608_4608_36864_0

################################################################################
# Gemma9B
python matmul_gencode.py 3584 256 4096 0 2>&1 | tee outlog/gemma9b/gemma9b_3584_256_4096_0.log
python matmul_timer.py 3584 256 4096 0 2>&1 | tee out/gemma9b/gemma9b_3584_256_4096_0

python matmul_gencode.py 256 8192 3584 0 2>&1 | tee outlog/gemma9b/gemma9b_256_8192_3584_0.log
python matmul_timer.py 256 8192 3584 0 2>&1 | tee out/gemma9b/gemma9b_256_8192_3584_0

python matmul_gencode.py 256 3584 8192 0 2>&1 | tee outlog/gemma9b/gemma9b_256_3584_8192_0.log
python matmul_timer.py 256 3584 8192 0 2>&1 | tee out/gemma9b/gemma9b_256_3584_8192_0

python matmul_gencode.py 3584 3584 14336 0 2>&1 | tee outlog/gemma9b/gemma9b_3584_3584_14336_0.log
python matmul_timer.py 3584 3584 14336 0 2>&1 | tee out/gemma9b/gemma9b_3584_3584_14336_0

################################################################################
# Gemma7B
python matmul_gencode.py 3072 256 4096 0 2>&1 | tee outlog/gemma7b/gemma7b_3072_256_4096_0.log
python matmul_timer.py 3072 256 4096 0 2>&1 | tee out/gemma7b/gemma7b_3072_256_4096_0

python matmul_gencode.py 256 8192 3072 0 2>&1 | tee outlog/gemma7b/gemma7b_256_8192_3072_0.log
python matmul_timer.py 256 8192 3072 0 2>&1 | tee out/gemma7b/gemma7b_256_8192_3072_0

python matmul_gencode.py 256 3072 8192 0 2>&1 | tee outlog/gemma7b/gemma7b_256_3072_8192_0.log
python matmul_timer.py 256 3072 8192 0 2>&1 | tee out/gemma7b/gemma7b_256_3072_8192_0

python matmul_gencode.py 3072 3072 24576 0 2>&1 | tee outlog/gemma7b/gemma7b_3072_3072_24576_0.log
python matmul_timer.py 3072 3072 24576 0 2>&1 | tee out/gemma7b/gemma7b_3072_3072_24576_0

################################################################################
# Gemma2B
python matmul_gencode.py 2048 256 4096 0 2>&1 | tee outlog/gemma2b/gemma2b_2048_256_4096_0.log
python matmul_timer.py 2048 256 4096 0 2>&1 | tee out/gemma2b/gemma2b_2048_256_4096_0

python matmul_gencode.py 256 8192 2048 0 2>&1 | tee outlog/gemma2b/gemma2b_256_8192_2048_0.log
python matmul_timer.py 256 8192 2048 0 2>&1 | tee out/gemma2b/gemma2b_256_8192_2048_0

python matmul_gencode.py 256 2048 8192 0 2>&1 | tee outlog/gemma2b/gemma2b_256_2048_8192_0.log
python matmul_timer.py 256 2048 8192 0 2>&1 | tee out/gemma2b/gemma2b_256_2048_8192_0

python matmul_gencode.py 2048 2048 16384 0 2>&1 | tee outlog/gemma2b/gemma2b_2048_2048_16384_0.log
python matmul_timer.py 2048 2048 16384 0 2>&1 | tee out/gemma2b/gemma2b_2048_2048_16384_0

