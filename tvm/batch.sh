#!/bin/bash

# Llama3
python matmul_gencode.py 4096 128 4096 2>&1 | tee outlog/llama3_4096_128_4096.log
python matmul_timer.py 4096 128 4096 2>&1 | tee llama3_4096_128_4096

python matmul_gencode.py 128 8192 4096 2>&1 | tee outlog/llama3_128_8192_4096.log
python matmul_timer.py 128 8192 4096 2>&1 | tee llama3_128_8192_4096

python matmul_gencode.py 128 4096 8192 2>&1 | tee outlog/llama3_128_4096_8192.log
python matmul_timer.py 128 4096 8192 2>&1 | tee llama3_128_4096_8192

python matmul_gencode.py 4096 4096 4096 2>&1 | tee outlog/llama3_4096_4096_4096.log
python matmul_timer.py 4096 4096 4096 2>&1 | tee llama3_4096_4096_4096

################################################################################
# Gemma27B
python matmul_gencode.py 4608 256 4096 2>&1 | tee outlog/gemma27b_4608_256_4096.log
python matmul_timer.py 4608 256 4096 | tee gemma27b_4608_256_4096

python matmul_gencode.py 256 8192 4608 2>&1 | tee outlog/gemma27b_256_8192_4608.log
python matmul_timer.py 256 8192 4608 2>&1 | tee gemma27b_256_8192_4608

python matmul_gencode.py 256 4608 8192 2>&1 | tee outlog/gemma27b_256_4608_8192.log
python matmul_timer.py 256 4608 8192 2>&1 | tee gemma27b_256_4608_8192

python matmul_gencode.py 4608 4608 36864 2>&1 | tee outlog/gemma27b_4608_4608_36864.log
python matmul_timer.py 4608 4608 36864 2>&1 | tee gemma27b_4608_4608_36864

################################################################################
# Gemma9B
python matmul_gencode.py 3584 256 4096 2>&1 | tee outlog/gemma9b_3584_256_4096.log
python matmul_timer.py 3584 256 4096 2>&1 | tee gemma9b_3584_256_4096

python matmul_gencode.py 256 8192 3584 2>&1 | tee outlog/gemma9b_256_8192_3584.log
python matmul_timer.py 256 8192 3584 2>&1 | tee gemma9b_256_8192_3584

python matmul_gencode.py 256 3584 8192 2>&1 | tee outlog/gemma9b_256_3584_8192.log
python matmul_timer.py 256 3584 8192 2>&1 | tee gemma9b_256_3584_8192

python matmul_gencode.py 3584 3584 14336 2>&1 | tee outlog/gemma9b_3584_3584_14336.log
python matmul_timer.py 3584 3584 14336 2>&1 | tee gemma9b_3584_3584_14336

################################################################################
# Gemma7B
python matmul_gencode.py 3072 256 4096 2>&1 | tee outlog/gemma7b_3072_256_4096.log
python matmul_timer.py 3072 256 4096 2>&1 | tee gemma7b_3072_256_4096

python matmul_gencode.py 256 8192 3072 2>&1 | tee outlog/gemma7b_256_8192_3072.log
python matmul_timer.py 256 8192 3072 2>&1 | tee gemma7b_256_8192_3072

python matmul_gencode.py 256 3072 8192 2>&1 | tee outlog/gemma7b_256_3072_8192.log
python matmul_timer.py 256 3072 8192 2>&1 | tee gemma7b_256_3072_8192

python matmul_gencode.py 3072 3072 24576 2>&1 | tee outlog/gemma7b_3072_3072_24576.log
python matmul_timer.py 3072 3072 24576 2>&1 | tee gemma7b_3072_3072_24576

################################################################################
# Gemma2B
python matmul_gencode.py 2048 256 4096 2>&1 | tee outlog/gemma2b_2048_256_4096.log
python matmul_timer.py 2048 256 4096 2>&1 | tee gemma2b_2048_256_4096

python matmul_gencode.py 256 8192 2048 2>&1 | tee outlog/gemma2b_256_8192_2048.log
python matmul_timer.py 256 8192 2048 2>&1 | tee gemma2b_256_8192_2048

python matmul_gencode.py 256 2048 8192 2>&1 | tee outlog/gemma2b_256_2048_8192.log
python matmul_timer.py 256 2048 8192 2>&1 | tee gemma2b_256_2048_8192

python matmul_gencode.py 2048 2048 16384 2>&1 | tee outlog/gemma2b_2048_2048_16384.log
python matmul_timer.py 2048 2048 16384 2>&1 | tee gemma2b_2048_2048_16384
