import os
import sys
import numpy as np
import tvm

def main():
    argv = sys.argv
    argv_len = len(argv)
    if argv_len != 5:
        print("Invalid Usage!")
        print("Usage: python matmuladd_timer.py M N K ThreadNum")
        print("0 for ThreadNum by default!")
        exit(1)
    
    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])
    ThreadNum = int(argv[4])

    so_path = f"M{M}_N{N}_K{K}_{ThreadNum}/matmul_add.so"
    if not os.path.exists(so_path):
        print(f"Error! So file: <so_path> not exists!")
        exit(1)

    np.random.seed(149)
    a_np = np.random.uniform(size=(M, K)).astype(np.float32)
    b_np = np.random.uniform(size=(K, N)).astype(np.float32)
    c_np = np.random.uniform(size=(M, N)).astype(np.float32)
    out_np = a_np.dot(b_np) + c_np

    dev = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=dev)
    b_tvm = tvm.nd.array(b_np, device=dev)
    c_tvm = tvm.nd.array(c_np, device=dev)
    out_tvm = tvm.nd.empty(out_np.shape, device=dev)

    module = tvm.runtime.load_module(so_path)
    fmatmul_add = module['matmul_add']
    print(f"Matmuladd Evaluation for M={M} N={N} K={K} ThreadNum={ThreadNum}")
    ops = 2*M*N*K
    print(f"Total OPs is {ops}")
    
    if ThreadNum != 0:
        os.environ["TVM_NUM_THREADS"] = str(ThreadNum)
        print(f"ThreadNum is {ThreadNum}")
    else:
        print("ThreadNum by default")

    for i in range(9):
        fmatmul_add(a_tvm, b_tvm, c_tvm, out_tvm)
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    print()

    print("Median : ")
    print("GFLOPs = \n")

if __name__ == "__main__":
    main()
