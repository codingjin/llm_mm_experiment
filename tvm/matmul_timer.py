import os
import sys
import numpy as np
import tvm

def main():
    argv = sys.argv
    if len(argv) != 4:
        print("Invalid input!")
        print("python matmuladd_timer.py M N K!")
        exit(1)
    
    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])

    so_path = f"M{M}_N{N}_K{K}/matmul_add.so"
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
    print(f"Matmuladd Evaluation for M={M} N={N} K={K}")
    ops = 2*M*N*K
    print(f"Total OPs is {ops}")
    
    os.environ["TVM_NUM_THREADS"] = str(20)
    print("ThreadNum is 20")

    for i in range(9):
        fmatmul_add(a_tvm, b_tvm, c_tvm, out_tvm)
        np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
    print()

    print("Median : ")
    print("GFLOPs = ")
    """
    thread_nums = [0, 8, 16, 20]
    for threadnum in thread_nums:
        if threadnum == 0:
            print("ThreadNum by default")
        else:
            os.environ["TVM_NUM_THREADS"] = str(threadnum)
            print(f"ThreadNum = {threadnum}")

        for i in range(9):
            fmatmul_add(a_tvm, b_tvm, c_tvm, out_tvm)
            np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
        print()
    """

if __name__ == "__main__":
    main()
