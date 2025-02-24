import os
import sys
import tvm
from tvm import te, auto_scheduler
import time
import shutil

@auto_scheduler.register_workload
def matmul_add(M, N, K, dtype):
    A = te.placeholder((M, K), name = "A", dtype = dtype)
    B = te.placeholder((K, N), name = "B", dtype = dtype)
    C = te.placeholder((M, N), name = "C", dtype = dtype)

    k = te.reduce_axis((0, K), name = "k")
    matmul = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
        name = "matmul",
        attrs = {"layout_free_placeholders": [B]},
    )
    out = te.compute((M, N), lambda i, j: matmul[i, j] + C[i, j], name = "out")
    return [A, B, C, out]


def main():
    argv = sys.argv
    argv_len = len(argv)
    if argv_len != 5:
        print("Invalid Usage!")
        print("Usage: python matmul_gencode.py M N K ThreadNum")
        print("0 for ThreadNum by default")
    
    M = int(argv[1])
    N = int(argv[2])
    K = int(argv[3])
    ThreadNum = int(argv[4])
    print(f"mm_gencode: M={M}, N={N} K={K} ThreadNum={ThreadNum}")
    foldername = f"M{M}_N{N}_K{K}_{ThreadNum}/"
    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.mkdir(foldername)
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(M, N, K, "float32"), target=target)
    log_file = foldername + "matmul_add.json"

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        #num_measure_trials=10,
        runner=auto_scheduler.LocalRunner(timeout=2000, repeat=10, enable_cpu_cache_flush=True),
        #runner=auto_scheduler.LocalRunner(timeout=1000, repeat=2, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=0,
    )
    
    # Set the ThreadNum, usually set to the number of logical cores could get the optimal performance
    if ThreadNum != 0:
        os.environ["TVM_NUM_THREADS"] = str(ThreadNum)
        print(f"ThreadNum is {ThreadNum}")
    else:
        print("ThreadNum by default")

    # Step 1: generate llvm code file
    # Run auto-tuning (search)
    try:
        task.tune(tune_option)
    except Exception as e:
        print(e)
        print("Exception happens in tuning")
        exit(1)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target, name="matmul_add")
    ll_code_str = func.get_source()
    ll_code_filename = foldername + "matmul_add.ll"
    ll_code_file = open(ll_code_filename, "w")
    ll_code_file.write(ll_code_str)
    ll_code_file.close()
    time.sleep(2)
    print("[Step 1: generate llvm code file] COMPLETED!")

    # Step 2: insert timestamp
    ll_code_file = open(ll_code_filename, 'r')
    lines, linecounter = ll_code_file.readlines(), 0
    # start
    for line in lines:
        if (line.find('define dllexport i32 @matmul_add(')!=-1 and lines[linecounter+1].strip()=='entry:'):
            lines.insert(linecounter+2, '  call void @get_time(i32 1)\n')
            break
        linecounter += 1

    # end
    linecounter = 0
    for line in lines:
        if (line.find('tail call fastcc i32 @matmul_add_compute_(')!=-1 and lines[linecounter+1].find('br label %common.ret')!=-1):
            lines[linecounter] = lines[linecounter].replace('tail', '')
            lines.insert(linecounter+1, '  call void @get_time(i32 0)\n')
            break
        linecounter += 1
    
    # add get_time(i32) declaration
    newLines, linecounter = [], 0
    for line in lines:
        if (line.find('attributes #0 = {')!=-1 and lines[linecounter+1].find('attributes #1 = {')!=-1):
            newLines.extend(lines[0:linecounter-1])
            newLines.insert(linecounter, 'declare void @get_time(i32)\n\n')
            newLines.extend(lines[linecounter:])
            break
        linecounter += 1
    ll_code_file.close()

    ll_code_newfilename = foldername + "matmul_add_new.ll"
    ll_code_newfile = open(ll_code_newfilename, 'w')
    ll_code_newfile.writelines(newLines)
    ll_code_newfile.close()
    time.sleep(2)
    print("[Step 2: insert timestamp] COMPLETED!")

    # Step 3: generate so file
    # clang -shared -fPIC -mavx2 -march=native -O3 M1024_N1024_K1024/matmul_add_new.ll -o M1024_N1024_K1024/matmul_add.so gettime.c
    sofile_path = f"{foldername}matmul_add.so"
    gen_so_cmd = f"clang -shared -fPIC -mavx2 -march=native -O3 {ll_code_newfilename} -o {sofile_path} gettime.c"
    os.system(gen_so_cmd)
    print(f"[Step 3 : generate so file <{sofile_path}>] COMPLETED!")

if __name__ == "__main__":
    main()
