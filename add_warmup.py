import sys
p = sys.argv[1]
s = open(p).read()
marker = "        /* Start timer. */"
assert s.count(marker) == 1 and "WARMUP" not in s
warm = """        /* WARMUP: lancements a blanc puis remise de C a sa valeur initiale */
        for (int w = 0; w < 20; w++)
                gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
        cudaDeviceSynchronize();
        cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

"""
open(p, "w").write(s.replace(marker, warm + marker))
