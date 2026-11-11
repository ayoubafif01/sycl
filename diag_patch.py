p = "GEMM_diag/gemm.cu"
s = open(p).read()
old = """        for (int w = 0; w < 20; w++)
                gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
        cudaDeviceSynchronize();
"""
new = """        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 20; w++) {
                cudaEventRecord(e0);
                gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                fprintf(stderr, "lancement %d : %.3f ms\\n", w, ms);
        }
        cudaDeviceSynchronize();
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
