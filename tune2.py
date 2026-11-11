import numpy as np
import pandas as pd
from kernel_tuner import tune_kernel

N = 512
ALPHA, BETA = 32412.0, 2123.0

kernel = """
#define N 512
__global__ void gemm_kernel(float alpha, float beta, float *a, float *b, float *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
#if USE_REGISTER
        float acc = c[i*N+j] * beta;
        for (int k = 0; k < N; k++) acc += alpha * a[i*N+k] * b[k*N+j];
        c[i*N+j] = acc;
#else
        c[i*N+j] *= beta;
        for (int k = 0; k < N; k++) c[i*N+j] += alpha * a[i*N+k] * b[k*N+j];
#endif
    }
}
"""

rng = np.random.default_rng(0)
A = rng.random((N, N), dtype=np.float32)
B = rng.random((N, N), dtype=np.float32)
C = rng.random((N, N), dtype=np.float32)
ref = (ALPHA * (A.astype(np.float64) @ B.astype(np.float64)) + BETA * C.astype(np.float64)).astype(np.float32).ravel()

args = [np.float32(ALPHA), np.float32(BETA), A.ravel(), B.ravel(), C.ravel()]
answer = [None] * 4 + [ref]

params = {
    "block_size_x": [16, 32, 64, 128],
    "block_size_y": [4, 8, 16, 32],
    "USE_REGISTER": [0, 1],
}

def verify(answer, result, atol=None):
    return np.allclose(answer[4], result[4], rtol=1e-3, atol=1.0)

results, _ = tune_kernel("gemm_kernel", kernel, (N, N), args, params,
                         grid_div_x=["block_size_x"], grid_div_y=["block_size_y"],
                         restrictions=["block_size_x * block_size_y <= 1024"],
                         answer=answer, verify=verify, iterations=50, verbose=False)

df = pd.DataFrame(results).sort_values("time")
df.to_csv("tune2.csv", index=False)
cols = ["block_size_x", "block_size_y", "USE_REGISTER", "time"]
print("\n5 meilleures configurations :")
print(df[cols].head(5).to_string(index=False))
orig = df[(df.block_size_x == 32) & (df.block_size_y == 8) & (df.USE_REGISTER == 0)]
print("\nConfiguration d'origine (32x8, USE_REGISTER=0) :")
print(orig[cols].to_string(index=False))
