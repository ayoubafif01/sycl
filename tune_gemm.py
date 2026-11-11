import numpy as np
import pandas as pd
from kernel_tuner import tune_kernel

NI = NJ = NK = 512            # même taille que gemm.cuh
ALPHA, BETA = 32412.0, 2123.0 # constantes de gemm.cu
LANG = None                   # en cas d'erreur de backend, essaie "NVCUDA"

header = f"#define NI {NI}\n#define NJ {NJ}\n#define NK {NK}\n#define DATA_TYPE float\n"
kernel_string = header + """
__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
                            DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < NI) && (j < NJ))
    {
        c[i * NJ + j] *= beta;
        for (int k = 0; k < NK; k++)
        {
            c[i * NJ + j] += alpha * a[i * NK + k] * b[k * NJ + j];
        }
    }
}
"""

rng = np.random.default_rng(0)
A = rng.random((NI, NK), dtype=np.float32)
B = rng.random((NK, NJ), dtype=np.float32)
C = rng.random((NI, NJ), dtype=np.float32)

C_ref = ALPHA * (A.astype(np.float64) @ B.astype(np.float64)) + BETA * C.astype(np.float64)
C_ref = C_ref.astype(np.float32).ravel()

args = [np.int32(NI), np.int32(NJ), np.int32(NK),
        np.float32(ALPHA), np.float32(BETA),
        A.ravel(), B.ravel(), C.ravel()]
answer = [None] * 7 + [C_ref]   # seul C est une sortie

tune_params = {
    "block_size_x": [8, 16, 32, 64, 128, 256],
    "block_size_y": [1, 2, 4, 8, 16, 32],
}
restrictions = ["block_size_x * block_size_y <= 1024",
                "block_size_x * block_size_y >= 32"]

def verify(answer, result, atol=None):
    return np.allclose(answer[7], result[7], rtol=1e-3, atol=1.0)

results, env = tune_kernel(
    "gemm_kernel", kernel_string, (NJ, NI), args, tune_params,
    grid_div_x=["block_size_x"], grid_div_y=["block_size_y"],
    restrictions=restrictions, answer=answer, verify=verify,
    lang=LANG, iterations=50,
)

df = pd.DataFrame(results).sort_values("time")
df.to_csv("gemm_tuning.csv", index=False)
print("\nTop 10 :")
print(df[["block_size_x", "block_size_y", "time"]].head(10).to_string(index=False))
best = df.iloc[0]
print(f"\nMeilleure : {int(best.block_size_x)}x{int(best.block_size_y)} -> {best.time:.4f} ms")
ref = df[(df.block_size_x == 32) & (df.block_size_y == 8)]
if len(ref):
    print(f"Origine 32x8 -> {float(ref.time.iloc[0]):.4f} ms")
