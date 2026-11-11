import numpy as np, pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("/home/afif/proj/gemm_comparison.csv")
w = df.pivot(index="tour", columns="version", values="gpu_ms")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(w), (10000, len(w)))

def cmp(label, a, b):
    x, y = w[a].to_numpy(), w[b].to_numpy()
    r = np.median(x[idx], axis=1) / np.median(y[idx], axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    p = wilcoxon(x, y).pvalue
    print(f"{label:24} {np.median(x)/np.median(y):.3f}  IC95% [{lo:.3f} ; {hi:.3f}]  p={p:.4f}")

print("rapport = temps(a) / temps(b) ; > 1 : b est plus rapide\n")
cmp("CUDA : origine / tunée", "CUDA origine", "CUDA tunee")
cmp("SYCL : origine / tunée", "SYCL origine", "SYCL tunee")
cmp("origine : SYCL / CUDA", "SYCL origine", "CUDA origine")
cmp("tunée : SYCL / CUDA", "SYCL tunee", "CUDA tunee")
