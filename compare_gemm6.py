import os, random, re, statistics as st, subprocess, sys, csv
from pathlib import Path
import numpy as np

H = Path.home()
V = {
 "CUDA origine":  (H/"proj/polybenchGpu/CUDA/GEMM/gemm.exe", False),
 "CUDA bloc":     (H/"proj/polybenchGpu/CUDA/GEMM_tuned/gemm.exe", False),
 "CUDA bloc+reg": (H/"proj/polybenchGpu/CUDA/GEMM_reg/gemm.exe", False),
 "SYCL origine":  (H/"proj/sycl/gemm/CUDA/GEMM/gemm_sycl.exe", True),
 "SYCL bloc":     (H/"proj/sycl/gemm_tuned/CUDA/GEMM_tuned/gemm_sycl.exe", True),
 "SYCL bloc+reg": (H/"proj/sycl/gemm_reg/CUDA/GEMM_reg/gemm_sycl.exe", True),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = H/"proj/gemm_comparison_6.csv"
for n, (p, _) in V.items():
    if not p.exists(): sys.exit(f"introuvable : {p}")

def run(name):
    p, sycl = V[name]
    env = os.environ.copy()
    if sycl: env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    out = subprocess.run([str(p)], capture_output=True, text=True, env=env, cwd=p.parent).stdout.splitlines()
    t = float(out[next(i for i, l in enumerate(out) if l.startswith("GPU Time")) + 1]) * 1000
    ok = bool(re.search(r"Percent:\s*0\s*$", out[-1]))
    return t, ok

rng = random.Random(0)
names = list(V)
for _ in range(2):
    for n in names: run(n)

res = {n: [] for n in names}; bad = {n: 0 for n in names}; rows = []
for r in range(N):
    rng.shuffle(names)
    for n in names:
        t, ok = run(n)
        res[n].append(t); bad[n] += (not ok); rows.append((r, n, t, ok))
    if (r + 1) % 10 == 0: print(f"tour {r + 1}/{N}", flush=True)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tour", "version", "gpu_ms", "resultat_ok"]); w.writerows(rows)

print(f"\n{'version':15}{'n':>4}{'médiane':>9}{'moyenne':>9}{'écart-t.':>9}{'min':>8}{'max':>8}{'erreurs':>8}")
for n in V:
    x = res[n]
    print(f"{n:15}{len(x):4}{st.median(x):9.3f}{st.mean(x):9.3f}{st.stdev(x):9.3f}{min(x):8.3f}{max(x):8.3f}{bad[n]:8}")

idx = np.random.default_rng(0).integers(0, N, (10000, N))
def ratio(a, b):   # temps(a) / temps(b)
    x, y = np.array(res[a]), np.array(res[b])
    r = np.median(x[idx], axis=1) / np.median(y[idx], axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return np.median(x) / np.median(y), lo, hi

print("\nGain par rapport à la version d'origine (x2 = deux fois plus rapide) :")
for lang in ("CUDA", "SYCL"):
    for v in ("bloc", "bloc+reg"):
        g, lo, hi = ratio(f"{lang} origine", f"{lang} {v}")
        print(f"  {lang} {v:9} : x{g:.2f}   IC95% [{lo:.2f} ; {hi:.2f}]")
print("\nTemps SYCL / temps CUDA (< 1 : SYCL plus rapide) :")
for v in ("origine", "bloc", "bloc+reg"):
    g, lo, hi = ratio(f"SYCL {v}", f"CUDA {v}")
    print(f"  {v:9} : {g:.2f}   IC95% [{lo:.2f} ; {hi:.2f}]")
print(f"\nDonnées : {OUT}")
