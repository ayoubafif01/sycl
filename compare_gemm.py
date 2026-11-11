import os, random, re, statistics as st, subprocess, sys, csv
from pathlib import Path

H = Path.home()
V = {
 "CUDA origine": (H/"proj/polybenchGpu/CUDA/GEMM/gemm.exe", False),
 "CUDA tunee":   (H/"proj/polybenchGpu/CUDA/GEMM_tuned/gemm.exe", False),
 "SYCL origine": (H/"proj/sycl/gemm/CUDA/GEMM/gemm_sycl.exe", True),
 "SYCL tunee":   (H/"proj/sycl/gemm_tuned/CUDA/GEMM_tuned/gemm_sycl.exe", True),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
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

res = {n: [] for n in names}
bad = {n: 0 for n in names}
rows = []
for r in range(N):
    rng.shuffle(names)
    for n in names:
        t, ok = run(n)
        res[n].append(t); bad[n] += (not ok); rows.append((r, n, t, ok))

with open(H/"proj/gemm_comparison.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["tour", "version", "gpu_ms", "resultat_ok"]); w.writerows(rows)

print(f"{'version':14}{'n':>4}{'médiane':>9}{'moyenne':>9}{'écart-t.':>9}{'min':>8}{'max':>8}{'erreurs':>8}")
for n in V:
    x = res[n]
    print(f"{n:14}{len(x):4}{st.median(x):9.3f}{st.mean(x):9.3f}{st.stdev(x):9.3f}{min(x):8.3f}{max(x):8.3f}{bad[n]:8}")
m = {n: st.median(res[n]) for n in V}
print(f"\nCUDA tunée / origine : {m['CUDA origine']/m['CUDA tunee']:.3f}x")
print(f"SYCL tunée / origine : {m['SYCL origine']/m['SYCL tunee']:.3f}x")
print(f"SYCL / CUDA (origine): {m['SYCL origine']/m['CUDA origine']:.3f}x en temps")
print(f"SYCL / CUDA (tunée)  : {m['SYCL tunee']/m['CUDA tunee']:.3f}x en temps")
