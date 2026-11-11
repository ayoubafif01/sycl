import os, random, re, statistics as st, subprocess, sys
import numpy as np
from pathlib import Path

# usage : python compare.py N nom1=chemin1 nom2=chemin2[:sycl] ...   (1er = référence)
N = int(sys.argv[1]); V = {}
for a in sys.argv[2:]:
    name, path = a.split("=", 1)
    sycl = path.endswith(":sycl")
    V[name] = (Path(path.removesuffix(":sycl")).expanduser(), sycl)
for n, (p, _) in V.items():
    if not p.exists(): sys.exit(f"introuvable : {p}")

def run(name):
    p, sycl = V[name]
    env = os.environ.copy()
    if sycl: env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    out = subprocess.run([str(p)], capture_output=True, text=True, env=env, cwd=p.parent).stdout.splitlines()
    t = float(out[next(i for i, l in enumerate(out) if l.startswith("GPU Time")) + 1]) * 1000
    return t, bool(re.search(r"Percent:\s*0\s*$", out[-1]))

rng = random.Random(0); names = list(V)
for _ in range(2):
    for n in names: run(n)
res = {n: [] for n in names}; bad = {n: 0 for n in names}
for r in range(N):
    rng.shuffle(names)
    for n in names:
        t, ok = run(n); res[n].append(t); bad[n] += (not ok)

print(f"{'version':16}{'n':>4}{'médiane':>9}{'moyenne':>9}{'min':>8}{'max':>8}{'erreurs':>8}")
for n in V:
    x = res[n]
    print(f"{n:16}{len(x):4}{st.median(x):9.3f}{st.mean(x):9.3f}{min(x):8.3f}{max(x):8.3f}{bad[n]:8}")
idx = np.random.default_rng(0).integers(0, N, (10000, N))
first = list(V)[0]
print(f"\nTemps / temps de '{first}' (< 1 : plus rapide), IC95% :")
for n in list(V)[1:]:
    x, y = np.array(res[n]), np.array(res[first])
    r = np.median(x[idx], axis=1) / np.median(y[idx], axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    print(f"  {n:16} {np.median(x)/np.median(y):.3f}  [{lo:.3f} ; {hi:.3f}]")
