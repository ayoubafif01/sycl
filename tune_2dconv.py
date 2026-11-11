import re, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from kernel_tuner import tune_kernel

SRC = Path.home() / "proj/polybenchGpu/CUDA/2DCONV"
DST = SRC.parent / "2DCONV_tuned"
cu  = (SRC / "2DConvolution.cu").read_text()
cuh = (SRC / "2DConvolution.cuh").read_text()

def define(text, name):
    m = re.search(rf"^\s*#\s*define\s+{name}\s+([0-9.]+)", text, re.M)
    assert m, f"#define {name} introuvable"
    return m.group(1)

def extract_function(text, marker):
    start = text.index(marker)
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    raise ValueError("accolade fermante introuvable")

assert re.search(r"#\s*define\s+DATA_TYPE\s+float", cuh), "DATA_TYPE n'est pas float : script a adapter"
NI, NJ = int(define(cuh, "NI")), int(define(cuh, "NJ"))
bx0, by0 = int(define(cuh, "DIM_THREAD_BLOCK_X")), int(define(cuh, "DIM_THREAD_BLOCK_Y"))
kernel_src = extract_function(cu, "__global__ void convolution2D_kernel")
coef = {k: float(v) for k, v in re.findall(r"c(\d\d)\s*=\s*([+-]?[0-9.]+)\s*;", kernel_src)}
assert len(coef) == 9, f"9 coefficients attendus, trouves : {coef}"
print(f"Lu dans Polybench : NI={NI} NJ={NJ} bloc d'origine={bx0}x{by0}")
print("Coefficients :", coef)

preamble = f"""
#define NI {NI}
#define NJ {NJ}
#define _PB_NI NI
#define _PB_NJ NJ
#define DATA_TYPE float
"""
kernel_string = preamble + kernel_src

rng = np.random.default_rng(0)
A = rng.random((NI, NJ), dtype=np.float32)
B = np.zeros((NI, NJ), dtype=np.float32)

Af = A.astype(np.float64)
ref = np.zeros((NI, NJ))
ref[1:-1, 1:-1] = (coef["11"]*Af[:-2, :-2] + coef["21"]*Af[:-2, 1:-1] + coef["31"]*Af[:-2, 2:]
                 + coef["12"]*Af[1:-1, :-2] + coef["22"]*Af[1:-1, 1:-1] + coef["32"]*Af[1:-1, 2:]
                 + coef["13"]*Af[2:, :-2]  + coef["23"]*Af[2:, 1:-1]  + coef["33"]*Af[2:, 2:])
ref = ref.astype(np.float32).ravel()

args = [np.int32(NI), np.int32(NJ), A.ravel(), B.ravel()]
answer = [None, None, None, ref]

tune_params = {"block_size_x": [8, 16, 32, 64, 128, 256],
               "block_size_y": [1, 2, 4, 8, 16, 32]}
restrictions = ["block_size_x * block_size_y <= 1024", "block_size_x * block_size_y >= 32"]

def verify(answer, result, atol=None):
    return np.allclose(answer[3], result[3], rtol=1e-4, atol=1e-4)

results, env = tune_kernel("convolution2D_kernel", kernel_string, (NJ, NI), args, tune_params,
                           grid_div_x=["block_size_x"], grid_div_y=["block_size_y"],
                           restrictions=restrictions, answer=answer, verify=verify, iterations=50)

df = pd.DataFrame(results).sort_values("time")
df.to_csv(Path.home() / "proj/2dconv_tuning.csv", index=False)
best = df.iloc[0]
bx, by = int(best.block_size_x), int(best.block_size_y)
ref_row = df[(df.block_size_x == bx0) & (df.block_size_y == by0)]

print("\n===== RESUME =====")
print(f"{len(df)} configurations valides")
print("\n5 meilleures :")
print(df[["block_size_x", "block_size_y", "time"]].head(5).to_string(index=False))
print(f"\nMeilleure : {bx}x{by} -> {best.time:.4f} ms")
if len(ref_row):
    t0 = float(ref_row.time.iloc[0])
    print(f"Origine   : {bx0}x{by0} -> {t0:.4f} ms  (gain x{t0 / best.time:.2f})")

if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("*.exe", "*.orig"))
h = (DST / "2DConvolution.cuh").read_text()
h = re.sub(r"(#\s*define\s+DIM_THREAD_BLOCK_X\s+)\d+", rf"\g<1>{bx}", h)
h = re.sub(r"(#\s*define\s+DIM_THREAD_BLOCK_Y\s+)\d+", rf"\g<1>{by}", h)
(DST / "2DConvolution.cuh").write_text(h)
subprocess.run(["make"], cwd=DST, check=True, capture_output=True)
print(f"\nVersion reglee generee et compilee : {DST}")
