import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

CSV  = sys.argv[1] if len(sys.argv) > 1 else "/home/afif/proj/gemm_comparison.csv"
BASE = sys.argv[2] if len(sys.argv) > 2 else "/home/afif/proj/gemm_figure"
TITRE = "GEMM : effet du réglage de la taille de bloc, en CUDA et en SYCL"
BLOC_ORIG, BLOC_TUNE = "32×8", "64×16"
ENV = "GTX 1650 (WSL2) · CUDA 12.8 · DPC++ nightly 2026-10-25 · SYCLomatic 20260928"
C_ORIG, C_TUNE, C_NS = "#A0AAB4", "#1F4E79", "#7F7F7F"

def fr(x, d=1, signed=False):
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s.replace(".", ",")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

df = pd.read_csv(CSV)
w = df.pivot(index="tour", columns="version", values="gpu_ms")
n = len(w)
idx = np.random.default_rng(0).integers(0, n, (10000, n))   # même rééchantillonnage (apparié)

def med_ci(col):
    x = w[col].to_numpy()
    lo, hi = np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])
    return np.median(x), lo, hi

def gain(ref, cmp_):   # en % ; positif = cmp_ plus rapide que ref
    x, y = w[ref].to_numpy(), w[cmp_].to_numpy()
    g = (np.median(x[idx], axis=1) / np.median(y[idx], axis=1) - 1) * 100
    lo, hi = np.percentile(g, [2.5, 97.5])
    return (np.median(x) / np.median(y) - 1) * 100, lo, hi

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})

# (a) temps médians, axe depuis zéro
bw, his = 0.36, []
for gi, (o, t) in enumerate([("CUDA origine", "CUDA tunee"), ("SYCL origine", "SYCL tunee")]):
    for k, (col, color, lab) in enumerate([(o, C_ORIG, f"Origine (bloc {BLOC_ORIG})"),
                                           (t, C_TUNE, f"Tunée (bloc {BLOC_TUNE})")]):
        m, lo, hi = med_ci(col)
        x = gi + (k - 0.5) * bw * 1.08
        a1.bar(x, m, bw, color=color, label=lab if gi == 0 else None, zorder=2)
        a1.errorbar(x, m, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]], color="black",
                    capsize=4, lw=1.2, zorder=3)
        a1.text(x, hi + 0.08, fr(m, 2), ha="center", va="bottom", fontsize=10)
        his.append(hi)
a1.set_xticks([0, 1]); a1.set_xticklabels(["CUDA", "SYCL"], fontsize=12)
a1.set_ylim(0, max(his) * 1.35)
a1.set_ylabel("Temps GPU, médiane (ms)")
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fr(v, 0)))
a1.grid(axis="x", visible=False)
a1.legend(frameon=False, loc="upper center", ncol=2, fontsize=10)
a1.set_title("(a) Temps médian et intervalle de confiance à 95 %")

# (b) écarts entre versions
rows = [("Réglage du bloc, CUDA\n(origine → tunée)", "CUDA origine", "CUDA tunee"),
        ("Réglage du bloc, SYCL\n(origine → tunée)", "SYCL origine", "SYCL tunee"),
        ("SYCL contre CUDA\n(versions d'origine)",    "CUDA origine", "SYCL origine"),
        ("SYCL contre CUDA\n(versions tunées)",       "CUDA tunee",   "SYCL tunee")]
ys = np.arange(len(rows))[::-1]
res = [gain(r, c) for _, r, c in rows]
for y, (g, lo, hi) in zip(ys, res):
    etabli = lo > 0 or hi < 0
    a2.errorbar(g, y, xerr=[[max(g - lo, 0)], [max(hi - g, 0)]], fmt="o", ms=9,
                color=C_TUNE if etabli else C_NS, mfc=C_TUNE if etabli else "white",
                mew=1.8, capsize=5, lw=1.8, zorder=3)
    a2.text(g, y + 0.2, f"{fr(g, 1, True)} %", ha="center", va="bottom", fontsize=10, fontweight="bold")
a2.axvline(0, color="black", lw=1, zorder=1)
xmin = min(0, min(r[1] for r in res)); xmax = max(0, max(r[2] for r in res)); pad = 0.18 * (xmax - xmin)
a2.set_xlim(xmin - pad, xmax + pad)
a2.set_ylim(-1.0, len(rows) - 0.4)
a2.set_yticks(ys); a2.set_yticklabels([r[0] for r in rows])
a2.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: "0" if abs(v) < 1e-9 else fr(v, 0 if abs(v - round(v)) < 1e-9 else 1, True)))
a2.set_xlabel("Gain de performance (%) ; positif : la version comparée est plus rapide")
a2.grid(axis="y", visible=False)
a2.legend(handles=[
    Line2D([0], [0], marker="o", lw=0, color=C_TUNE, mfc=C_TUNE, ms=8, label="Écart établi (IC 95 % exclut 0)"),
    Line2D([0], [0], marker="o", lw=0, color=C_NS, mfc="white", mew=1.6, ms=8, label="Écart non établi (IC 95 % contient 0)")],
    loc="lower right", frameon=False, fontsize=9)
a2.set_title("(b) Écarts entre versions")

fig.suptitle(TITRE, fontsize=14, fontweight="bold", y=0.985)
fig.text(0.5, 0.012,
         f"Chronomètre Polybench (un lancement par exécution), {n} tours en ordre mélangé. Médiane ; IC à 95 % par bootstrap apparié "
         f"(10 000 rééchantillonnages).\n{ENV}",
         ha="center", va="bottom", fontsize=8.5, color="#555555")
fig.tight_layout(rect=[0, 0.07, 1, 0.94])
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{BASE}.{ext}", dpi=300, bbox_inches="tight")
print("écrit :", BASE + ".png / .pdf / .svg")
for (lab, _, _), (g, lo, hi) in zip(rows, res):
    print(f"{lab.replace(chr(10), ' '):45} {g:+6.1f} %  [{lo:+.1f} ; {hi:+.1f}]")
