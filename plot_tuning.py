import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path.home() / "proj"
SERIES = [("GEMM", H / "tune2.csv"), ("2DCONV", H / "2dconv_tuning.csv")]
BASE = H / "tuning_figure"
ORIG = (32, 8)            # taille de bloc d'origine de Polybench
SHOW_ORIGIN = True        # cercle gris sur la configuration d'origine
SHOW_GAIN = True          # gain dans le titre de chaque panneau
C_LINE, C_REG, C_ORIG = "#1F4E79", "#E6550D", "#7F7F7F"

def fr(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (name, path) in zip(axes, SERIES):
    if not path.exists():
        sys.exit(f"fichier introuvable : {path}")
    df = pd.read_csv(path)
    has_reg = "USE_REGISTER" in df.columns
    if not has_reg:
        df["USE_REGISTER"] = 0
    df["threads"] = df.block_size_x * df.block_size_y

    cfgs = (df[["block_size_x", "block_size_y", "threads"]].drop_duplicates()
            .sort_values(["threads", "block_size_x"]).reset_index(drop=True))
    labels = [f"{x}×{y}" for x, y in zip(cfgs.block_size_x, cfgs.block_size_y)]
    pos = np.arange(len(cfgs))
    key = {(x, y): i for i, (x, y) in enumerate(zip(cfgs.block_size_x, cfgs.block_size_y))}

    for reg, color, lab in [(0, C_LINE, "Version initiale"),
                            (1, C_REG, "Accumulation en registre")]:
        sub = df[df.USE_REGISTER == reg].copy()
        if sub.empty:
            continue
        sub["p"] = [key[(x, y)] for x, y in zip(sub.block_size_x, sub.block_size_y)]
        sub = sub.sort_values("p")
        ax.plot(sub.p, sub.time, "-o", color=color, lw=1.6, ms=6, zorder=3,
                label=lab if has_reg else None)

    title = name
    o = df[(df.block_size_x == ORIG[0]) & (df.block_size_y == ORIG[1]) & (df.USE_REGISTER == 0)]
    if len(o):
        t_orig, t_best = float(o.time.iloc[0]), float(df.time.min())
        if SHOW_ORIGIN:
            ax.plot(key[ORIG], t_orig, "o", ms=14, mfc="none", mec=C_ORIG, mew=2.4, zorder=4)
        if SHOW_GAIN:
            title += f" : gain du réglage ×{fr(t_orig / t_best)}"
        msg = f"{name:7} origine {t_orig:.3f} ms | meilleure {t_best:.3f} ms | gain x{t_orig / t_best:.2f}"
        if has_reg:
            t_blk = float(df[df.USE_REGISTER == 0].time.min())
            msg += f" | bloc seul x{t_orig / t_blk:.2f}"
        print(msg)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylim(0, df.time.max() * 1.45)
    ax.set_xlabel("Taille du bloc de threads (X×Y)")
    ax.set_ylabel("Temps du kernel (ms)")
    ax.set_title(title)
    if has_reg:
        ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=2)

fig.suptitle("Réglage avec KernelTuner (CUDA)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf", "svg"):
    fig.savefig(f"{BASE}.{ext}", dpi=300, bbox_inches="tight")
print("écrit :", BASE.with_suffix(".png"))
