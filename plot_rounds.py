# ブロック数ごとの「反転ビット数の分布(二項分布)」を縦1列で表示(論文用)
# 使い方: python3 plot_blocks.py 1 2 4 8
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ===== 論文用のフォントサイズ =====
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

blocks = [int(b) for b in sys.argv[1:]] or [1, 2, 4, 8]

def read_meta(b):
    m = {}
    try:
        for line in open(f"meta_b{b}.txt"):
            k, v = line.strip().split("=")
            m[k] = v
    except FileNotFoundError:
        pass
    return m

n = len(blocks)
# 1段組みの幅(約3.4インチ)に合わせ、1パネルあたり高さ1.5インチ
fig, axes = plt.subplots(n, 1, figsize=(3.4, 1.5 * n), sharex=True)
axes = np.array(axes).reshape(-1)

for idx, b in enumerate(blocks):
    H = np.loadtxt(f"hist_b{b}.csv", delimiter=",", skiprows=1)
    k, cnt = H[:, 0], H[:, 1]
    out_bits = len(H) - 1
    total = cnt.sum()
    freq = cnt / total

    mean = (k * cnt).sum() / total

    ax = axes[idx]
    ax.bar(k, freq, width=1.0, color="#3a76c0", alpha=0.7, label="Measured")
    ax.plot(k, binom.pmf(k, out_bits, 0.5), "r-", lw=1.2,
            label=f"$B({out_bits},\\,1/2)$")

    sd_th = np.sqrt(out_bits) / 2
    ax.set_xlim(out_bits / 2 - 5 * sd_th, out_bits / 2 + 5 * sd_th)
    ax.set_ylabel("Rel. freq.")
    ax.text(0.03, 0.93, f"{b} block{'s' if b > 1 else ''}",
            transform=ax.transAxes, va="top", fontweight="bold")
    if idx == 0:
        ax.legend(loc="center right", framealpha=0.9)

    print(f"blocks={b}: mean={mean:.3f} (theory {out_bits/2:.1f}), ")

axes[-1].set_xlabel("Number of flipped output bits")
fig.tight_layout(h_pad=0.4)
fig.savefig("avalanche_blocks.pdf", bbox_inches="tight")
print("wrote avalanche_blocks.pdf")