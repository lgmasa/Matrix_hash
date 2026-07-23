# ブロック数ごとの「反転ビット数の分布(二項分布)」を 2x2 グリッドで表示(文字大きめ)
# 使い方: python3 plot_blocks.py 1 2 4 8
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# ===== 全体のフォントサイズを大きめに設定 =====
plt.rcParams.update({
    "font.size": 14,          # 基本サイズ
    "axes.titlesize": 16,     # 各パネルのタイトル
    "axes.labelsize": 15,     # 軸ラベル
    "xtick.labelsize": 13,    # 目盛り
    "ytick.labelsize": 13,
    "legend.fontsize": 13,    # 凡例
    "figure.titlesize": 18,   # 全体タイトル
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
ncols = 2 if n > 1 else 1
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.8 * nrows))
axes = np.array(axes).reshape(-1)

samples = read_meta(blocks[0]).get("samples", "?")

for idx, b in enumerate(blocks):
    H = np.loadtxt(f"hist_b{b}.csv", delimiter=",", skiprows=1)
    meta = read_meta(b)
    avg = meta.get("avg_ratio")
    mdev = meta.get("max_dev")
    out_bits = len(H) - 1

    ax = axes[idx]
    ax.bar(H[:, 0], H[:, 1], width=1.0, color="#3a76c0")
    ax.axvline(out_bits / 2, color="r", ls="--", lw=2, label="ideal n/2")
    ax.set_xlim(0, out_bits)
    ax.set_title(f"Flipped-bit distribution  ({b} block{'s' if b>1 else ''})", pad=8)
    ax.set_xlabel("flipped output bits"); ax.set_ylabel("count")
    ax.legend(loc="upper right")
    note = f"avg ratio = {avg}" if avg else ""
    if mdev is not None:
        note += (("\n" if note else "") + f"max|P-0.5| = {mdev}")
    if note:
        ax.text(0.03, 0.95, note, transform=ax.transAxes, va="top", fontsize=13,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

for k in range(len(blocks), len(axes)):
    axes[k].axis("off")

fig.suptitle(f"MATRIX avalanche test   |   samples = {samples} per block,   "
             f"block counts = {blocks},   output = 256 bit,   rounds = 10")
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("avalanche_blocks.png", dpi=140, bbox_inches="tight")
print("wrote avalanche_blocks.png")
for b in blocks:
    m = read_meta(b)
    print(f"  blocks={b}: avg_ratio={m.get('avg_ratio')}, max_dev={m.get('max_dev')}")