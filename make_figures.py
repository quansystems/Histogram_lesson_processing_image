"""
Sinh hình + số liệu cho báo cáo LaTeX:
- Ảnh hưởng của số lượng bins
- Ảnh hưởng của số mức xám L

Chạy:
    python make_figures.py

Kết quả:
    figures/anh_huong_bins.png
    figures/anh_huong_L.png

Đồng thời in bảng số liệu ra terminal.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. Cấu hình
# ============================================================

IMAGE_NAME = "toi.png"

os.makedirs("figures", exist_ok=True)


# ============================================================
# 2. Đọc ảnh
# ============================================================

def load(path):
    """Đọc ảnh và trả về ảnh RGB hoặc ảnh grayscale."""
    a = np.array(Image.open(path).convert("RGB"))

    # Nếu ảnh thực chất là ảnh xám
    if (a[..., 0] == a[..., 1]).all() and \
       (a[..., 1] == a[..., 2]).all():
        return a[..., 0]

    return a


# ============================================================
# 3. Lấy kênh độ sáng
# ============================================================

def luma(a):
    """
    Ảnh xám:
        sử dụng trực tiếp ảnh.

    Ảnh màu:
        chuyển RGB -> HSV và lấy kênh V.
    """

    if a.ndim == 2:
        return a

    hsv = np.array(Image.fromarray(a).convert("HSV"))
    return hsv[..., 2]


# ============================================================
# 4. Histogram Equalization tự viết
# ============================================================

def he(ch, bins=256, L=256):
    """
    Histogram Equalization tự viết.

    bins:
        Số bin của histogram.

    L:
        Số mức xám của ảnh đầu ra.
    """

    # --------------------------------------------------------
    # Gom 256 mức xám ban đầu vào bins
    # --------------------------------------------------------

    idx = (ch.astype(int) * bins) // 256

    # Histogram
    h = np.bincount(
        idx.ravel(),
        minlength=bins
    )

    # --------------------------------------------------------
    # CDF
    # --------------------------------------------------------

    cdf = np.cumsum(h) / ch.size

    # CDF nhỏ nhất khác 0
    cmin = cdf[h > 0][0]

    # --------------------------------------------------------
    # Hàm biến đổi HE
    # --------------------------------------------------------

    s = np.round(
        (cdf - cmin)
        / (1 - cmin)
        * (L - 1)
    )

    # Ánh xạ pixel
    out = s[idx]

    # Quy về khoảng 0...255 để hiển thị
    out = np.round(
        out * 255 / (L - 1)
    ).astype(np.uint8)

    return out


# ============================================================
# 5. Tính số liệu
# ============================================================

def stats(ch):
    """Tính các thông số thống kê của ảnh."""

    p = np.bincount(
        ch.ravel(),
        minlength=256
    ) / ch.size

    # Entropy
    ent = -(
        p[p > 0] * np.log2(p[p > 0])
    ).sum()

    return (
        ch.mean(),
        ch.std(),
        int(ch.min()),
        int(ch.max()),
        len(np.unique(ch)),
        ent
    )


def row(label, ch):
    """In một dòng số liệu."""

    mean, std, min_val, max_val, levels, entropy = stats(ch)

    print(
        f"{label:15s} "
        f"mean={mean:6.1f} "
        f"std={std:6.1f} "
        f"min={min_val:3d} "
        f"max={max_val:3d} "
        f"levels={levels:3d} "
        f"H={entropy:.2f}"
    )


# ============================================================
# 6. Hiển thị ảnh
# ============================================================

def show(ax, a, title=""):
    """Hiển thị ảnh."""

    ax.imshow(
        a,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    ax.set_title(
        title,
        fontsize=9
    )

    ax.axis("off")


# ============================================================
# 7. Vẽ histogram
# ============================================================

def hist(ax, ch, color="gray", title=""):
    """Vẽ histogram."""

    ax.bar(
        range(256),
        np.bincount(
            ch.ravel(),
            minlength=256
        ),
        width=1,
        color=color
    )

    ax.set_xlim(0, 255)

    ax.set_title(
        title,
        fontsize=9
    )

    ax.set_xlabel("Mức sáng")
    ax.set_ylabel("Số pixel")


# ============================================================
# 8. Đọc ảnh toi.png
# ============================================================

img = load(IMAGE_NAME)

# Lấy kênh V của HSV
src = luma(img)


# ============================================================
# 9. Ảnh hưởng của BINS
# ============================================================

print("\n==============================================")
print("ẢNH HƯỞNG CỦA SỐ LƯỢNG BINS")
print("==============================================")

bins_values = [256, 64, 16, 4]

fig, ax = plt.subplots(
    2,
    4,
    figsize=(15, 6)
)

for j, bins in enumerate(bins_values):

    # HE với bins thay đổi
    result = he(
        src,
        bins=bins,
        L=256
    )

    # Hiển thị ảnh
    show(
        ax[0, j],
        result,
        f"bins = {bins}"
    )

    # Histogram
    hist(
        ax[1, j],
        result,
        "tab:blue",
        f"Histogram (bins = {bins})"
    )

    # In số liệu
    row(
        f"bins = {bins}",
        result
    )

plt.tight_layout()

plt.savefig(
    "figures/anh_huong_bins.png",
    dpi=130
)

plt.close()


# ============================================================
# 10. Ảnh hưởng của L
# ============================================================

print("\n==============================================")
print("ẢNH HƯỞNG CỦA SỐ MỨC XÁM L")
print("==============================================")

L_values = [256, 64, 16, 4]

fig, ax = plt.subplots(
    2,
    4,
    figsize=(15, 6)
)

for j, L in enumerate(L_values):

    # HE với L thay đổi
    result = he(
        src,
        bins=256,
        L=L
    )

    # Hiển thị ảnh
    show(
        ax[0, j],
        result,
        f"L = {L}"
    )

    # Histogram
    hist(
        ax[1, j],
        result,
        "tab:blue",
        f"Histogram (L = {L})"
    )

    # In số liệu
    row(
        f"L = {L}",
        result
    )

plt.tight_layout()

plt.savefig(
    "figures/anh_huong_L.png",
    dpi=130
)

plt.close()


print("\nĐã tạo:")
print("  figures/anh_huong_bins.png")
print("  figures/anh_huong_L.png")