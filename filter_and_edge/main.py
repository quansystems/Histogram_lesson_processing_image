"""Script demo xử lý ảnh bằng các bộ lọc nhiễu và phát hiện biên.

File này đọc các ảnh mẫu, áp dụng các thuật toán lọc nhiễu, phát hiện cạnh, rồi lưu
kết quả ra nhiều hình ảnh minh họa trong thư mục figures/.
"""

import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filters import median_filter, gaussian_filter, psnr
from edge import sobel, laplace, LAPLACE4, LAPLACE8

# Đường dẫn lưu ảnh mẫu; đây là nơi chứa các tập tin ảnh cần load.
D = "/mnt/user-data/uploads/"
load = lambda n: np.array(Image.open(D + n).convert("L"))
clean, gauss, sp = (
    load("pure_cameraman.png"),
    load("cameraman_gaussian_noise.png"),
    load("cameraman_salt_pepper_noise.png"),
)


def show(imgs, titles, name, cols):
    """Hiển thị nhiều ảnh trong một lưới và lưu ra file PNG."""
    rows = (len(imgs) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for a in np.ravel(ax):
        a.axis("off")
    for a, im, t in zip(np.ravel(ax), imgs, titles):
        a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.set_title(t, fontsize=9)
    plt.tight_layout()
    plt.savefig("../figures/" + name, dpi=130)
    plt.close()


# 1. Lọc nhiễu: so sánh ảnh gốc với ảnh sau khi áp dụng bộ lọc median/Gaussian.
med = median_filter(sp, 3)
gau = gaussian_filter(gauss, 5, 1.0)
print("PSNR muối tiêu: %.2f -> Median: %.2f" % (psnr(clean, sp), psnr(clean, med)))
print("PSNR Gauss:     %.2f -> Gaussian: %.2f" % (psnr(clean, gauss), psnr(clean, gau)))
show(
    [sp, med, gauss, gau],
    ["Nhiễu muối tiêu", "Median 3x3", "Nhiễu Gauss", "Gaussian 5x5, sigma=1"],
    "loc.png",
    4,
)

# 2. Tách biên: so sánh ảnh gốc với các kết quả phát hiện cạnh bằng Sobel và Laplace.
show(
    [clean, sobel(clean, 2), sobel(clean, 1), laplace(clean, LAPLACE4), laplace(clean, LAPLACE8)],
    ["Ảnh gốc", "Sobel (w=2)", "Prewitt (w=1)", "Laplace 4 láng giềng", "Laplace 8 láng giềng"],
    "bien.png",
    5,
)

# 3. Ảnh hưởng của nhiễu và lọc nhiễu lên biên: xem sự thay đổi cạnh trước/sau khi làm sạch ảnh.
show(
    [
        gauss, sobel(gauss), laplace(gauss), gau, sobel(gau), laplace(gau),
        sp, sobel(sp), laplace(sp), med, sobel(med), laplace(med),
    ],
    [
        "Nhiễu Gauss", "Sobel", "Laplace", "Sau Gaussian", "Sobel", "Laplace",
        "Nhiễu muối tiêu", "Sobel", "Laplace", "Sau Median", "Sobel", "Laplace",
    ],
    "nhieu_bien.png",
    6,
)
