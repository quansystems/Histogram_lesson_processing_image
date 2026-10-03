

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATHS = ["anh_mau_toi.jpg", "sang.png", "toi.png"]


# ---------------------------------------------------------------
# 1. Tính histogram (tự viết, không dùng np.histogram)
# ---------------------------------------------------------------
def compute_histogram(channel):
    """Trả về mảng 256 phần tử: số pixel ứng với mỗi mức 0..255."""
    hist = np.zeros(256, dtype=np.int64)
    for level in range(256):
        hist[level] = np.count_nonzero(channel == level)
    return hist


# ---------------------------------------------------------------
# 2. Histogram Equalization (tự viết)
# ---------------------------------------------------------------
def histogram_equalization(channel):
    """
    Cân bằng histogram cho một kênh uint8 (ảnh xám hoặc kênh V).
    Các bước:
      1. Tính histogram h(k)
      2. Xác suất p(k) = h(k) / N
      3. Hàm phân phối tích lũy CDF(k) = sum_{j<=k} p(j)
      4. Chuẩn hóa: s(k) = round( (CDF(k) - CDF_min) / (1 - CDF_min) * 255 )
      5. Ánh xạ mỗi pixel: new = s(old)
    """
    hist = compute_histogram(channel)
    n_pixels = channel.size

    cdf = np.cumsum(hist) / n_pixels
    cdf_min = cdf[hist > 0][0]  # CDF tại mức nhỏ nhất thực sự xuất hiện

    if cdf_min >= 1.0:  # chỉ có 1 mức duy nhất -> không cần cân bằng
        return channel.copy()

    lut = np.round((cdf - cdf_min) / (1.0 - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[channel]  # áp dụng bảng tra (look-up table)


def equalize_image(img_array):
    """
    - Ảnh xám (2D): HE trực tiếp.
    - Ảnh màu: chuyển RGB -> HSV, chỉ cân bằng kênh V (độ sáng),
      giữ nguyên H (màu) và S (độ đậm) rồi ghép lại thành RGB.
    """
    if img_array.ndim == 2:
        return histogram_equalization(img_array)

    hsv = np.array(Image.fromarray(img_array).convert("HSV"))
    hsv[..., 2] = histogram_equalization(hsv[..., 2])  # kênh V
    return np.array(Image.fromarray(hsv, mode="HSV").convert("RGB"))


# ---------------------------------------------------------------
# 3. Hàm hỗ trợ
# ---------------------------------------------------------------
def load_image(path):
    """Đọc ảnh; nếu 3 kênh giống nhau (ảnh xám lưu dạng RGB) thì đổi về 2D."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 3 and np.array_equal(arr[..., 0], arr[..., 1]) \
            and np.array_equal(arr[..., 1], arr[..., 2]):
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def get_V(arr):
    """Lấy kênh độ sáng để vẽ histogram: ảnh xám -> chính nó, ảnh màu -> kênh V."""
    if arr.ndim == 2:
        return arr
    return np.array(Image.fromarray(arr).convert("HSV"))[..., 2]


def show_original_histograms(images):
    """Hiển thị ảnh gốc + histogram của từng bức ảnh."""
    fig, axes = plt.subplots(2, len(images), figsize=(5 * len(images), 7))
    for i, (name, arr) in enumerate(images.items()):
        axes[0, i].imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=255)
        axes[0, i].set_title(f"Ảnh gốc: {name}")
        axes[0, i].axis("off")

        label = "Mức xám" if arr.ndim == 2 else "Giá trị V (độ sáng)"
        axes[1, i].bar(range(256), compute_histogram(get_V(arr)), width=1.0, color="gray")
        axes[1, i].set_title("Histogram gốc")
        axes[1, i].set_xlabel(label)
        axes[1, i].set_ylabel("Số pixel")
        axes[1, i].set_xlim(0, 255)
    fig.suptitle("HISTOGRAM CỦA TỪNG BỨC ẢNH", fontsize=14, fontweight="bold")
    plt.tight_layout()


def show_equalization_result(name, original, equalized):
    """Hiển thị: ảnh gốc | ảnh sau HE ; histogram gốc | histogram mới."""
    cmap = "gray" if original.ndim == 2 else None
    label = "Mức xám" if original.ndim == 2 else "Giá trị V (độ sáng)"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].imshow(original, cmap=cmap, vmin=0, vmax=255)
    axes[0, 0].set_title("Ảnh gốc")
    axes[0, 1].imshow(equalized, cmap=cmap, vmin=0, vmax=255)
    axes[0, 1].set_title("Ảnh sau Histogram Equalization")
    for ax in axes[0]:
        ax.axis("off")

    axes[1, 0].bar(range(256), compute_histogram(get_V(original)), width=1.0, color="gray")
    axes[1, 0].set_title("Histogram gốc")
    axes[1, 1].bar(range(256), compute_histogram(get_V(equalized)), width=1.0, color="tab:blue")
    axes[1, 1].set_title("Histogram sau HE")
    for ax in axes[1]:
        ax.set_xlim(0, 255)
        ax.set_xlabel(label)
        ax.set_ylabel("Số pixel")

    fig.suptitle(f"Kết quả HE (HSV) - {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()


# ---------------------------------------------------------------
# 4. Chương trình chính
# ---------------------------------------------------------------
def main():
    images = {path: load_image(path) for path in IMAGE_PATHS}

    # (a) Histogram của từng ảnh gốc
    show_original_histograms(images)

    # (b) + (c) Dùng HE tự viết, hiển thị ảnh kết quả và histogram mới
    for name, arr in images.items():
        result = equalize_image(arr)
        show_equalization_result(name, arr, result)
        out_name = "HE_HSV_" + name.rsplit(".", 1)[0] + ".png"
        Image.fromarray(result).save(out_name)
        print(f"Đã lưu: {out_name}")

    plt.show()


if __name__ == "__main__":
    main()
