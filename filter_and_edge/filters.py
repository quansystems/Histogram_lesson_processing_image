"""Các phép biến đổi và lọc hình ảnh cơ bản dùng trong xử lý ảnh.

File này bao gồm:
- đệm biên để xử lý cạnh ảnh khi tích chập,
- tích chập 2D bằng vòng lặp,
- bộ lọc median và Gaussian,
- chỉ số PSNR để so sánh chất lượng ảnh.
"""

import numpy as np


def pad(img, r):
    """Đệm biên bằng cách lặp lại giá trị pixel ở mép ảnh."""
    return np.pad(img, r, mode="edge")


def conv(img, kernel):
    """Thực hiện tích chập 2D đơn giản bằng vòng lặp Python."""
    k = kernel.shape[0]
    r = k // 2
    # Tạo ảnh có đệm để giữ nguyên kích thước đầu ra sau khi quét kernel
    p = pad(img.astype(float), r)
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.sum(p[i:i + k, j:j + k] * kernel)
    return out


def median_filter(img, k=3):
    """Lọc nhiễu bằng giá trị median trong cửa sổ kxk."""
    r = k // 2
    p = pad(img, r)
    out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.median(p[i:i + k, j:j + k])
    return out.astype(np.uint8)


def gaussian_kernel(k=5, sigma=1.0):
    """Tạo ma trận Gaussian 2D theo kích thước và độ lệch chuẩn cho trước."""
    ax = np.arange(k) - k // 2
    x, y = np.meshgrid(ax, ax)
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_filter(img, k=5, sigma=1.0):
    """Lọc nhiễu Gaussian bằng cách tích chập với kernel Gaussian."""
    return np.clip(conv(img, gaussian_kernel(k, sigma)), 0, 255).astype(np.uint8)


def psnr(a, b):
    """Tính chỉ số PSNR giữa hai ảnh, đơn vị tính bằng dB."""
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * np.log10(255**2 / mse)
