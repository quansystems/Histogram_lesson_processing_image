"""Module xử lý phát hiện biên trong ảnh bằng các bộ lọc Sobel và Laplace.

Các hàm ở đây giúp tính gradient của ảnh và chuẩn hóa giá trị về phạm vi [0, 255]
để thuận tiện hiển thị hoặc so sánh.
"""

import numpy as np
from filters import conv


def norm(a):
    """Chuẩn hóa mảng số về mức xám 8-bit trong khoảng [0, 255].

    Bước này lấy giá trị tuyệt đối của ảnh để tránh âm, sau đó scale theo giá trị lớn
    nhất của ảnh để duy trì độ tương phản và ép về kiểu uint8.
    """
    a = np.abs(a)
    return (a / a.max() * 255).astype(np.uint8)


def sobel(img, w=2):
    """Tính biên ảnh bằng bộ lọc Sobel.

    w=2: Sobel chuẩn (trọng số khác nhau)
    w=1: Prewitt (trọng số đều)
    """
    # Hệ số nhân cho đạo hàm theo trục x; ma trận y là chuyển vị của x.
    kx = np.array([[-1, 0, 1], [-w, 0, w], [-1, 0, 1]], float)
    gx, gy = conv(img, kx), conv(img, kx.T)
    # magnitude = sqrt(gx^2 + gy^2): độ lớn của gradient tại mỗi pixel.
    return norm(np.hypot(gx, gy))


# Ma trận Laplace với 4 hàng xóm và 8 hàng xóm gần nhất.
LAPLACE4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
LAPLACE8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], float)


def laplace(img, mask=LAPLACE4):
    """Tính biên ảnh theo phép toán Laplace.

    Hàm này dùng tích chập với kernel Laplace để phát hiện các điểm thay đổi mạnh
    về cường độ, thường biểu diễn các đường viền.
    """
    return norm(conv(img, mask))
