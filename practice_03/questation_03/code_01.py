import cv2
import numpy as np
import matplotlib.pyplot as plt


def homomorphic_filter(img, gamma_l, gamma_h, d0, c=1):
    img = img.astype(np.float32)
    img_log = np.log1p(img)

    F = np.fft.fft2(img_log)
    F_shift = np.fft.fftshift(F)

    rows, cols = img.shape
    u, v = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    D_uv = np.sqrt((u - rows / 2) ** 2 + (v - cols / 2) ** 2)

    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D_uv ** 2) / (d0 ** 2))) + gamma_l
    G = H * F_shift
    img_ifft = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    img_exp = np.expm1(img_ifft)
    img_out = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)

    return img_out.astype(np.uint8)

img1 = cv2.imread("./image_01.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./image_02.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found.")

params_1 = (0.4, 2.5, 50)
params_2 = (0.6, 3.5, 100)

res1_p1 = homomorphic_filter(img1, *params_1)
res1_p2 = homomorphic_filter(img1, *params_2)

res2_p1 = homomorphic_filter(img2, *params_1)
res2_p2 = homomorphic_filter(img2, *params_2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 01 - Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(res1_p1, cmap='gray')
plt.title("γL=0.4, γH=2.5, D0=50")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(res1_p2, cmap='gray')
plt.title("γL=0.6, γH=3.5, D0=100")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img2, cmap='gray')
plt.title("Image 02 - Original")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(res2_p1, cmap='gray')
plt.title("γL=0.4, γH=2.5, D0=50")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(res2_p2, cmap='gray')
plt.title("γL=0.6, γH=3.5, D0=100")
plt.axis('off')

plt.tight_layout()
plt.show()
