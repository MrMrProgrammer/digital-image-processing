import matplotlib
matplotlib.use('TkAgg') 

import cv2
import numpy as np

# ---------- Load Image ----------
img = cv2.imread("image.jpg")      # آدرس تصویر
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # تبدیل برای نمایش درست در matplotlib

# ---------- Gamma Correction ----------
def gamma_correction(image, gamma):
    # ساخت جدول LUT برای سرعت بیشتر
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ---------- Log Transformation ----------
def log_transform(image, c=1):
    img_float = image.astype(np.float32)
    log_img = c * np.log1p(img_float)
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    return log_img.astype(np.uint8)

# ---------- Apply Transformations ----------
gammas = [0.4, 0.8, 1.2, 2.2]

gamma_images = []
for g in gammas:
    gamma_images.append(gamma_correction(img, g))

log_image = log_transform(img, c=20)

# ---------- Show Results ----------
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

for i, (g, g_img) in enumerate(zip(gammas, gamma_images)):
    plt.subplot(3, 2, i + 2)
    plt.imshow(g_img)
    plt.title(f"Gamma = {g}")
    plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(log_image)
plt.title("Log Transformation")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- Optional: Save outputs ----------
# cv2.imwrite("gamma_0.4.png", cv2.cvtColor(gamma_images[0], cv2.COLOR_RGB2BGR))
# cv2.imwrite("log_transform.png", cv2.cvtColor(log_image, cv2.COLOR_RGB2BGR))
