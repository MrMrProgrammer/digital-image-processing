import cv2
import numpy as np
import matplotlib.pyplot as plt


def power_law_transform(img, gamma):
    img = img.astype(np.float32) / 255.0
    img_gamma = np.power(img, gamma)
    img_out = np.uint8(img_gamma * 255)
    return img_out


img1 = cv2.imread("./image_01.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./image_02.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found.")

gamma_1 = 0.5
gamma_2 = 1.5

res1_g1 = power_law_transform(img1, gamma_1)
res1_g2 = power_law_transform(img1, gamma_2)

res2_g1 = power_law_transform(img2, gamma_1)
res2_g2 = power_law_transform(img2, gamma_2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Image 01 - Original")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(res1_g1, cmap='gray')
plt.title("γ = 0.5")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(res1_g2, cmap='gray')
plt.title("γ = 1.5")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img2, cmap='gray')
plt.title("Image 02 - Original")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(res2_g1, cmap='gray')
plt.title("γ = 0.5")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(res2_g2, cmap='gray')
plt.title("γ = 1.5")
plt.axis('off')

plt.tight_layout()
plt.show()
