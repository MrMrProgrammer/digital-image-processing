import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)


def gamma_correction(image, gamma):
    img_float = image / 255.0
    img_gamma = np.power(img_float, gamma)
    img_gamma = np.uint8(img_gamma * 255)
    return img_gamma


def log_transformation(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(1 + image.astype(np.float64))
    log_image = np.uint8(log_image)
    return log_image


gamma_values = [0.5, 1.0, 2.0]

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

# نمایش Gamma
for i, g in enumerate(gamma_values):
    gamma_img = gamma_correction(img, g)
    plt.subplot(2, 3, i+2)
    plt.imshow(gamma_img, cmap='gray')
    plt.title(f"Gamma = {g}")
    plt.axis('off')

log_img = log_transformation(img)
plt.subplot(2, 3, 5)
plt.imshow(log_img, cmap='gray')
plt.title("Log Transformation")
plt.axis('off')

plt.tight_layout()
plt.show()
