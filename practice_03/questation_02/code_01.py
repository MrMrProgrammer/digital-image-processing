import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image_01.png", cv2.IMREAD_GRAYSCALE)

median_3 = cv2.medianBlur(img, 3)
median_5 = cv2.medianBlur(img, 5)
median_9 = cv2.medianBlur(img, 9)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Noisy Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(median_3, cmap='gray')
plt.title("Median Filter 3x3")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(median_5, cmap='gray')
plt.title("Median Filter 5x5")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(median_9, cmap='gray')
plt.title("Median Filter 9x9")
plt.axis("off")

plt.tight_layout()
plt.show()
