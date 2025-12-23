import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("image_01.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("image_02.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found.")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(img1.ravel(), bins=256, range=[0, 256])
plt.title("Histogram of image_01.png")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")

plt.subplot(1, 2, 2)
plt.hist(img2.ravel(), bins=256, range=[0, 256])
plt.title("Histogram of image_02.png")
plt.xlabel("Intensity")
plt.ylabel("Number of pixels")

plt.tight_layout()
plt.show()
