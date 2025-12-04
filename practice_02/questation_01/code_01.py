import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_hist_eq(gray_img):
    hist, bins = np.histogram(gray_img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_norm = cdf_norm.astype('uint8')
    equalized = cdf_norm[gray_img]
    return equalized


img1 = cv2.imread('image_01.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
manual_eq1 = manual_hist_eq(gray1)
opencv_eq1 = cv2.equalizeHist(gray1)


img2 = cv2.imread('image_02.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
manual_eq2 = manual_hist_eq(gray2)
opencv_eq2 = cv2.equalizeHist(gray2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(gray1, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Manual EQ")
plt.imshow(manual_eq1, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("OpenCV EQ")
plt.imshow(opencv_eq1, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Original")
plt.imshow(gray2, cmap='gray')

plt.subplot(2, 3, 5)
plt.title("Manual EQ")
plt.imshow(manual_eq2, cmap='gray')

plt.subplot(2, 3, 6)
plt.title("OpenCV EQ")
plt.imshow(opencv_eq2, cmap='gray')

plt.tight_layout()
plt.show()
