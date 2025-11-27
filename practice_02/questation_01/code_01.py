import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# 1. Load image
# ============================
img = cv2.imread("image.png")   # آدرس تصویر را بگذارید
if img is None:
    raise ValueError("Image not found!")

# ============================
# 2. Convert image to grayscale
# ============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ============================
# 3. Manual Histogram Equalization
# ============================
def manual_hist_equalization(gray_img):
    # 1. Compute histogram
    hist, bins = np.histogram(gray_img.flatten(), 256, [0,256])
    
    # 2. Compute cumulative distribution function
    cdf = hist.cumsum()
    
    # 3. Normalize CDF into [0,255]
    cdf_masked = np.ma.masked_equal(cdf, 0)  # جلوگیری از تقسیم بر صفر
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    
    # 4. Replace masked values with 0
    lut = np.ma.filled(cdf_normalized, 0).astype('uint8')
    
    # 5. Map each pixel through LUT
    equalized_img = lut[gray_img]
    
    return equalized_img, hist, lut

# اجرای الگوریتم
gray_eq_manual, hist_before, lut = manual_hist_equalization(gray)

# ============================
# 4. Show results
# ============================
plt.figure(figsize=(6,6))
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')

plt.figure(figsize=(6,6))
plt.imshow(gray_eq_manual, cmap='gray')
plt.title("Manual Histogram Equalized Image")
plt.axis('off')

plt.figure(figsize=(8,4))
plt.plot(hist_before)
plt.title("Histogram Before Equalization")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.figure(figsize=(8,4))
plt.plot(lut)
plt.title("CDF → LUT Mapping")
plt.xlabel("Original Intensity")
plt.ylabel("Mapped Intensity")

plt.show()

# ============================
# 5. Save output
# ============================
cv2.imwrite("gray_output.png", gray)
cv2.imwrite("equalized_manual.png", gray_eq_manual)
