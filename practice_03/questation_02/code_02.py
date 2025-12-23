import cv2
import numpy as np
import matplotlib.pyplot as plt


def adaptive_median_filter(img, Smax=7):
    padded_img = np.pad(img, Smax//2, mode='edge')
    output = img.copy()
    rows, cols = img.shape

    for i in range(rows):
        for j in range(cols):
            S = 3
            while True:
                window = padded_img[i:i+S, j:j+S]

                Zmin = np.min(window)
                Zmax = np.max(window)
                Zmed = np.median(window)
                Zxy = img[i, j]

                if Zmed > Zmin and Zmed < Zmax:
                    if Zxy > Zmin and Zxy < Zmax:
                        output[i, j] = Zxy
                    else:
                        output[i, j] = Zmed
                    break
                else:
                    S += 2
                    if S > Smax:
                        output[i, j] = Zmed
                        break
    return output


img = cv2.imread("image_02.png", cv2.IMREAD_GRAYSCALE)

denoised_nlm = cv2.fastNlMeansDenoising(
    img,
    None,
    h=10,
    templateWindowSize=7,
    searchWindowSize=21
)

adaptive_result = adaptive_median_filter(img, Smax=7)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(denoised_nlm, cmap='gray')
plt.title("Median Filter 3x3")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(adaptive_result, cmap='gray')
plt.title("Adaptive Median Filter")
plt.axis("off")

plt.tight_layout()
plt.show()
