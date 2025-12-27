import cv2
import numpy as np

img = cv2.imread("image_03.png", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

mask = np.ones((rows, cols), np.uint8)

cx, cy = cols // 2, rows // 2
width = 4
gap = 30  # keep center frequencies

# Vertical line (except center)
mask[:cy-gap, cx-width:cx+width] = 0
mask[cy+gap:, cx-width:cx+width] = 0

# Horizontal line (except center)
mask[cy-width:cy+width, :cx-gap] = 0
mask[cy-width:cy+width, cx+gap:] = 0

filtered_fft = fft_shift * mask

ifft = np.fft.ifftshift(filtered_fft)
img_filtered = np.fft.ifft2(ifft)
img_filtered = np.abs(img_filtered)

img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
img_filtered = img_filtered.astype(np.uint8)

cv2.imshow("Filtered Image", img_filtered)
cv2.imshow("Mask", mask * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()
