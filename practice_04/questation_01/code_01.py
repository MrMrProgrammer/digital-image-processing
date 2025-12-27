import cv2
import numpy as np

# ================== Global variables ==================
notch_points = []
radius = 12

# ================== Mouse callback ==================
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        notch_points.append((x, y))
        print(f"Selected point: ({x}, {y})")

# ================== Load image ==================
img = cv2.imread("image_03.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not found!")

rows, cols = img.shape

# ================== FFT ==================
fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

# ================== Magnitude Spectrum ==================
magnitude = np.log(np.abs(fft_shift) + 1)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

# ================== Mask ==================
mask = np.ones((rows, cols), np.uint8)

# ================== UI ==================
cv2.namedWindow("Fourier Spectrum")
cv2.setMouseCallback("Fourier Spectrum", select_point)

while True:
    display = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    for (x, y) in notch_points:
        cv2.circle(display, (x, y), radius, (0, 0, 255), 1)
        cv2.circle(mask, (x, y), radius, 0, -1)

        # symmetric notch
        sx = cols - x
        sy = rows - y
        cv2.circle(display, (sx, sy), radius, (0, 0, 255), 1)
        cv2.circle(mask, (sx, sy), radius, 0, -1)

    cv2.imshow("Fourier Spectrum", display)
    key = cv2.waitKey(1)

    if key == 13:  # Enter → apply filter
        break
    elif key == 27:  # Esc → exit
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Fourier Spectrum")

# ================== Apply Notch Filter ==================
filtered_fft = fft_shift * mask

# ================== Inverse FFT ==================
ifft_shift = np.fft.ifftshift(filtered_fft)
img_filtered = np.fft.ifft2(ifft_shift)
img_filtered = np.abs(img_filtered)

# ================== Normalize ==================
img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
img_filtered = img_filtered.astype(np.uint8)

# ================== Show Results ==================
cv2.imshow("Original Image", img)
cv2.imshow("Denoised Image", img_filtered)
cv2.imshow("Notch Mask", mask * 255)

cv2.waitKey(0)
cv2.destroyAllWindows()
