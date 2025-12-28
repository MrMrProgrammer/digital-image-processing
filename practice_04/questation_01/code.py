import cv2
import numpy as np

notch_points = []
radius = 12


def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        notch_points.append((x, y))


img = cv2.imread("image_01.png", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

fft = np.fft.fft2(img)
fft_shift = np.fft.fftshift(fft)

magnitude = np.log(np.abs(fft_shift) + 1)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

cv2.namedWindow("Fourier Spectrum")
cv2.setMouseCallback("Fourier Spectrum", select_point)
cv2.namedWindow("Original Image")
cv2.namedWindow("Filtered Image")

while True:
    mask = np.ones((rows, cols), np.uint8)
    display = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

    for (x, y) in notch_points:
        cv2.circle(display, (x, y), radius, (0, 0, 255), 1)
        cv2.circle(mask, (x, y), radius, 0, -1)
        sx = cols - x
        sy = rows - y
        cv2.circle(display, (sx, sy), radius, (0, 0, 255), 1)
        cv2.circle(mask, (sx, sy), radius, 0, -1)

    filtered_fft = fft_shift * mask

    img_filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fft))
    img_filtered = np.abs(img_filtered)
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    img_filtered = img_filtered.astype(np.uint8)

    cv2.imshow("Fourier Spectrum", display)
    cv2.imshow("Original Image", img)
    cv2.imshow("Filtered Image", img_filtered)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('u'):
        if notch_points:
            removed_point = notch_points.pop()
            print(f"Removed point: {removed_point}")

cv2.destroyAllWindows()
