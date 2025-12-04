import cv2
import numpy as np


def apply_contrast_stretch(img, a, ya, b, yb):
    img = img.astype(np.float32)

    L = 255.0
    out = np.zeros_like(img)

    m1 = ya / max(a, 1)
    out[img <= a] = m1 * img[img <= a]

    m2 = (yb - ya) / max(b - a, 1)
    out[(img > a) & (img <= b)] = ya + m2 * (img[(img > a) & (img <= b)] - a)

    m3 = (L - yb) / max(L - b, 1)
    out[img > b] = yb + m3 * (img[img > b] - b)

    return np.clip(out, 0, 255).astype(np.uint8)


img = cv2.imread("image.png", 0)

cv2.namedWindow("Contrast Stretching")

cv2.createTrackbar("a", "Contrast Stretching", 10, 255, lambda x: None)
cv2.createTrackbar("ya", "Contrast Stretching", 10, 255, lambda x: None)
cv2.createTrackbar("b", "Contrast Stretching", 100, 255, lambda x: None)
cv2.createTrackbar("yb", "Contrast Stretching", 150, 255, lambda x: None)

while True:
    a = cv2.getTrackbarPos("a", "Contrast Stretching")
    ya = cv2.getTrackbarPos("ya", "Contrast Stretching")
    b = cv2.getTrackbarPos("b", "Contrast Stretching")
    yb = cv2.getTrackbarPos("yb", "Contrast Stretching")

    if a >= b:
        b = a + 1

    result = apply_contrast_stretch(img, a, ya, b, yb)

    cv2.imshow("Contrast Stretching", result)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
