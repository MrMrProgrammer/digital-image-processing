import cv2
import numpy as np

img = cv2.imread('image.png')
img = cv2.resize(img, (800, 500))

cv2.imshow("Original Image", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

red_color = np.array([0, 0, 255])

# =============================================================================

dist_l1 = np.sum(np.abs(img - red_color), axis=2)
threshold_l1 = 150
mask_l1 = dist_l1 < threshold_l1

result_l1 = gray_3ch.copy()
result_l1[mask_l1] = img[mask_l1]

cv2.imshow("Cube Distance (L1)", result_l1)

# =============================================================================

dist_l2 = np.linalg.norm(img - red_color, axis=2)
threshold_l2 = 120
mask_l2 = dist_l2 < threshold_l2

result_l2 = gray_3ch.copy()
result_l2[mask_l2] = img[mask_l2]

cv2.imshow("Euclidean Distance (L2)", result_l2)

# =============================================================================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask_red = mask1 | mask2

result_inrange = gray_3ch.copy()
result_inrange[mask_red > 0] = img[mask_red > 0]

cv2.imshow("inRange (HSV)", result_inrange)

# =============================================================================

cv2.waitKey(0)
cv2.destroyAllWindows()
