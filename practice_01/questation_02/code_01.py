import cv2
import numpy as np

# ---------------------------
# خواندن تصویر
# ---------------------------
img = cv2.imread("image.png")
img_display = img.copy()

# ---------------------------
# پنجره کنترل
# ---------------------------
def nothing(x):
    pass

cv2.namedWindow("Control Panel")
cv2.createTrackbar("Radius", "Control Panel", 20, 100, nothing)
cv2.createTrackbar("Red",    "Control Panel", 255, 255, nothing)
cv2.createTrackbar("Green",  "Control Panel", 255, 255, nothing)
cv2.createTrackbar("Blue",   "Control Panel", 255, 255, nothing)

# ---------------------------
# تابع کلیک ماوس
# ---------------------------
def draw_circle(event, x, y, flags, param):
    global img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        radius = cv2.getTrackbarPos("Radius", "Control Panel")

        # رنگ انتخاب شده در بخش (الف)
        b = cv2.getTrackbarPos("Blue", "Control Panel")
        g = cv2.getTrackbarPos("Green", "Control Panel")
        r = cv2.getTrackbarPos("Red", "Control Panel")

        color = (b, g, r)

        # رسم دایره روی تصویر
        cv2.circle(img_display, (x, y), radius, color, -1)

# ثبت تابع کلیک
cv2.namedWindow("Main")
cv2.setMouseCallback("Main", draw_circle)


# ---------------------------
# حلقه نمایش
# ---------------------------
while True:
    cv2.imshow("Main", img_display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break

cv2.destroyAllWindows()
