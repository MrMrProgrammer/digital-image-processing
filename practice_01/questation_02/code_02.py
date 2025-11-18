import cv2
import numpy as np

img = cv2.imread("image.png")
img_display = img.copy()

def remove_mole_inpaint(event, x, y, flags, param):
    global img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        radius = 10

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)

        img_display = cv2.inpaint(img_display, mask, 3, cv2.INPAINT_TELEA)

cv2.namedWindow("Remove Moles")
cv2.setMouseCallback("Remove Moles", remove_mole_inpaint)

while True:
    cv2.imshow("Remove Moles", img_display)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord('s'):
        cv2.imwrite("output.jpg", img_display)
        print("Saved as output.jpg")

cv2.destroyAllWindows()
