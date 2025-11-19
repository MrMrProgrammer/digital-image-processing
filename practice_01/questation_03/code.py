import cv2
import numpy as np


img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Image not found! Check the file path.")
    exit()

def gamma_transform(image, gamma):
    normalized = image / 255.0
    gamma_corrected = np.power(normalized, gamma)
    return np.uint8(gamma_corrected * 255)

cv2.namedWindow("Gamma Correction")
cv2.createTrackbar("Gamma x10", "Gamma Correction", 10, 50, lambda x: None)

while True:
    gamma_val = cv2.getTrackbarPos("Gamma x10", "Gamma Correction") / 10
    output = gamma_transform(img, gamma_val)
    display = output.copy()
    cv2.putText(
        display,
        f"Gamma = {gamma_val}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2
    )

    cv2.imshow("Gamma Correction", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
