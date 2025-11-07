import cv2
import numpy as np

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (512, 512))

def gamma_correction(image, gamma):
    img_float = image / 255.0
    img_gamma = np.power(img_float, gamma)
    return np.uint8(np.clip(img_gamma * 255, 0, 255))

def log_transformation(image):
    image_float = image.astype(np.float64)
    if np.max(image_float) == 0:
        return image.copy()
    c = 255 / np.log(1 + np.max(image_float))
    log_image = c * np.log(1 + image_float)
    return np.uint8(np.clip(log_image, 0, 255))

cv2.namedWindow('Live Transform')
cv2.createTrackbar('Gamma', 'Live Transform', 10, 50, lambda x: None)
cv2.createTrackbar('Mode', 'Live Transform', 0, 1, lambda x: None)

while True:
    gamma_val = cv2.getTrackbarPos('Gamma', 'Live Transform') / 10
    mode = cv2.getTrackbarPos('Mode', 'Live Transform')

    if mode == 0:
        gamma_val = max(gamma_val, 0.1)
        output = gamma_correction(img, gamma_val)
    else:
        output = log_transformation(img)

    cv2.imshow('Live Transform', output)

    if cv2.waitKey(50) & 0xFF == 27:
        break

cv2.destroyAllWindows()
