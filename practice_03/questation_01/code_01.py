import cv2
import numpy as np

img = cv2.imread('image.jpg', 0)
img = cv2.resize(img, (512, 512))
img_float = np.float32(img)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

def ideal_highpass_filter(shape, D0, alpha):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    H = np.ones((rows, cols), np.float32)

    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            if D <= D0:
                H[u, v] = alpha
    return H

def nothing(x):
    pass

cv2.namedWindow("Result")
cv2.createTrackbar("D0", "Result", 10, 200, nothing)
cv2.createTrackbar("Alpha x100", "Result", 0, 100, nothing)

while True:
    D0 = cv2.getTrackbarPos("D0", "Result")
    alpha = cv2.getTrackbarPos("Alpha x100", "Result") / 100.0

    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    H = ideal_highpass_filter(img.shape, D0, alpha)
    H = cv2.merge([H, H])

    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    sharpened = cv2.normalize(img + img_back, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow("Original", img)
    cv2.imshow("HighPass", cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))
    cv2.imshow("Result", sharpened.astype(np.uint8))

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
