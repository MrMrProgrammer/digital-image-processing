import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("image.jpg")
if img is None:
    raise ValueError("Original image not found.")


binary_img = cv2.imread("replacement.jpg", cv2.IMREAD_GRAYSCALE)
if binary_img is None:
    raise ValueError("Replacement image not found.")


binary_img = cv2.resize(binary_img, (img.shape[1], img.shape[0]))
_, binary_img = cv2.threshold(binary_img, 127, 1, cv2.THRESH_BINARY)


def replace_bit_plane(image, binary, bit):
    channels = cv2.split(image)
    new_channels = []
    for ch in channels:
        new_ch = ch.copy()
        mask = 255 ^ (1 << bit)
        new_ch = new_ch & mask
        new_ch = new_ch | (binary.astype(np.uint8) << bit)
        new_channels.append(new_ch)
    return cv2.merge(new_channels)


bit_planes = [0, 2, 5]
images = [img]

for bit in bit_planes:
    new_img = replace_bit_plane(img, binary_img, bit)
    images.append(new_img)

titles = ["Original", "Bit 1 replaced", "Bit 3 replaced", "Bit 6 replaced"]
plt.figure(figsize=(15,5))

for i, (im, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 4, i+1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
