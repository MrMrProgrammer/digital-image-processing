import cv2
import matplotlib.pyplot as plt


img = cv2.imread("image.jpg")
if img is None:
    raise ValueError("Image not found. Make sure image.jpg exists.")

b, g, r = cv2.split(img)
channels = [b, g, r]
channel_names = ["Blue", "Green", "Red"]

plt.figure(figsize=(15, 20))

plot_index = 1

for ch_idx, ch in enumerate(channels):
    for bit in range(8):
        mask = 1 << bit
        bit_plane = (ch & mask)
        bit_plane = bit_plane * 255 // mask  

        plt.subplot(3, 8, plot_index)
        plt.imshow(bit_plane, cmap='gray')
        plt.title(f"{channel_names[ch_idx]} - Bit {bit}")
        plt.axis('off')

        plot_index += 1

plt.tight_layout()
plt.show()
