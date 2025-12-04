import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (BGR)
img = cv2.imread("image.jpg")
if img is None:
    raise ValueError("Image not found. Make sure image.jpg exists.")

# Split into color channels
b, g, r = cv2.split(img)
channels = [b, g, r]
channel_names = ["Blue", "Green", "Red"]

# Create a figure
plt.figure(figsize=(15, 20))

plot_index = 1

for ch_idx, ch in enumerate(channels):
    for bit in range(8):
        # Create bit mask
        mask = 1 << bit
        # Extract bit-plane
        bit_plane = (ch & mask)
        # Scale to 0–255 for display
        bit_plane = bit_plane * 255 // mask  

        # Plot
        plt.subplot(3, 8, plot_index)
        plt.imshow(bit_plane, cmap='gray')
        plt.title(f"{channel_names[ch_idx]} - Bit {bit}")
        plt.axis('off')

        plot_index += 1

plt.tight_layout()
plt.show()
