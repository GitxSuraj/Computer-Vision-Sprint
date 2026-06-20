import numpy as np
# Create a dummy 4x4 color image (Shape: 4, 4, 3)
color_img = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)

# Calculate the average color across the ENTIRE image (collapses all axes)
print("Global Mean:", np.mean(color_img)) 

# Calculate the average for each individual color channel (R, G, and B)
# We average out the Height (axis 0) and Width (axis 1)
channel_means = np.mean(color_img, axis=(0, 1))
print("Per-Channel Means (R, G, B):", channel_means) # Outputs 3 numbers