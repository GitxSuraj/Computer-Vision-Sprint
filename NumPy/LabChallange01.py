"""
Write a quick script that uses these concepts.
 Open your IDE or Notebook and write code that does the following:
 Create a 3D NumPy array of shape (10, 15, 3) filled with random integers between 0 and 255 (a dummy color image).
 "Crop" the image down to just the upper-left $3 \times 3$ pixels (keep all 3 color channels).
 Switch that cropped image's layout from standard format (HWC) to PyTorch format (CHW).
 
 Run that in your environment. 
 Did the shapes output exactly what you expected, or did you hit a dimensional mismatch error?
"""
import numpy as np
clrImg = np.random.randint(0,256,size=(10,15,3), dtype=np.uint8)
crpImg=clrImg[0:-3,0:-3]
print("Standard fromat Image Shape",clrImg.shape)
print("Cropped Image Shape",crpImg.shape)

transposedImg = np.transpose(crpImg,(2,0,1))

print("PyTorch format Image Shape",transposedImg.shape)