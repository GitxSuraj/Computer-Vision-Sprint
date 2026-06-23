#Now we will work on actual Visiual data
#OpenCV (cv2) - work house of actual vision data ingestion
# A quirk of OpenCV - While in genral we use RGB (Red, Green, Blue) - OpenCV reads image in BGR (Blue, Green, Red), hence swapping Blue & Red
# This happens due to how early cam devs structured their data
# If we feed direct image data from OpenCV to PyTorch - all the images would be inverted


# Handaling Standard image - color conversion - Resizing
import cv2 
import numpy as np
img = cv2.imread('images/test.jpg')
print(f"Loaded image Shape: {img.shape} - HWC format shape")
print(f"Loaded image data type: {img.dtype}") #unit8 by default
#fixing color space from BGR to RGB
rgbImg=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(rgbImg.shape)
#Resize the image - crucial coz neural network - requires fixed dimensions 
#OpenCV - resize funtion expects (Width, Height) - reverse of NumPy.
targetWidth = 224
targetHeight = 224
resizedImg = cv2.resize(rgbImg,(targetWidth,targetHeight))
print("Resized Image Shape: ",resizedImg.shape)
