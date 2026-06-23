# Canny Edge Detection
"""
We use Canny Edge Detection, which is a 4-step mathematical filter applied to an image matrix.

How it works under the hood:
1. Noise Reduction:
-> Images contain digital static.
-> The algorithm applies a Gaussian Blur (smooths the image) so the computer doesn't mistake random pixel noise for a real edge.

2. Finding Gradients: 
-> It calculates where the brightness changes most drastically (just like HOG).

3. Non-Maximum Suppression: 
-> This is a thinning phase. The algorithm looks along the thick blurry edges found in step 2 and deletes any pixels that aren't the absolute sharpest peak of the change, leaving a line that is exactly 1 pixel wide.

4. Hysteresis Thresholding: 
-> It uses two thresholds (a high limit and a low limit):
-> If an edge pixel value is above the high limit, it is kept as a strong edge.
-> If it is below the low limit, it is deleted.
-> If it is in between, it is only kept if it is physically connected to a strong edge. This prevents broken lines.
"""



import cv2
clrImg = cv2.imread("images/test.jpg")
grayImg = cv2.imread("images/test.jpg",0)
# Load our real test image in grayscale mode (0 tells OpenCV to load as grayscale)
# Grayscale is required because edge detection only cares about intensity, not color.
# Well about that grayscale thing let's find out the diffrence 


# Apply Canny Edge Detection
# threshold1 = Low Threshold, threshold2 = High Threshold
edgesGray = cv2.Canny(grayImg, threshold1=40, threshold2=90)
edgesColor = cv2.Canny(clrImg, threshold1=40, threshold2=50)
#play around with threshold

cv2.imwrite("images/cannyEdgesGray.jpg", edgesGray)
cv2.imwrite("images/cannyEdgescolor.jpg", edgesColor)