# The OpenCV co-ordinate system
"""
Image co-orddinates starts at top-left corner(0,0).
- Going right increases X
- Going left increases Y
"""
# Drawing Bounding Boxes & Text - OpenCV has hightly Optimized built-in function for that
# Modify Image matrices by adding shapes and overlays
'''
-> cv2.recatnagle(img, pt1, pt2, color, thickness)
-> cv2.putText(img, text, org, font, font-scale, color, thickness)
''' 
# Note: OpenCV uses BGR - color tuple to pass expects - (Bleu, Green, Red)

# Simulating a model finding an objext -> draw box around it -> write a label -> Saving it on disk
 
import cv2
#Loading image - have to keep it in bgr format coz we have to perform operations in OpenCV only
img = cv2.imread("test.jpg")
if img is None:
    raise FileNotFoundError("Make Sure 'image.jpg' is in your folder!")
# defining a bonding box coordinates - simulating an ai predicting box location
# print(img.shape[0]) - for testing

height = img.shape[0]
width = img.shape[1]
startPoint = (50,50) #(x1,y1) - top left
endPoint = (width-50,height-50) #(x2,y2) - bottom right

# print(startPoint, endPoint) - for testing

#drawing a green bounding box - 3 pixels thick
colorBox = (0,255,0)
thcikness = height//100
cv2.rectangle(img, startPoint, endPoint, colorBox,thcikness)

text = "Target Object: 98%"
org = (50+thcikness, 100+thcikness+height//200+height//400) #-> Slightly below the green box
font = cv2.FONT_HERSHEY_PLAIN
fontScale = height//400
colorText = (0,0,255) #Red text
cv2.putText(img, text, org, font, fontScale, colorText,thickness=height//200)

#Save the output final annoted text in your disk
cv2.imwrite("OutputImg.jpg", img)



