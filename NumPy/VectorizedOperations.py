import numpy as np
#generating a 10 X 10 gray image array
gryImg= np.random.randint(0,256,size=(10,10), dtype = np.uint8)
#brightning imgae by adding 20 to every pixel
brightImg = gryImg + 20
#Image Normalization - Used in DL
#convert image data 0-255 down to 0.0 - 1.0 float numbers
normImg = gryImg/255.0 

print(f"Normalized Image \n{normImg}\nNormalized Data type -  {normImg.dtype}")