import numpy as np
#dummy color image 3x3 pixels , 3 coloe channel
# np.random.randit(low,high,size)
image = np.random.randint(0,256,size=(3,3,3),dtype = np.uint8)
#np.unit8 - unsigned 8 bit integers stores integer from 0 to 255 - standard memory type 
#float 64 will waste very much RAM
print(image) #image array
print(image.shape) #(height, width, channels)

