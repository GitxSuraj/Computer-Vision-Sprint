import numpy as np
#generating a 10 X 10 gray image array
gryImg= np.random.randint(0,256,size=(10,10), dtype = np.uint8)
#cropping it to 4 X 4 pixels
crpImg= gryImg[3:7,3:7]
print(f"Original image array {gryImg} \n Cropped image array \n{crpImg}")
print(f"Original image Shape {gryImg.shape} \n Cropped image Shape {crpImg.shape}")