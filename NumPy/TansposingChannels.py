# OpenCV & Standarad image are in shape -  HWC(Height, Width, Channel)
# but in Deep learning - PyTorch expects shape strictly in CWH ( Channel, Width,Height)
# Feeding HWC will crash the model
# So we transpose the shape from HWC to make it CWH
import numpy as np
standardImg = np.random.randint(0,256,size=(480,640,3), dtype=np.uint8) #Starndard 480 X 640 pixels 3 channel color image array
pyTorchReadyImg = np.transpose(standardImg,(2,0,1)) # Rearrange dimensions from axis (0, 1, 2) to (2, 0, 1)
print(f"Original Shape: {standardImg.shape}\nPyTorch Ready Shape: { pyTorchReadyImg.shape}")
