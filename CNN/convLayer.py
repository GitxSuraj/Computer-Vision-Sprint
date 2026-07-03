# Leaving 1 D vector behind moving to 2 D grids
# The core concept - Teh sliding Filter(Kernel) a 3X3 or 5X5 grid of weights
#Feature map
#Automatic Feature Extraction
#Stride & Padding
import torch
import torch.nn as nn

# -> Simulate batch of RGB Images
# PyTorch Spatial layout rule : (Batch Size, Channels, Height, Width)
# We will simulate 2 color images (3 Channels: Red, Green, Blue) at 32x32 pixels
inputImg = torch.randn(size=(2,3,32,32))
print("\nOriginal", inputImg)

print("\nOriginal Input Shape", inputImg.shape)

# -> Define the Convolutional Layer
# in_channel = 3(R,G,B); out_channel = 16 to map 16 distinct features
# kernel_size = 3(3x3 matrix)
# stride = 1 (slide across 1 pixels at a time); padding = 1(Add a 1 pixel zero-border around the edge to maintain 32x32 size)
convLayer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)


# -> Execute the forward pass
outputFeatures = convLayer(inputImg)
print("\nShape Transformation")
print(outputFeatures)
print("\nOutput Features Map Shape:",outputFeatures.shape)