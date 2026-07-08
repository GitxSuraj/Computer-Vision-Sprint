# Max Pooling - used to kind of compress the image, while keeping most of the data intact
# it reduces image resultion, say fromm 32 X 32 to 16 X 16, 
# to reduce the calulation, so that the load from increasing 
# features does't overload the gpu and fry it

# -> the most common is 2X2 pooling with stride of 2



import torch 
import torch.nn as nn

#high depth image -demo
feature_map = torch.randn(size=(2,16,32,32))
print("Original Feature Map Shape:", feature_map.shape)

#kernel size = 2 -> 2X2 window, stride = 2 -> Hops 2 pixels
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
compressed_output = pool_layer(feature_map)
print("\n--- After Max Pooling ---")
print("Compressed Output Shape:   ", compressed_output.shape)
#models reduces total number of pixels by 75% without losing vital structal edges