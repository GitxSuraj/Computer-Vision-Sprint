import torch 
import torch.nn as nn
feature_map = torch.randn(size=(2,16,32,32))
print("Original Feature Map Shape:", feature_map.shape)
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
compressed_output = pool_layer(feature_map)
print("\n--- After Max Pooling ---")
print("Compressed Output Shape:   ", compressed_output.shape)