# PyTorch Tensor - Similar to NumPy multi dimensional array
"""
While NumPy - works on CPU, PyTorch can be shifted to GPU, Also uses automatic derivations (autograd)
"""
import torch
import numpy as np
vector = torch.tensor([1.0,2.0,3.0]) # Creating Tensors - it can be created directly by python list
# Can Instantly convert NumPy Array into a PyTorch Tensor
npArray = np.array([[1,2], [3,4]],dtype= np.int32)
tensorFromNp = torch.from_numpy(npArray)

print(f"Vector: {vector}\nTensor From NumPy:\n{tensorFromNp}")

#Inspecting Tensor Properties - Every Tensor has a shape, data type, and a device location (CPU/GPU)
x = torch.rand(size=(3,4)) #Generates random floats b/w 0.0 & 0.1
print(f"\nThe Tensor:\n{x}\n\nTensor Attributes:\nShape: {x.shape}\nData Tpe: {x.dtype}\nDevice: {x.device}")

# DEVICE-AGNOSTIC CODING - Crucial for proudction
# - In proudction, code must run seemlessly - GPU presnet or not
# - torch.cuda.is_available() checks if NVIDIA graphics card with CUDA is installed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nSelected computing Device: ", device)

#Shifting the tensor from CPU Ram to selected hardware memory - (GPU VRAM or CPU)
xMoved = x.to(device)
print("New Device location of x:", xMoved.device)
