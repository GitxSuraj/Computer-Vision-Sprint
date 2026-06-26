"""
-> In ML we pass hand chosen features in models like SVM
-> In DL we through raw numbers into an artificail neuron
-> let it learn features itself using weight and bias.

Artifical Neurons{
-> takes input
    -> performs a weighted sum 
        -> adds a baseline adjustmnet
            -> Fires output
    }

    Mathematically: 
        y = w.x+b

        where:
            x - input value e.g. a pixel value
            w (Weight) - the strength of the connection - decide importance of that pixel
            b(bias) - offset value that allows the neuron to shift its activation up or down independent of the input
    
    While handaling an entire image 
        x - a vector of pixels
        w - a matrix of weights
    Operation turns into matrix multiplication 
        z = X.W^T + b
"""
# Activation Function: Non-Linearity
"""
-> only using y = wx+b - doesn't matter how many millions neurons we stack together
-> the entire network remains one gaint linear eqation (i.e. a straight line)
-> But real world images hasve complex non-linear shapes
"""
# Solution?
"""
-> To break the straight lines, we pass the neuron's output through an Activation Function.
-> The undisputed standard in computer vision is 
-> "ReLU(Rectified Linear Unit)".
"""
# Maths beahin ReLU is elegnatly simple: f(x) = max(0,x)
# - if incoming number is '-ve' -> ReLU crushed it to 0 -> the neuran turns off
# - if number is '+ve' -> passes through completely unchanged
# * This simple "on/off" switch allows the network to learn incredibly intricate, non-linear patterns.


import torch
import torch.nn as nn

# 1. SIMULATE AN INPUT IMAGE
# Imagine we have 1 grayscale image of 8x8 pixels. 
# We flatten it into a 1D vector of 64 pixels.
# PyTorch expects float32 for model operations!
inputTensor = torch.randn(size=(1,64), dtype = torch.float32)
print("Input Tensor Shape: ", inputTensor.shape)

# 2. DEFINE A LINEAR LAYER (The Perceptron Bank)
# nn.Linear(in_features, out_features) automatically builds a Weight matrix and a Bias vector.
# in_features = 64 (matching our 64 pixels)
# out_features = 10 (we want this layer to output 10 numbers/features)
linearLayer = nn.Linear(in_features=64, out_features=10)

# 3. DEFINE THE ACTIVATION FUNCTION
relu = nn.ReLU()

# 4. THE FORWARD PASS
# Pass the input data through the linear math equation
outputLinear = linearLayer(inputTensor)
print("Shape after Linear Layer(wx+b):", outputLinear.shape)

# Pass that result through the ReLU non-linear filter
finalOutput = relu(outputLinear)

print("\nRaw Output Values from the layer:\n", finalOutput)