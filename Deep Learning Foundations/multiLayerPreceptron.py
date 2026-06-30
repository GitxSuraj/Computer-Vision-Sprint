# The Deep Architecture: Input, Hidden, and Output layers
"""
- Input Layer: Takes raw Pixel Values
- Hidden Layer: -Intermideiate layer - Network extracts features here
                - first hidden layer - raw edges
                - second hidden layer combines those edges into shapes
- Output Layer: Final Layer that outputs the model's final decision ro class scores
"""
# A Golden Rule of Neural Network Stacking : outFeatures of one layer must match exactly with the inFeatures of very next layer.
# If they don't match the matrix multiplication breaks & PyTorch throws a dimensional mismatch error.

# Production Code: Building a MLP using nn.Sequential
# -> PyTorch give an elegant wrapper - nn.Squential to chain layers cleanly - data automatically flows from one to next.

import torch
import torch.nn as nn
# Demo batch of img - 4pics, with 64 flattened pixels - processing in batches makes training stable
inputBatch = torch.randn(size=(4,64), dtype=torch.float32)
print("\nInput Batch Shape: ", inputBatch.shape)

#Deep multi layered perceptron - data flows from top to bottom through this container
mlpModel = nn.Sequential(
    #First layer
    nn.Linear(in_features=64, out_features=32),
    nn.ReLU(),
    #Second layer
    nn.Linear(in_features=32, out_features=16),
    # Notice: in_features matches the previous out_features (32)!
    nn.ReLU(),

    #third layer - output layer
    nn.Linear(in_features=16,out_features=10)
    # Let's say we are classifying digits from 0 to 9, so we need 10 outputs.
)

# Run the forward pass, - PAss all 4 images simultaneously through the entire network
predictions = mlpModel(inputBatch)
print(f"""
        {inputBatch}\n
        Model Execution Complete\n
        Final Output Shape (Batch Size, Classes): {predictions.shape}\n\n
        Raw Prediction Matrix for 4 images: \n{predictions}
""")