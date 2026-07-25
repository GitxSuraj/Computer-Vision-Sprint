# So far we have built prediction model wit accuracy of 65%
# Now instead building models from scratcha dn and forcing it to learn small things over hours of training
# We Use transfer learning
# Core Intuition:
"""
To learn motercycle - You don't reinvent the wheel - you tranfer your existing knowledge of riding bicycle
Neural network does exact same thing - Research lab spends millions training massive models (Like ResNET or VGG)
on millions of high res images (the ImageNet Data Set ) - they are already masters at sptting edges, textures, shapes 
and complex ligting conditions.
"""

# Every deep vision network is split in two structural halves:
"""
1. The Feauture Extractor (BackBone) - Scans the image for universal visual shapes - these features are identical weather you're looking at a medical scan or a sport's car.
2. The Classifier Head: Final Layer that maps those shapes to specific labels
"""
# With transfer learning 
# -> we download world class backbone 
# -> freeze it's weights so they don't change
# -> tear-off it's original 1,000-class classification head
# -> Wire up our own custom head to solve our specific problem

import torch
import torch.nn as nn
import torchvision.models as models

# 1. Download a pre-trained world-class backbone - ResNet18 - incredibly efficient - Highly powerful residual network
print("Pre Trained ResNet18 backbone")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 2. Freeze the Feature Extractor - We tell PyTorch to stop tracking gradients for these layers
for param in model.parameters():
    param.requires_grad = False # Locks the weights permanently so they don't change during training.


# 3. Inspect & Replace the Classifier Head 
# in ResNet - Final Layer is Named 'fc' (Fully Connected)  
print(f"\nOriginal ResNet Classifier Head: {model.fc}")

in_features_count = model.fc.in_features # Grab the number of incoming features entering the final layer

model.fc = nn.Linear(in_features=in_features_count, out_features=10) # Swap out original head with brand new, untrained linear layer
# target out_features = 10 (matching our 10 CIFAR-10 classes!)

print(f"Modified Custom Classifier Head: {model.fc}")

# 4. Verification Pass
# -> New Head automatically has "requires_grad = True" by default
print("\nVerification: Checking trainable layers...")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"-> Layer to be trained: {name}")