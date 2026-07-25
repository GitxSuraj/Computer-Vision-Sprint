import torch
import torch.nn as nn
import torchvision.models as models
# Pre-trained ResNet18 model - loaded
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers - Initially
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head (fc) with a new one for CIFAR-10
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)  # CIFAR-10 has 10 classes

# Selectively unfreeze the last convolutional black - Layer 4
# ResNet has 4 main blocks (layer1, layer2, layer3, layer4)
# Unfreeze the last block (layer4) - let's the high level shape detectors adapt to our new task!
for param in model.layer4.parameters():
    param.requires_grad = True

#Verify which parameters will be updated
print("Parametres set for Training\n")
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable",name)