import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

img_dog = Image.open("images/test_dog.jpg") #Testing on a Dog's Image
img_me = Image.open("images/test.jpg") #Testing on my Image

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval() #Loading model with default weight, 
# in evaluation mode (.eval()) so that it doesn't start storing info and learning

firstConvLayer = model.features[0] #It's Conv2d layer, A model has multiple features layer, selcting the index feature for this evalution
preprocess = transforms.Compose([
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
])
inputTensorDog = preprocess(img_dog).unsqueeze(0)
inputTensorMe = preprocess(img_me).unsqueeze(0)
#.unsqueeze(0) adds batch size convers it from (C,W,H) to (Batch Size, C, W, H) with batch size = 1, i.e 1 image

with torch.no_grad():
    featureMapsDog = firstConvLayer(inputTensorDog)
    featureMapsMe = firstConvLayer(inputTensorMe)

print(f"Feature map out put shape\nFor dog: {featureMapsDog.shape}\nFor Human: {featureMapsMe.shape}")

fig, axes = plt.subplots(4,4, figsize=(10,10))
fig.suptitle("Real-Time Visualization: First 16 Convolutional Channels", fontsize=16)

mapsDog = featureMapsDog.squeeze(0).cpu().numpy()
mapsMe = featureMapsMe.squeeze(0).cpu().numpy()
"""
.squeeze(0) - removes that extra dimenstion of batch size, converts back to (C,W,H)
.cpu() - brings the data from gpu to cpu so that matplotlib can read it
.numpy() - converts from tensor to numpy
"""
show = int(input("Press 1 for dog and 0 for human: "))
if show == 1:
    print("\nFor Dog Image\n")
    maps = mapsDog

else:
    print("\nFor Human Image\n")
    maps = mapsMe
for i in range(16):
    row = i//4
    col = i%4
    ax = axes[row, col]
    ax.imshow(maps[i],cmap='viridis')
    ax.axis('off')
    ax.set_title(f"channel{i}")

plt.tight_layout()
print("Displaying Feature maps windows...")
plt.show()
