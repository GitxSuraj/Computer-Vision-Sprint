import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

img_dog = Image.open("images/test_dog.jpg")
img_me = Image.open("images/test.jpg")

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval()

firstConvLayer = model.features[0]
preprocess = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
])
inputTensorDog = preprocess(img_dog).unsqueeze(0)
inputTensorMe = preprocess(img_me).unsqueeze(0)

with torch.no_grad():
    featureMapsDog = firstConvLayer(inputTensorDog)
    featureMapsMe = firstConvLayer(inputTensorMe)

print(f"Feature map out put shape\nFor dog: {featureMapsDog.shape}\nFor Human: {featureMapsMe.shape}")

fig, axes = plt.subplots(4,4, figsize=(10,10))
fig.suptitle("Real-Time Visualization: First 16 Convolutional Channels", fontsize=16)

mapsDog = featureMapsDog.squeeze(0).cpu().numpy()
mapsMe = featureMapsMe.squeeze(0).cpu().numpy()
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
