import torch
import torchvision.transforms as transforms
from PIL import Image
from CNN.customCNN import TinyVGG

rawImg = Image.open("images/test_dog.jpg")

preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
inputTensor = preprocess(rawImg).unsqueeze(0)

model = TinyVGG(inputShape=3,hiddenUnits=16,outputShape=10)
model.eval()

with torch.no_grad():
    rawScores = model(inputTensor)

print(f"\nExecution Successful\nInput Tensor\n{inputTensor}\nShape: {inputTensor.shape}\nOutput Array\n{rawScores.shape}\nThe 10 raw prediction Scores (Logits):\n{rawScores}")