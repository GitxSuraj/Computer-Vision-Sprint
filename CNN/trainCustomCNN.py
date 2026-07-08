import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from customCNN import TinyVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Executing Pipeline on", device)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

print("Loading CIFAR-10 dataset")
trainSet = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet,batch_size=64,shuffle=True)

testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=False)

model = TinyVGG(inputShape=3,hiddenUnits=16,outputShape=10).to(device)

#Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
print("Initializing training loop")
for epoch in range(epochs):
    model.train()
    runningLoss=0.0

    for batch_idx, (images,labels) in enumerate(trainLoader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        runningLoss+=loss.item()

        if (batch_idx+1)%200==0:
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(trainLoader)}] | Average Loss = {runningLoss/200:.4f}")
            runningLoss = 0.0
print("Training Completed, Evaluating Performance on unseen data")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs= model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct+=(predicted==labels).sum().item()
accuracy = (correct/total)*100
print(f"Final Model Accurracy: {accuracy :.2f}")

#Serialize & Save model weights
ModelPath = "tinyVGG_CIFAR-10.pth"
torch.save(obj=model.state_dict(), f=ModelPath)
print(f"\n[SUCCESS] Model weigths Saved permanently at path: {ModelPath}")