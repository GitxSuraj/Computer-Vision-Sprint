import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Acceleration Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Transfer Learning on: {device}")

# Image Piepeline - Resizing to 224 X 224 is critical for ResNet.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    # - these specific Normalization constants are required for ResNet
])
print("Loading CIFAR-10 Dataset...")
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Build the Architecture
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze All layers
for param in model.parameters():
    param.requires_grad = False

# Swap the classification head (fc)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)  # CIFAR-10 has 10 classes
model = model.to(device)

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
# Important: Only pass the parameters of the new head to the optimizer!
# Wasting time tracking gradients for frozen layers slows down the code
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# The fast training loop
epochs = 1
print("\nStarting Fine-Tuning...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # We limit the tracking run to 250 batches so that we don't have too wait all day
        if batch_idx>=250:
            break
        images, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch [{batch_idx + 1}/250] | Current Loss: {running_loss / 50:.4f}")
            running_loss = 0.0

print('\nRunning Evaluation on Test Set...')

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        #Evalute over first 50 test batches for speed check
        if batch_idx>=50:
            break
        images, labels = inputs.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Batch [{batch_idx + 1}/100] | Accuracy: {accuracy:.2f}%")