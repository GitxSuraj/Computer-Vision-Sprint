import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#Hardware acceleraation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ",device)
#load and transfrom real images
# transform.ToTensor() converts image to float32 and scale pixels to [0.0, 1.0]
transform = transforms.Compose([transforms.ToTensor()])

#Download the training data splits
trainDataset = torchvision.datasets.MNIST(root='./data', train=True,download=True,transform=transform)

#DataLoaders automatically handle shuffling and splitting data into batches
trainLoader = torch.utils.data.DataLoader(trainDataset,batch_size=64, shuffle=True)

print(f"Loaded {len(trainDataset)} training images grouped into batches of 64")

#define the MLP structure
# MNIST images are 28 X 28 pixels. Flattend they become a 1 D vector of 784 numbers.
class DigitMLP(nn.Module):
    def __init__(self):
        super(DigitMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64,out_features=10) #10 outputs from digits 0-9
        )

    def forward(self,x):
        #Flatten the incoming batch of 2d images from (64,1,28,28) down to (64,784)
        x=x.view(x.size(0),-1)
        return self.network(x)
    
# Instantiate model and send it to GPU/CPU
model = DigitMLP().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

# The real taining loop
print("\nStarting Training Pass")
model.train() #put model in training mode

# Loop throught first 200 batches of kmages to see it learn
for batch_idx, (images, labels) in enumerate(trainLoader):
    if batch_idx>=200:
        break
    
    #Move data to the selected hardware
    images,labels=images.to(device), labels.to(device)
    
    # Clear out gradient from previous round
    optimizer.zero_grad()

    # Pass images forward to through the network to get predections
    outputs = model(images)

    # Calculate the loss - How wrong the guesses were
    loss=criterion(outputs, labels)

    # Back propogation (PyTorch calculates all calculus derivatives instantly)
    loss.backward()

    # Update the weights based on gradient calculations
    optimizer.step()

    #Print progress everey 40 batches
    if batch_idx%40 == 0:
        print(f"Batch {batch_idx}/200 | Current Loss: {loss.item(): .4f}")
print("\n Training Segment complete!")

# --- STARTING TESTING PHASE ---
print("\n--- Starting Evaluation on Test Dataset ---")

# 1. LOAD THE SEPARATE TEST DATASET
# Notice train=False, which pulls the 10,000 images reserved strictly for testing
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# 2. INITIALIZE TRACKING VARIABLES
correct = 0
total = 0

# 3. SET THE PYTORCH GATES FOR EVALUATION
model.eval() 

with torch.no_grad(): 
    for images, labels in test_loader:
        # Shift data to the same device as the model (CPU or GPU)
        images, labels = images.to(device), labels.to(device)
        
        # Pass test images forward to get raw scores (logits)
        outputs = model(images)
        
        # torch.max looks at the 10 outputs and finds the highest score.
        # It returns: (highest_value, index_location_of_highest_value)
        # We only care about the index location, which is the predicted digit!
        _, predicted = torch.max(outputs.data, dim=1)
        
        # Accumulate totals
        total += labels.size(0) # Adds batch size (64) to the running total
        correct += (predicted == labels).sum().item() # Counts how many predictions matched actual labels

# 4. CALCULATE FINAL ACCURACY
final_accuracy = (correct / total) * 100
print(f"\nFinal Test Results:")
print(f"Total Test Images Processed: {total}")
print(f"Correctly Classified Digits: {correct}")
print(f"Model Accuracy on Unseen Data: {final_accuracy:.2f}%")