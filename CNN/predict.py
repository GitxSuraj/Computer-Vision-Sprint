import torch
import torchvision.transforms as transforms
from PIL import Image
from customCNN import TinyVGG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyVGG(inputShape=3,hiddenUnits=16,outputShape=10).to(device)
ModelPath = "tinyVGG_CIFAR-10.pth"
model.load_state_dict(torch.load(ModelPath, map_location=device))
model.eval()
print("[SUCCESS] Loaded Trained weight into our TinyVGG Model")

preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

try:
    img = Image.open("images/frogcat.jpg")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        prediction_index = torch.argmax(logits, dim=1).item()
        
    print(f"\nPrediction Result: The model thinks this image is a [{class_names[prediction_index]}]!")

except FileNotFoundError:
    raise FileNotFoundError