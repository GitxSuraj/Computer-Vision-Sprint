import torch
import torch.nn as nn
import torchvision.models as models

class BoundingBoxDetector(nn.Module):
    def __init__(self):
        super(BoundingBoxDetector, self).__init__()
        # Pretrained feature extractor
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze backbone parameters 
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Replace the classification head with regression head for 4 coordinates
        # Outputs: [x_min, y_min, x_max, y_max] normalized between 0.0 and 1.0
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid() # Keeps coordinates scaled betwwen 0 & 1
        )
    def forward(self, x):
        return self.backbone(x)
# Quick verification 
if __name__ == "__main__":
    model = BoundingBoxDetector()
    dummy_image = torch.randn(1,3,224,224) # Batch of 1 image
    predicted_box = model(dummy_image)

    print("Predicted Normalized Bounding Box [x_min, y_min, x_max, y_max]:")
    print(predicted_box.detach().numpy()) #.detach(): Strips gradient tracking off the tensor, use .cpu() before converting it to numpy array if running on cuda