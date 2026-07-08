import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self,inputShape: int, hiddenUnits: int, outputShape: int):
        super().__init__()

        #Block 1 - extracts simple features - edges, lines
        self.convBlock1=nn.Sequential(
            nn.Conv2d(in_channels=inputShape, out_channels=hiddenUnits,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #Block 2 - combines simple features into shape
        self.convBlock2=nn.Sequential(
            nn.Conv2d(in_channels=hiddenUnits, out_channels=hiddenUnits*2, kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        #Block 3 - the classification head- decision maker
        self.classifier = nn.Sequential(
            nn.Flatten(), #2d Feature maps -> 1d vector
            nn.Linear(in_features=(hiddenUnits*2)*8*8, out_features=outputShape)
        )

    def forward(self,x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.classifier(x)
        return x
    
# Verifcation Block
if __name__=="__main__":
    model = TinyVGG(inputShape=3,hiddenUnits=16,outputShape=10)
    print("\nModel Architecture Blueprint Loaded\n",model)

    dummyImg = torch.randn(size=(1,3,32,32))

    with torch.no_grad():
        predictions = model(dummyImg)
    print("\nSuccessful Verification Pass")
    print("Final output shape ", predictions.shape)