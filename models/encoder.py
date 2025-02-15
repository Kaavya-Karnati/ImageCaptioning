import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512):  # Keep embed_size as 512
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 512)  # Ensure it's 512 to match the saved model
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images).squeeze()
        features = self.fc(features)
        features = self.relu(features)
        return features
