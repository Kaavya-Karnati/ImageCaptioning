import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze all ResNet layers
        for param in resnet.parameters():
            param.requires_grad = False
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)  # Reduce to embed_size (512)
        self.relu = nn.ReLU()

    def forward(self, images):
        features = self.resnet(images).squeeze()  # Shape: (batch_size, 2048)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 2048)
        features = self.fc(features)  # Shape: (batch_size, embed_size)
        features = self.relu(features)
        return features
