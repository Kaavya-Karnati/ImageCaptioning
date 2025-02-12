import torch
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from PIL import Image
import torchvision.transforms as transforms

# Load trained models
encoder = EncoderCNN(256)
encoder.load_state_dict(torch.load("models/encoder.pth"))
encoder.eval()

decoder = DecoderRNN(256, 512, 5000, 1)
decoder.load_state_dict(torch.load("models/decoder.pth"))
decoder.eval()

# Load and preprocess image
image = Image.open("data/test.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)

# Generate caption
features = encoder(image)
caption = decoder.sample(features)
print("Generated Caption:", caption)
