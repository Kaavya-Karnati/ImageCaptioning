import torch
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from PIL import Image
import torchvision.transforms as transforms

def test():
    # Load trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderCNN(256).to(device)
    decoder = DecoderRNN(256, 512, 5000, 1).to(device)

    encoder.load_state_dict(torch.load("models/encoder.pth"))
    decoder.load_state_dict(torch.load("models/decoder.pth"))

    encoder.eval()
    decoder.eval()

    # Load and preprocess an example image
    image = Image.open("data/test.jpg")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        features = encoder(image)
        caption = decoder.sample(features)  # You need to implement the `sample` method in the decoder
        print("Generated Caption:", caption)
