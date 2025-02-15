# import torch
# from models.encoder import EncoderCNN
# from models.decoder import DecoderRNN
# from PIL import Image
# import torchvision.transforms as transforms

# def test():
#     # Load trained models
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     encoder = EncoderCNN(512).to(device)
#     decoder = DecoderRNN(512, 512, 5000, 1).to(device)

#     encoder.load_state_dict(torch.load("models/encoder.pth"))
#     decoder.load_state_dict(torch.load("models/decoder.pth"))

#     encoder.eval()
#     decoder.eval()

#     # Load and preprocess an example image
#     # image = Image.open("data/test.jpg")
#     image = torch.randn(1, 3, 224, 224).to(device)  # Random image tensor with shape (1, 3, 224, 224)
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     image = transform(image).unsqueeze(0).to(device)

#     # Generate caption
#     with torch.no_grad():
#         features = encoder(image)
#         caption = decoder.sample(features)  # You need to implement the `sample` method in the decoder
#         print("Generated Caption:", caption)




import torch
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from PIL import Image
import torchvision.transforms as transforms

import torch
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from PIL import Image
import torchvision.transforms as transforms

def test():
    # Load trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderCNN(512).to(device)
    decoder = DecoderRNN(512, 512, 5000, 1).to(device)

    encoder.load_state_dict(torch.load("models/encoder.pth"))
    decoder.load_state_dict(torch.load("models/decoder.pth"))

    encoder.eval()
    decoder.eval()

    # Load and preprocess an example image
    image = Image.open("data/test.jpg")  # Replace with an actual image path
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)

    # Extract features using the encoder
    with torch.no_grad():
        features = encoder(image)  # Shape should be (1, 512)
        print(f"features shape from encoder: {features.shape}")  # Debugging: Check encoder output shape
        
        # Generate caption using the sample method
        caption_ids = sample(decoder, features, device)
        print("Generated Caption IDs:", caption_ids)

# Sample method definition
def sample(decoder, features, device, max_length=20):
    """Generate a caption from image features."""
    sampled_ids = []

    # Ensure inputs have shape (batch_size, 1, 512)
    inputs = features.unsqueeze(1)  # Correct shape: (batch_size, 1, 512)
    print(f"Initial inputs shape: {inputs.shape}")  # Debugging: Check initial input shape

    for _ in range(max_length):
        # Forward pass through the LSTM
        hiddens, _ = decoder.lstm(inputs)  # Ensure input size matches input_size=512
        outputs = decoder.fc(hiddens.squeeze(1))  # Shape: (batch_size, vocab_size)
        _, predicted = outputs.max(1)  # Get the index of the max log-probability
        sampled_ids.append(predicted.item())
        
        # Update inputs with the predicted token's embedding
        inputs = decoder.embedding(predicted).unsqueeze(1)  # Shape: (batch_size, 1, embed_size)
        print(f"Updated inputs shape: {inputs.shape}")  # Debugging: Check input size for each iteration

        if predicted.item() == 2:  # Assuming 2 is <EOS> token
            break

    return sampled_ids
