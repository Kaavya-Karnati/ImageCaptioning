import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.dataset import Flickr8kDataset
from utils.collate_fn import collate_fn
import torchvision.transforms as transforms
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    VOCAB_SIZE = 5000
    NUM_LAYERS = 1
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset and DataLoader
    train_dataset = Flickr8kDataset(root_dir="data/Flickr8k", captions_file="data/captions.csv", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize encoder and decoder
    encoder = EncoderCNN(embed_size=EMBED_SIZE).to(device)
    decoder = DecoderRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS).to(device)

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        total_loss = 0

        for i, (images, captions) in enumerate(train_loader):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            features = encoder(images)                      # (batch_size, embed_size)
            outputs = decoder(features, captions)           # (batch_size, seq_len, vocab_size)

            # Debugging shapes
            print(f"features shape: {features.shape}")
            print(f"outputs shape before flattening: {outputs.shape}")
            print(f"captions shape: {captions.shape}")

            # Flatten outputs and captions for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))    # (batch_size * seq_len, vocab_size)
            captions = captions.view(-1)                    # (batch_size * seq_len)

            # Compute loss
            loss = criterion(outputs, captions)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")

    # Save the models
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(encoder.state_dict(), "models/encoder.pth")
    torch.save(decoder.state_dict(), "models/decoder.pth")
    print("Training completed. Models saved!")

if __name__ == "_main_":
    train()