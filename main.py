import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from utils.dataset import Flickr8kDataset, collate_fn  # Custom dataset loader
import os

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_SIZE = 5000  # Modify based on actual vocabulary size
NUM_LAYERS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Load dataset
train_dataset = Flickr8kDataset(root_dir="data/Flickr8k/", captions_file="data/captions.txt")
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Initialize model
encoder = EncoderCNN(embed_size=EMBED_SIZE).to(device)
decoder = DecoderRNN(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)
        
        # Forward pass
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Compute loss
        loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Save trained model
if not os.path.exists("models"):
    os.makedirs("models")
torch.save(encoder.state_dict(), "models/encoder.pth")
torch.save(decoder.state_dict(), "models/decoder.pth")

print("Training completed. Models saved!")
