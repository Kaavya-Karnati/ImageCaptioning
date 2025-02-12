import torch
import torch.optim as optim
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

# Define model parameters
embed_size = 256
hidden_size = 512
vocab_size = 5000
num_layers = 1
learning_rate = 0.001

# Initialize models
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
for epoch in range(10):
    for images, captions in dataloader:
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(encoder.state_dict(), "models/encoder.pth")
torch.save(decoder.state_dict(), "models/decoder.pth")
