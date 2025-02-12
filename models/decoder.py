import torch
import torch.nn as nn
from models.attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size, hidden_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        h, _ = self.lstm(embeddings)
        context, _ = self.attention(features, h)
        outputs = self.fc(context)
        return outputs
