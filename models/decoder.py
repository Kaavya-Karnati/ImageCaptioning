# import torch
# import torch.nn as nn
# from models.attention import Attention

# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size=512, hidden_size=512, vocab_size=5000, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)
#         self.attention = Attention(hidden_size, hidden_size, hidden_size)

#     def forward(self, features, captions):
#         embeddings = self.embedding(captions)  # Shape: (batch_size, seq_length, embed_size)
#         h, _ = self.lstm(embeddings)  # Shape: (batch_size, seq_length, hidden_size)
#         context, _ = self.attention(features.unsqueeze(1), h[:, -1, :])  # Use last hidden state for attention
#         outputs = self.fc(context)  # Shape: (batch_size, vocab_size)
#         return outputs


"""
import torch
import torch.nn as nn
from models.attention import Attention

class DecoderRNN(nn.Module):
    def _init_(self, embed_size=512, hidden_size=512, vocab_size=5000, num_layers=1):
        super(DecoderRNN, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size, hidden_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)  # Shape: (batch_size, seq_length, embed_size)
        h, _ = self.lstm(embeddings)  # Shape: (batch_size, seq_length, hidden_size)
        context, _ = self.attention(features.unsqueeze(1), h[:, -1, :])  # Use last hidden state for attention
        outputs = self.fc(context)  # Shape: (batch_size, vocab_size)
        return outputs
"""
# import torch
# import torch.nn as nn
# from models.attention import Attention

# class DecoderRNN(nn.Module):
#     def _init_(self, embed_size=512, hidden_size=512, vocab_size=5000, num_layers=1):
#         super(DecoderRNN, self)._init_()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)
#         self.attention = Attention(hidden_size, hidden_size, hidden_size)

#     def forward(self, features, captions):
#         embeddings = self.embedding(captions)  # Shape: (batch_size, seq_len, embed_size)
#         h, _ = self.lstm(embeddings)  # Shape: (batch_size, seq_len, hidden_size)

#         # Apply attention at each time step
#         outputs = []
#         for t in range(h.size(1)):  # Iterate over each time step
#             context, _ = self.attention(features.unsqueeze(1), h[:, t, :])  # Apply attention at time step t
#             output = self.fc(context)  # Shape: (batch_size, vocab_size)
#             outputs.append(output)

#         # Stack outputs to get shape (batch_size, seq_len, vocab_size)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs  # Shape: (batch_size, seq_len, vocab_size)


import torch
import torch.nn as nn
from models.attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, vocab_size=5000, num_layers=1):
        super(DecoderRNN, self).__init__()  # Corrected __init__ method and super()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size, hidden_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)  # Shape: (batch_size, seq_len, embed_size)
        h, _ = self.lstm(embeddings)  # Shape: (batch_size, seq_len, hidden_size)

        # Apply attention at each time step
        outputs = []
        for t in range(h.size(1)):  # Iterate over each time step
            context, _ = self.attention(features.unsqueeze(1), h[:, t, :].unsqueeze(1))  # Apply attention at time step t
            output = self.fc(context.squeeze(1))  # Shape: (batch_size, vocab_size)
            outputs.append(output)

        # Stack outputs to get shape (batch_size, seq_len, vocab_size)
        outputs = torch.stack(outputs, dim=1)
        return outputs  # Shape: (batch_size, seq_len, vocab_size)
