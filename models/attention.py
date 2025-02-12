import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=512, attention_dim=256):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att = self.full_att(torch.relu(self.encoder_att(encoder_out) + self.decoder_att(decoder_hidden).unsqueeze(1)))
        alpha = self.softmax(att)  # Attention weights
        attention_weighted_encoding = (encoder_out * alpha).sum(dim=1)
        return attention_weighted_encoding, alpha
