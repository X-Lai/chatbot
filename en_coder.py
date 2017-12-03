from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.autograd as autograd

class encoder_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=1):
        super(encoder_model, self).__init__()
        self.layers = layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input, lengths, hidden):
        embedded = self.embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        output = packed
        output, hidden = self.gru(output, hidden)

        return output, hidden
