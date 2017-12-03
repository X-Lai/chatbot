from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.autograd as autograd

class decoder_model(nn.Module):
    def __init__(self, hidden_dim, output_dim, layers=1, ):
        super(decoder_model, self).__init__()
        self.layers = layers
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.z = nn.Linear(hidden_dim, output_dim)
        self.softmax=nn.LogSoftmax()

    def forward(self, input, hidden):
        batch_size = input.size(1)
        embedded = self.embedding(input).view(1, batch_size, -1)
        out = embedded
        for i in range(self.layers):
            out, hidden = self.gru(out, hidden)
        output=self.softmax(self.z(out[0]))
        return output, hidden
