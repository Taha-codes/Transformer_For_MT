import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Modlule):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, d_model) 

        # An embedding in pytorch is just a mapping between a number and a vector of dimension 512.
        # The embedding vector can be of any dimension, not necessarily 512.
        # I chose 512 just because I am following the implementation details of the paper.

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        # create a matrix of shape (sed_len, d_model)
        pe = torch.zeros(seq_len, d_model) 

        # create a vector that will represent the position of the word inside a sentence (seq_len)
        # Generate position indices
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #Tensor of shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Since we will have a batch of sentences
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.apha = nn.Parameter(torch.ones(1)) # We multiply alpha with the standardized x 
        self.bias = nn.Parameter(torch.zeros(1)) # After that we add this bias 

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std = x.std(dim = -1, keepdim = True)
            return self.alpha * (x - mean) / (std * self.eps) + self.bias