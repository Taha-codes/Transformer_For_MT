import torch
import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
        MHA attention layer from the transformer architecture
    """
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__() 
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        #Query, Key, and Value projections
        self.w_q = 