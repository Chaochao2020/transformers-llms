import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nums_head, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nums_head = nums_head
        self.dropout = nn.Dropout(dropout)

        assert d_model % nums_head == 0

        self.d_k = d_model // nums_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
    
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            attention_score.masked_fill__(mask == 0, -1e-9)
        attention_score = attention_score.softmax(dim = -1)
        if dropout:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score
    
    def forward(self, q, k, v, mask):
        
        query = self.w_q(q)
        value = self.w_v(v)
        key = self.w_k(k)

        query = query.view(query.shape[0], query.shape[1], self.nums_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.nums_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.nums_head, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).view(x.shape[0], -1, self.nums_head * self.d_k)

        return self.w_o(x)
    
    
