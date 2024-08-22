import torch
import torch.nn as nn
import math

# InputEmbedding Layer
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        # d_model: 词向量维度
        # vocab_size: 词表大小
         super().__init__()
         self.d_model = d_model
         self.vocab_size = vocab_size
         self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
         return self.embedding(x) * math.sqrt(self.d_model)
    


# Positional Encoding
class PositinalEncoding(nn.Module):
     def __init__(self, d_model: int,seq_len: int, dropout: float):
        # seq_len: 句子最大长度
        # dropout: 减少过拟合
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(dropout)

        # Create a matrix with seq_len rows and d_model columns (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # 公式: PE(pos, 2i) = sin(pos / 10000 ** (2i / d_model)) PE(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
        # 这里的 pos 是每个 token 在句子中的位置，而 2i / 2i + 1 则是 d_model 中的奇偶数

        # Create a vectory of shape (seq_len, 1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # for pos in range(seq_len):
        #     for i in range(d_model):
        #         if i % 2 == 0:
        #             pe[pos, i] = torch.sin(pos / (10000 ** (2 * i / d_model)))
        #         else:
        #             pe[pos, i] = torch.cos(pos / (10000 ** (2 * i / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 接下来添加一个 batch 维度，seq_len 只是一个句子的长度
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # 将位置编码（pe）注册为一个缓冲区。
        # 通过将位置编码注册为缓冲区，可以保证其在模型的多次前向传播过程中不会被更新，从而使得模型能够学习到更加稳定的位置信息
        self.register_buffer('pe', pe)

     def forward(self, x):
          x += (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
          return self.dropout(x)
     

# Layer Normalization
class LayerNormalization(nn.Module):
     def __init__(self, eps: float = 10 ** -6 ) -> None:
          super().__init__()
          # eps 是用于防止除以 0 的情况
          self.eps = eps
          self.alpha = nn.Parameter(torch.ones(1))
          self.bias = nn.Parameter(torch.zeros(1))

     def forward(self, x):
          mean = x.mean(dim=-1, keepdim=True)
          std = x.std(dim=-1, keepdim=True)

          return self.alpha * (x- mean) / (std + self.eps) + self.bias


# Feed-Forward Block
class FeedForwardBlock(nn.Module):
     def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
          super().__init__()
          # d_ff (dimension of the feed-forward layer): 这是指内部全连接层（即linear_1）的输出维度，通常比d_model大，目的是增加模型的表达能力。较大的内部维度允许网络学习更复杂的特征表示。
          self.linear_1 = nn.Linear(d_model, d_ff)
          self.dropout = nn.Dropout(dropout)
          self.linear_2 = nn.Linear(d_ff, d_model)

     def forward(self, x):
          # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
          return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
     


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
     def __init__(self, d_model: int, num_heads: int, dropout: float):
          super().__init__()
          self.d_model = d_model
          self.num_heads = num_heads
          self.dropout = nn.Dropout(dropout)

          assert d_model % num_heads == 0, f"Embedding dimension ({d_model}) should be divisible by number of heads ({num_heads})"

          self.d_k = d_model // num_heads
          # W_q (query weight matrix): 这是一个线性层，用于将输入序列中的每个token映射到query向量。
          # W_k (key weight matrix): 这是一个线性层，用于将输入序列中的每个token映射到key向量。
          # W_v (value weight matrix): 这是一个线性层，用于将输入序列中的每个token映射到value向量。
          self.w_q = nn.Linear(d_model, d_model)
          self.w_k = nn.Linear(d_model, d_model)
          self.w_v = nn.Linear(d_model, d_model)

          # W_o (output weight matrix): 这是一个线性层，用于将每个head的输出映射到最终的输出向量。
          self.w_o = nn.Linear(d_model, d_model)
     
     @staticmethod
     def attention(query, key, value, mask, dropout: nn.Dropout):
          d_k = query.shape[-1]
          # Just apply the formula from the paper
          # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
          # @ 表示矩阵乘法
          attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
          if mask is not None:
               # Write a very low value (indicating -inf) to the positions where mask == 0
               attention_scores.masked_fill_(mask == 0, -1e9)
          attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
          if dropout is not None:
               attention_scores = dropout(attention_scores)
          # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
          # return attention scores which can be used for visualization
          return (attention_scores @ value), attention_scores

     def forward(self, q, k, v, mask):
          query = self.w_q(q)
          key = self.w_k(k)
          value = self.w_v(v)

          # (batch, seq_len, d_model) --> (batch, seq_len, num_heads, d_k) --> (batch, num_heads, seq_len, d_k)
          query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
          key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
          value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
          # Calculate attention
          x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

          # Combine all the heads together
          # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
          x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

          # Multiply by Wo
          # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
          return self.w_o(x)

class ResidualConnection(nn.Module):
     def __init__(self, features: int, dropout: float) -> None:
         self.dropout = nn.Dropout(dropout)
         self.norm = LayerNormalization(features)
     
     def forward(self, x, sublayer):
          return x + self.dropout(sublayer(self.norm(x)))
     

class EncoderBlock(nn.Module):
     def __init__(self, feature, attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:

          super().__init__()
          self.attention_block = attention_block
          self.feed_forward_block = feed_forward_block
          self.residual_connections = nn.ModuleList([ResidualConnection(feature, dropout) for _ in range(2)])
     def forward(self, x, src_mask):
          x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
          x = self.residual_connections[1](x, self.feed_forward_block)
          return x

          
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    


    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

          

     


     
     







        



