import torch
import math

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(
            num_embeddings = self.vocab_size, 
            embedding_dim= self.d_model,
            padding_idx=-100
        )
        
    def positional_encoding(self, num_words):
        position_vector = torch.zeros(num_words, self.d_model)
        
        for pos in range(num_words):
            i = torch.arange(self.d_model)
            
            i_even = i[0::2]
            i_odd = i[1::2]
            
            position_vector[pos, 0::2] = torch.sin(pos/(10000**(2*i_even/self.d_model)))
            position_vector[pos, 1::2] = torch.cos(pos/(10000**(2*i_odd/self.d_model)))
        
        return position_vector
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, 0) # adding batch of 1
        
        batch_size = x.size(0)
        seq_num = x.size(1)
        
        output = torch.zeros(batch_size, seq_num, self.d_model)
        for batch in range(batch_size):
            output[batch,:,:] = self.embedding(x[batch,:]) + self.positional_encoding(seq_num)
        # x = self.positional_encoding(seq_num)
        return output


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, d_h: int, mask: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_h = d_h
        self.mask = mask
        
        self.WK = torch.nn.Linear(self.d_model, self.d_h, bias=False)
        self.WQ = torch.nn.Linear(self.d_model, self.d_h, bias=False)
        self.WV = torch.nn.Linear(self.d_model, self.d_h, bias=False)
        self.softmax = torch.nn.Softmax(dim=1) # assumes batch first
        
        
    def forward(self, key, query, value):
        key = self.WK(key)
        query = self.WQ(query)
        value = self.WV(value)
        if self.mask:
            q_k = query @ torch.transpose(key, 1, 2)
            # create the mask
            mask = torch.ones(q_k.size(1), q_k.size(1)) # assumes batch first
            rows, cols = torch.triu_indices(mask.size(0), mask.size(1), offset=1)
            mask[rows, cols] = -1*torch.inf
            x = self.softmax((q_k + mask)/math.sqrt(float(self.d_model)))
            
        else:
            x = self.softmax((query @ torch.transpose(key, 1, 2))/math.sqrt(float(self.d_model)))
        
        x = x @ value
        return x
        
        
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, d_model: int, mask : bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        d_h = self.d_model // self.num_heads
        
        self.self_attention_blocks = torch.nn.ModuleList(
            [SelfAttention(
                self.d_model, d_h, mask
            ) for _ in range(self.num_heads)]
        )
        
        self.WO = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        
    def forward(self, key, query, value):
        x = [
                attention(key, query, value) for attention in self.self_attention_blocks
            ]
        x = torch.concat((x), 2)
        x = self.WO(x)
        return x
    
    
# ad-hoc model for attention is all you need architecture
class Feedforward(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.layer1 = torch.nn.Linear(self.d_model, 2048)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(2048, self.d_model)
        
    def forward(self, x):
        x = self.layer2(self.relu(self.layer1(x)))
        return x   
    

class Encoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, mask: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask
        # self.num_blocks = num_blocks
        
        self.multihead = MultiHeadAttention(
            num_heads= self.num_heads,
            d_model = self.d_model,
            mask=self.mask
        )
        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        
        self.feedforward = Feedforward(self.d_model)
        
    def forward(self, key, query, value):
        x_multihead = self.multihead(
            key, query, value
        )
        x_norm = self.layer_norm(key + x_multihead)
        
        x_feedforward = self.layer_norm(x_norm)
        
        x = self.layer_norm(x_feedforward + x_norm)
        
        return x
    
    

class Decoder(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multihead_mask = MultiHeadAttention(
            num_heads= self.num_heads,
            d_model = self.d_model,
            mask=True
        )
        self.multihead_cross = MultiHeadAttention(
            num_heads=self.num_heads,
            d_model = self.d_model,
            mask=False
        )
        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.feedforward = Feedforward(self.d_model)
        
    def forward(self,
                key, query, value,
                key_enc, value_enc):
        # first stage
        x_masked_multihead = self.multihead_mask(
            key, query, value
        )
        x_norm1 = self.layer_norm(key + x_masked_multihead)
        
        # second stage cross attention
        x_cross_attention = self.multihead_cross(
            key_enc, x_norm1, value_enc
        )

        x_norm2 = self.layer_norm(x_norm1 + x_cross_attention)
        
        # feedforward
        x_feedforward = self.feedforward(x_norm2)
        
        x = self.layer_norm(x_norm2 + x_feedforward)
        
        return x
    
    
class OutputBlock(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.layer1 = torch.nn.Linear(self.d_model, vocab_size)
        self.softmax = torch.nn.Softmax(dim=2) # assumes batch first
        
    def forward(self, x):
        output = self.softmax(self.layer1(x))
        return output
    
    
class Transformer(torch.nn.Module):
    def __init__(self, num_blocks: int, d_model: int,
                num_heads: int, vocab_size_input: int,
                vocab_size_output: int):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.vocab_size_input = vocab_size_input
        self.vocab_size_output = vocab_size_output
        
        self.input_embedding = EmbeddingLayer(
            vocab_size=self.vocab_size_input,
            d_model=self.d_model
        )
        
        self.output_embedding = EmbeddingLayer(
            vocab_size=self.vocab_size_output,
            d_model=self.d_model
        )
        
        self.encoder_blocks = torch.nn.ModuleList([
            Encoder(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mask=False
            ) for _ in range(self.num_blocks)
        ])
        
        self.decoder_blocks = torch.nn.ModuleList([
            Decoder(
                d_model=self.d_model,
                num_heads=self.num_heads
            )
        ])
        
        self.output = OutputBlock(
            d_model=self.d_model,
            vocab_size=self.vocab_size_output
        )
        
        
    def forward(self, x_tokens, y_tokens):
        x = self.input_embedding(x_tokens)
        y = self.output_embedding(y_tokens)
        
        # encoder
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x,x,x)
            
        x_enc = torch.clone(x)
        
        # decoder
        for i in range(self.num_blocks):
            y = self.decoder_blocks[i](
                key=y, 
                query=y,
                value=y,
                key_enc = x_enc,
                value_enc=x_enc
            )
        
        # output layer
        output = self.output(y)
        
        return output
        
        