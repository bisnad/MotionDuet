import torch
from torch import nn
from collections import OrderedDict
import math

"""
model architecture
"""

config = {
    "seq_length": 64,
    "data_dim": 308,
    "embed_dim": 512,
    "head_count": 8,
    "layer_count": 6,
    "dropout_p": 0.1,
    "device": "cuda",
    "weights_path": "../transformer_encoder/results/weights/transformer_encoder_weights_epoch_100"
    }

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class TransformerEncoder(nn.Module):

    # Constructor
    def __init__(
        self,
        data_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        dropout_p,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.data2embed = nn.Linear(data_dim, embed_dim) # map mocap data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=5000
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.embed2data = nn.Linear(embed_dim, data_dim) # map embedding to mocap data
       
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, data):
        
        #print("forward")
        
        #print("data s ", data.shape)

        data_mask = self.get_tgt_mask(data.shape[1]).to(data.device)
        
        data_embedded = self.data2embed(data) * math.sqrt(self.embed_dim)
        data_embedded = self.positional_encoder(data_embedded)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(data_embedded, mask=data_mask)
        out = self.embed2data(encoder_out)
        
        return out

    
def createModel(config):
    
    transformer = TransformerEncoder(data_dim=config["data_dim"], embed_dim=config["embed_dim"], num_heads=config["head_count"], num_encoder_layers=config["layer_count"], dropout_p=config["dropout"]).to(config["device"])

    if config["weights_path"] != "":
    
        if config["device"] == "cuda":
            transformer.load_state_dict(torch.load(config["weights_path"]))
        else:
            transformer.load_state_dict(torch.load(config["weights_path"], map_location=torch.device('cpu')))
        
    return transformer
