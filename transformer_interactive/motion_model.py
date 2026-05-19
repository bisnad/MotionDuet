import torch
from torch import nn
import math

config = {
    "mocap_dim": 168,
    "embed_dim": 512,
    "num_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dropout_p": 0.1,
    "seq_length": 64,
    "pos_encoding_max_length": 74, # Provided dynamically by the main script
    "device": "cuda",
    "weights_path": ""
}

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) 
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])

class Transformer(nn.Module):
    def __init__(self, mocap_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, pos_encoding_max_length):
        super().__init__()
        self.embed_dim = embed_dim
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim)
        self.positional_encoder = PositionalEncoding(dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2mocap = nn.Linear(embed_dim, mocap_dim)
        
    def forward(self, mocap_data_src, mocap_data_tgt):
        src_mask = nn.Transformer.generate_square_subsequent_mask(mocap_data_src.shape[1], device=mocap_data_src.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(mocap_data_tgt.shape[1], device=mocap_data_tgt.device)
        
        mocap_src_embedded = self.positional_encoder(self.mocap2embed(mocap_data_src) * math.sqrt(self.embed_dim))
        mocap_tgt_embedded = self.positional_encoder(self.mocap2embed(mocap_data_tgt) * math.sqrt(self.embed_dim))

        encoder_out = self.encoder(mocap_src_embedded, mask=src_mask)
        decoder_out = self.decoder(mocap_tgt_embedded, encoder_out, tgt_mask=tgt_mask)
        return self.embed2mocap(decoder_out)
    
def createModel(config):
    model = Transformer(
        mocap_dim=config["mocap_dim"], 
        embed_dim=config["embed_dim"], 
        num_heads=config["num_heads"], 
        num_encoder_layers=config["num_encoder_layers"], 
        num_decoder_layers=config["num_decoder_layers"], 
        dropout_p=config["dropout_p"], 
        pos_encoding_max_length=config["pos_encoding_max_length"]
    ).to(config["device"])
    
    if config["weights_path"] != "":
        if config["device"] == "cuda":
            model.load_state_dict(torch.load(config["weights_path"]))
        else:
            model.load_state_dict(torch.load(config["weights_path"], map_location=torch.device('cpu')))

    model.eval()
    return model