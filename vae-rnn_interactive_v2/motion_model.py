import torch
from torch import nn
from collections import OrderedDict
import math

"""
model architecture
"""

config = {
    "seq_length": 64,
    "pose_dim": 308,
    "latent_dim": 32,
    "ae_rnn_layer_count": 2,
    "ae_rnn_layer_size": 256,
    "ae_rnn_bidirectional": True,
    "ae_dense_layer_sizes": [ 512 ],
    "device": "cuda",
    "encoder_weights_path": "../vae-rnn/results/weights/encoder_weights_epoch_600",
    "decoder1_weights_path": "../vae-rnn/results/weights/decoder1_weights_epoch_600",
    "decoder2_weights_path": "../vae-rnn/results/weights/decoder2_weights_epoch_600"
    }


class Encoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, rnn_bidirectional, dense_layer_sizes):
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.rnn_bidirectional = rnn_bidirectional
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True, bidirectional=self.rnn_bidirectional)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_input_dim = self.rnn_layer_size if self.rnn_bidirectional == False else self.rnn_layer_size  * 2
        
        dense_layers.append(("encoder_dense_0", nn.Linear(dense_input_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create final dense layers
            
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z
        
    def forward(self, x):
        x, (_, _) = self.rnn_layers(x)
        x = x[:, -1, :] # only last time step 
        x = self.dense_layers(x)

        mu = self.fc_mu(x)
        std = self.fc_std(x)

        return mu, std

class Decoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, rnn_bidirectional, dense_layer_sizes):
        super(Decoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.rnn_bidirectional = rnn_bidirectional
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True, bidirectional=self.rnn_bidirectional)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        dense_input_dim = self.rnn_layer_size if self.rnn_bidirectional == False else self.rnn_layer_size  * 2
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(dense_input_dim, self.pose_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):

        # dense layers
        x = self.dense_layers(x)

        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.sequence_length, 1)
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        
        # final time distributed dense layer
        dense_input_dim = self.rnn_layer_size if self.rnn_bidirectional == False else self.rnn_layer_size  * 2

        x_reshaped = x.contiguous().view(-1, dense_input_dim)  # (batch_size * sequence, input_size)
        
        yhat = self.final_layers(x_reshaped)
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)

        return yhat
    
def createModel(config):
    
    encoder = Encoder(config["seq_length"], config["pose_dim"], config["latent_dim"], config["ae_rnn_layer_count"], config["ae_rnn_layer_size"], config["ae_rnn_bidirectional"], config["ae_dense_layer_sizes"]).to(config["device"])

    ae_dense_layer_sizes_reversed = config["ae_dense_layer_sizes"].copy()
    ae_dense_layer_sizes_reversed.reverse()
    
    decoder1 = Decoder(config["seq_length"], config["pose_dim"], config["latent_dim"], config["ae_rnn_layer_count"], config["ae_rnn_layer_size"], config["ae_rnn_bidirectional"], ae_dense_layer_sizes_reversed).to(config["device"])
    decoder2 = Decoder(config["seq_length"], config["pose_dim"], config["latent_dim"], config["ae_rnn_layer_count"], config["ae_rnn_layer_size"], config["ae_rnn_bidirectional"], ae_dense_layer_sizes_reversed).to(config["device"])

    if config["encoder_weights_path"] != "":
    
        if config["device"] == "cuda":
            encoder.load_state_dict(torch.load(config["encoder_weights_path"]))
        else:
            encoder.load_state_dict(torch.load(config["encoder_weights_path"], map_location=torch.device('cpu')))
    
    if config["decoder1_weights_path"] != "":
    
        if config["device"] == "cuda":
            decoder1.load_state_dict(torch.load(config["decoder1_weights_path"]))
        else:
            decoder1.load_state_dict(torch.load(config["decoder1_weights_path"], map_location=torch.device('cpu')))
            
    if config["decoder2_weights_path"] != "":
    
        if config["device"] == "cuda":
            decoder2.load_state_dict(torch.load(config["decoder2_weights_path"]))
        else:
            decoder2.load_state_dict(torch.load(config["decoder2_weights_path"], map_location=torch.device('cpu')))
        
    encoder.eval()
    decoder1.eval()
    decoder2.eval()

    return encoder, decoder1, decoder2
