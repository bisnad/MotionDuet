"""
same as transfomer.py
but creates entire dance sequences for dancer 2 and not only next poses
"""

"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt


# mocap specific imports

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_file_path = "D:/Data/mocap/stocos/Duets/Amsterdam_2024/bvh_50hz"
mocap_files = [ [ "Recording2_JS-001_jason.bvh", "Recording2_JS-001_sherise.bvh" ] ]
mocap_valid_frame_ranges = [ [ [ 490, 30679] ] ]
mocap_fps = 50

joint_loss_weights = [ 
    1.0, # Hips
    1.0, # RightUpLeg
    1.0, # RightLeg
    1.0, # RightFoot
    1.0, # RightToeBase
    1.0, # RightToeBase_Nub
    1.0, # LeftUpLeg
    1.0, # LeftLeg
    1.0, # LeftFoot
    1.0, # LeftToeBase
    1.0, # LeftToeBase_Nub
    1.0, # Spine
    1.0, # Spine1
    1.0, # Spine2
    1.0, # Spine3
    1.0, # LeftShoulder
    1.0, # LeftArm
    1.0, # LeftForeArm
    1.0, # LeftHand
    1.0, # LeftHand_Nub
    1.0, # RightShoulder
    1.0, # RightArm
    1.0, # RightForeArm
    1.0, # RightHand
    1.0, # RightHand_Nub
    1.0, # Neck
    1.0, # Head
    1.0 # Head_Nub
    ]

"""
Model Settings
"""

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 512
transformer_dropout = 0.1   
pos_encoding_max_length = 5000


"""
Training Settings
"""

seq_length = 64

batch_size = 32
test_percentage = 0.1

learning_rate = 1e-4
norm_loss_scale = 0.1
pos_loss_scale = 0.1
rot_loss_scale = 0.9

model_save_interval = 10
save_weights = True
epochs = 200

"""
Mocap Visualisation Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Load Data - Mocap
"""

bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data_dancer1 = []
all_mocap_data_dancer2 = []

for mocap_file_dancer1, mocap_file_dancer2 in mocap_files:
    
    print("process file for dancer 1 ", mocap_file_dancer1)
    
    bvh_data_dancer1 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer1)
    mocap_data_dancer1 = mocap_tools.bvh_to_mocap(bvh_data_dancer1)
    mocap_data_dancer1["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data_dancer1["motion"]["rot_local_euler"], mocap_data_dancer1["rot_sequence"])

    all_mocap_data_dancer1.append(mocap_data_dancer1)

    print("process file for dancer 2 ", mocap_file_dancer2)
    
    bvh_data_dancer2 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer2)
    mocap_data_dancer2 = mocap_tools.bvh_to_mocap(bvh_data_dancer2)
    mocap_data_dancer2["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data_dancer2["motion"]["rot_local_euler"], mocap_data_dancer2["rot_sequence"])

    all_mocap_data_dancer2.append(mocap_data_dancer2)


# retrieve mocap properties (from dancer 1, dancer 2 properties are supposed to be identical)
mocap_data = all_mocap_data_dancer1[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

# calculate normalization values
all_mocap_rot_local = []

for mocap_data_dancer1, mocap_data_dancer2 in zip(all_mocap_data_dancer1, all_mocap_data_dancer2):
    mocap_rot_local_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    mocap_rot_local_dancer2 = mocap_data_dancer2["motion"]["rot_local"]

    all_mocap_rot_local.append(mocap_rot_local_dancer1)
    all_mocap_rot_local.append(mocap_rot_local_dancer2)
    
all_mocap_rot_local = np.concatenate(all_mocap_rot_local, axis=0)
all_mocap_rot_local = np.reshape(all_mocap_rot_local, (-1, pose_dim))

mocap_mean = np.mean(all_mocap_rot_local, axis=0)
mocap_std = np.std(all_mocap_rot_local, axis=0)

"""
Create Dataset
"""

dancer1_data = []
dancer2_data = []

for i in range(len(all_mocap_data_dancer1)):
    
    mocap_data_dancer1 = all_mocap_data_dancer1[i]
    mocap_data_dancer2 = all_mocap_data_dancer2[i]
    
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    pose_sequence_dancer1 = np.reshape(pose_sequence_dancer1, (-1, pose_dim))
    
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    pose_sequence_dancer2 = np.reshape(pose_sequence_dancer2, (-1, pose_dim))

    print("shape ", pose_sequence_dancer1.shape)
    
    valid_frame_ranges = mocap_valid_frame_ranges[i]
    
    for valid_frame_range in valid_frame_ranges:
        
        frame_range_start = valid_frame_range[0]
        frame_range_end = valid_frame_range[1]
        
        print("frame range from ", frame_range_start, " to ", frame_range_end)
        
        for pI in np.arange(frame_range_start, frame_range_end - seq_length - 2):

            sequence_excerpt_dancer1 = pose_sequence_dancer1[pI:pI+seq_length + 1]
            dancer1_data.append(sequence_excerpt_dancer1)
            
            sequence_excerpt_dancer2 = pose_sequence_dancer2[pI:pI+seq_length + 1]
            dancer2_data.append(sequence_excerpt_dancer2)

dancer1_data = np.array(dancer1_data)
dancer2_data = np.array(dancer2_data)

dancer1_data = torch.from_numpy(dancer1_data).to(torch.float32)
dancer2_data = torch.from_numpy(dancer2_data).to(torch.float32)

class DuetDataset(Dataset):
    def __init__(self, dancer1_data, dancer2_data):
        self.dancer1_data = dancer1_data
        self.dancer2_data = dancer2_data
    
    def __len__(self):
        return self.dancer1_data.shape[0]
    
    def __getitem__(self, idx):
        return self.dancer1_data[idx, ...], self.dancer2_data[idx, ...]

full_dataset = DuetDataset(dancer1_data, dancer2_data)

X_item, y_item = full_dataset[0]

print("X_item s ", X_item.shape)
print("y_item s ", y_item.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X_batch, y_batch = next(iter(train_loader))

print("X_batch s ", X_batch.shape)
print("y_batch s ", y_batch.shape)

"""
Create Models - PositionalEncoding
"""

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


"""
Create Models - Transformer
"""

class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        mocap_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim) # map mocap data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2mocap = nn.Linear(embed_dim, mocap_dim) # map embedding to mocap data
        
    def get_src_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.ones(size, size)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
       
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
        
       
    def forward(self, mocap_data_src, mocap_data_tgt):
        
        #print("forward")
        
        #print("mocap_data_src s ", mocap_data_src.shape)
        #print("mocap_data_tgt s ", mocap_data_tgt.shape)

        src_mask = self.get_src_mask(mocap_data_src.shape[1]).to(mocap_data_src.device)
        tgt_mask = self.get_tgt_mask(mocap_data_tgt.shape[1]).to(mocap_data_tgt.device)
        
        #print("src_mask s ", src_mask.shape)
        #print("tgt_mask s ", tgt_mask.shape)
        
        mocap_src_embedded = self.mocap2embed(mocap_data_src) * math.sqrt(self.embed_dim)
        mocap_src_embedded = self.positional_encoder(mocap_src_embedded)
        
        #print("mocap_src_embedded s ", mocap_src_embedded.shape)
        
        mocap_tgt_embedded = self.mocap2embed(mocap_data_tgt) * math.sqrt(self.embed_dim)
        mocap_tgt_embedded = self.positional_encoder(mocap_tgt_embedded)
        
        #print("mocap_tgt_embedded s ", mocap_tgt_embedded.shape)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(mocap_src_embedded, mask=src_mask)
        
        #print("encoder_out s ", encoder_out.shape)
        
        decoder_out = self.decoder(mocap_tgt_embedded, encoder_out, tgt_mask =tgt_mask)
        
        #print("decoder_out s ", decoder_out.shape)
        
        out = self.embed2mocap(decoder_out)
        
        return out

mocap_dim = pose_dim

transformer = Transformer(mocap_dim=mocap_dim,
                          embed_dim=transformer_embed_dim, 
                          num_heads=transformer_head_count, 
                          num_encoder_layers=transformer_layer_count, 
                          num_decoder_layers=transformer_layer_count, 
                          dropout_p=transformer_dropout,
                          pos_encoding_max_length=pos_encoding_max_length).to(device)

print(transformer)

# test model

dancer1_input = X_batch[:, :-1, :].to(device)
dancer2_input = y_batch[:, :-1, :].to(device)
dancer2_output = transformer(dancer1_input, dancer2_input)

print("dancer1_input s ", dancer1_input.shape)
print("dancer2_input s ", dancer2_input.shape)
print("dancer2_output s ", dancer2_output.shape)

"""
Training
"""

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.336) # reduce the learning every 20 epochs by a factor of 10

mocap_mean_tensor = torch.tensor(mocap_mean).to(torch.float32).to(device)
mocap_std_tensor = torch.tensor(mocap_std).to(torch.float32).to(device)

mocap_mean_tensor = mocap_mean_tensor.reshape(1, 1, pose_dim)
mocap_std_tensor = mocap_std_tensor.reshape(1, 1, pose_dim)

# joint loss weights

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

def norm_loss(yhat):
    
    _yhat = yhat.reshape(-1, 4)
    _norm = torch.norm(_yhat, dim=1)
    _diff = (_norm - 1.0) ** 2
    _loss = torch.mean(_diff)
    return _loss

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.reshape(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.reshape((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.reshape((y.shape[0], y.shape[1], -1, 4))

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def rot_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.reshape((-1, 4))
    _yhat = yhat.reshape((-1, 4))

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    # inverse of quaternion: https://www.mathworks.com/help/aeroblks/quaternioninverse.html
    _yhat_inv = _yhat_norm * torch.tensor([[1.0, -1.0, -1.0, -1.0]], dtype=torch.float32).to(device)

    # calculate difference quaternion
    _diff = qmul(_yhat_inv, _y)
    # length of complex part
    _len = torch.norm(_diff[:, 1:], dim=1)
    # atan2
    _atan = torch.atan2(_len, _diff[:, 0])
    # abs
    _abs = torch.abs(_atan)
    
    _abs = _abs.reshape(-1, 1, joint_count)
    
    _abs_weighted = _abs * joint_loss_weights
    
    #print("_abs s ", _abs.shape)
    
    _loss = torch.mean(_abs_weighted)   
    return _loss

# transformer loss function
def loss(y, yhat):
    _norm_loss = norm_loss(yhat)
    _pos_loss = pos_loss(y, yhat)
    _rot_loss = rot_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * norm_loss_scale
    _total_loss += _pos_loss * pos_loss_scale
    _total_loss += _rot_loss * rot_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _rot_loss

def train_step(dancer1_mocap, dancer2_mocap):
    
    _dancer1_x = dancer1_mocap[:, :-1, :]
    _dancer2_x = dancer2_mocap[:, :-1, :]
    _dancer2_y = dancer2_mocap[:, 1:, :]
  
    _dancer1_x_norm = (_dancer1_x - mocap_mean_tensor) / mocap_std_tensor
    _dancer2_x_norm = (_dancer2_x - mocap_mean_tensor) / mocap_std_tensor
    _dancer1_x_norm = torch.nan_to_num(_dancer1_x_norm)
    _dancer2_x_norm = torch.nan_to_num(_dancer2_x_norm)
    
    _dancer2_yhat_norm = transformer(_dancer1_x_norm, _dancer2_x_norm)
     
    _dancer2_yhat = _dancer2_yhat_norm * mocap_std_tensor + mocap_mean_tensor
    
    _loss, _norm_loss, _pos_loss, _rot_loss = loss(_dancer2_y, _dancer2_yhat) 

    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss, _norm_loss, _pos_loss, _rot_loss

def test_step(dancer1_mocap, dancer2_mocap):
    
    _dancer1_x = dancer1_mocap[:, :-1, :]
    _dancer2_x = dancer2_mocap[:, :-1, :]
    _dancer2_y = dancer2_mocap[:, 1:, :]
  
    _dancer1_x_norm = (_dancer1_x - mocap_mean_tensor) / mocap_std_tensor
    _dancer2_x_norm = (_dancer2_x - mocap_mean_tensor) / mocap_std_tensor
    _dancer1_x_norm = torch.nan_to_num(_dancer1_x_norm)
    _dancer2_x_norm = torch.nan_to_num(_dancer2_x_norm)
    
    with torch.no_grad():
        _dancer2_yhat_norm = transformer(_dancer1_x_norm, _dancer2_x_norm)
         
        _dancer2_yhat = _dancer2_yhat_norm * mocap_std_tensor + mocap_mean_tensor
        
        _loss, _norm_loss, _pos_loss, _rot_loss = loss(_dancer2_y, _dancer2_yhat) 
    
    return _loss, _norm_loss, _pos_loss, _rot_loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []
    loss_history["norm"] = []
    loss_history["pos"] = []
    loss_history["rot"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []
        _norm_loss_per_epoch = []
        _pos_loss_per_epoch = []
        _rot_loss_per_epoch = []

        for train_batch in train_dataloader:
            X_batch = train_batch[0].to(device)
            y_batch = train_batch[1].to(device)

            _loss, _norm_loss, _pos_loss, _rot_loss = train_step(X_batch, y_batch)
            
            _loss = _loss.detach().cpu().numpy()
            _norm_loss = _norm_loss.detach().cpu().numpy()
            _pos_loss = _pos_loss.detach().cpu().numpy()
            _rot_loss = _rot_loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)
            _norm_loss_per_epoch.append(_norm_loss)
            _pos_loss_per_epoch.append(_pos_loss)
            _rot_loss_per_epoch.append(_rot_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _norm_loss_per_epoch = np.mean(np.array(_norm_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))
        _rot_loss_per_epoch = np.mean(np.array(_rot_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            batch_mocap = test_batch[0].to(device)
            batch_audio = test_batch[1].to(device)
            
            _loss, _, _, _ = test_step(batch_mocap, batch_audio)

            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["norm"].append(_norm_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        loss_history["rot"].append(_rot_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} norm {:01.4f} pos {:01.4f} rot {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _norm_loss_per_epoch, _pos_loss_per_epoch, _rot_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(image_file_name)

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(transformer.state_dict(), "results/weights/transformer_weights_epoch_{}".format(epochs))

"""
Inference
"""

# inference and rendering 
poseRenderer = PoseRenderer(edge_list)

# visualization settings
view_ele = 90.0
view_azi = -90.0
view_line_width = 4.0
view_size = 8.0

# create ref pose sequence
def create_ref_sequence_anim(mocap_index, start_pose_index, pose_count, file_name1, file_name2):
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    
    sequence_excerpt_dancer1 = pose_sequence_dancer1[start_pose_index:start_pose_index + pose_count]
    sequence_excerpt_dancer2 = pose_sequence_dancer2[start_pose_index:start_pose_index + pose_count]

    sequence_excerpt_dancer1 = torch.tensor(np.expand_dims(sequence_excerpt_dancer1, axis=0)).to(torch.float32).to(device)
    sequence_excerpt_dancer2 = torch.tensor(np.expand_dims(sequence_excerpt_dancer2, axis=0)).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence_dancer1 = forward_kinematics(sequence_excerpt_dancer1, zero_trajectory)
    skel_sequence_dancer2 = forward_kinematics(sequence_excerpt_dancer2, zero_trajectory)

    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1.cpu().numpy())
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer1[0].save(file_name1, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0)

    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2.cpu().numpy())
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer2[0].save(file_name2, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0)

def create_pred_sequence_anim(mocap_index, start_pose_index, pose_count, base_pose, file_name):
    
    transformer.eval()
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    sequence_excerpt_dancer1 = pose_sequence_dancer1[start_pose_index:start_pose_index + pose_count]
    sequence_excerpt_dancer1 = torch.from_numpy(sequence_excerpt_dancer1).to(torch.float32).to(device)
    sequence_excerpt_dancer1 = torch.reshape(sequence_excerpt_dancer1, (pose_count, pose_dim))
    
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    sequence_excerpt_dancer2 = pose_sequence_dancer2[start_pose_index:start_pose_index + pose_count]
    sequence_excerpt_dancer2 = torch.from_numpy(sequence_excerpt_dancer2).to(torch.float32).to(device)
    sequence_excerpt_dancer2 = torch.reshape(sequence_excerpt_dancer2, (pose_count, pose_dim))
    
    _input_dancer1 = sequence_excerpt_dancer1[:seq_length, :]
    _input_dancer2 = sequence_excerpt_dancer2[:seq_length, :]

    gen_sequence = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - seq_length):
        
        print("pI ", pI, " out of ", (pose_count - seq_length))
        
        _input_dancer1 = torch.unsqueeze(_input_dancer1, axis=0)
        _input_dancer2 = torch.unsqueeze(_input_dancer2, axis=0)

        _input_dancer1_norm = (_input_dancer1 - mocap_mean_tensor) / mocap_std_tensor
        _input_dancer2_norm = (_input_dancer2 - mocap_mean_tensor) / mocap_std_tensor
        
        _input_dancer1_norm = torch.nan_to_num(_input_dancer1_norm)
        _input_dancer2_norm = torch.nan_to_num(_input_dancer2_norm)
        
        with torch.no_grad():
            _pred_dancer2_norm = transformer(_input_dancer1_norm, _input_dancer2_norm)
        
        _pred_dancer2 = _pred_dancer2_norm * mocap_std_tensor + mocap_mean_tensor
        
        gen_sequence[pI] = _pred_dancer2[0, -1, :].detach().cpu().reshape(1, joint_count, joint_dim).numpy()

        _input_dancer1 = sequence_excerpt_dancer1[pI:seq_length + pI, :]
        _input_dancer2 = torch.cat([_input_dancer2[0, 1:, :].detach().clone(), _pred_dancer2[0, -1:, :].detach().clone()], axis=0)
        
    # fix quaternions in gen sequence
    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((pose_count, joint_count, joint_dim))
    gen_sequence = qfix(gen_sequence)
    gen_sequence = np.expand_dims(gen_sequence, axis=0)
    gen_sequence = torch.from_numpy(gen_sequence).to(torch.float32).to(device)
        
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_sequence = forward_kinematics(gen_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0) 
    
    transformer.train()
    
def create_pred_sequence_anim(mocap_index, start_pose_index, pose_count, base_pose, file_name):
    
    transformer.eval()

    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    sequence_excerpt_dancer1 = pose_sequence_dancer1[start_pose_index:start_pose_index + pose_count]
    
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    sequence_excerpt_dancer2 = pose_sequence_dancer2[start_pose_index:start_pose_index + pose_count]
    
    #print("sequence_excerpt_dancer1 s ", sequence_excerpt_dancer1.shape)

    start_seq_dancer1 = sequence_excerpt_dancer1[:seq_length, :]
    start_seq_dancer1 = torch.from_numpy(start_seq_dancer1).to(torch.float32).to(device)
    start_seq_dancer1 = torch.reshape(start_seq_dancer1, (seq_length, pose_dim))
    
    start_seq_dancer2 = sequence_excerpt_dancer2[:seq_length, :]
    start_seq_dancer2 = torch.from_numpy(start_seq_dancer2).to(torch.float32).to(device)
    start_seq_dancer2 = torch.reshape(start_seq_dancer2, (seq_length, pose_dim))

    _input_dancer1 = start_seq_dancer1
    _input_dancer2 = start_seq_dancer2
    
    gen_sequence = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - seq_length):
        
        print("pI ", pI, " out of ", (pose_count - seq_length))
        
        _input_dancer1_norm = (_input_dancer1 - mocap_mean_tensor)  / mocap_std_tensor
        _input_dancer2_norm = (_input_dancer2 - mocap_mean_tensor)  / mocap_std_tensor
        
        _input_dancer1_norm = torch.nan_to_num(_input_dancer1_norm)
        _input_dancer2_norm = torch.nan_to_num(_input_dancer2_norm)
        
        #print("_input_dancer1_norm s ", _input_dancer1_norm.shape)
        #print("_input_dancer2_norm s ", _input_dancer2_norm.shape)
        
        with torch.no_grad():
            _pred_dancer2_norm = transformer(_input_dancer1_norm, _input_dancer2_norm)

        _pred_dancer2_norm = _pred_dancer2_norm.reshape(-1, seq_length, pose_dim)
        
        #print("_pred_dancer2_norm s ", _pred_dancer2_norm.shape)
        
        _pred_dancer2 = _pred_dancer2_norm * mocap_std_tensor + mocap_mean_tensor
        
        #print("_pred_dancer2 s ", _pred_dancer2.shape)
        


        # shift input seqeunces one to the right
        # remove item from beginning input sequence
        # detach necessary to avoid error concerning running backprob a second time
        
        _input_dancer1 = sequence_excerpt_dancer1[pI:pI + seq_length, :]
        _input_dancer1 = torch.from_numpy(_input_dancer1).to(torch.float32).to(device)
        _input_dancer1 = torch.reshape(_input_dancer1, (seq_length, pose_dim))
        
        _input_dancer2 = _input_dancer2.detach().clone()
        _pred_dancer2 = _pred_dancer2[0].detach().clone()
        
        print("_input_dancer2 s ", _input_dancer2.shape)
        print("_pred_dancer2 s ", _pred_dancer2.shape)
        
        _input_dancer2 = torch.cat([_input_dancer2[1:, :], _pred_dancer2[-1:, :]], axis=0)
        
        _pred_dancer2_np = _pred_dancer2.cpu().numpy()
        _pred_dancer2_np = np.reshape(_pred_dancer2_np, (seq_length, joint_count, joint_dim))
        gen_sequence[pI] = _pred_dancer2_np[-1:, ...]
        

    # fix quaternions in gen sequence
    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((pose_count, joint_count, joint_dim))
    gen_sequence = qfix(gen_sequence)
    gen_sequence = np.expand_dims(gen_sequence, axis=0)
    gen_sequence = torch.from_numpy(gen_sequence).to(torch.float32).to(device)
        
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    skel_sequence = forward_kinematics(gen_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0) 
    
    transformer.train()

base_pose = all_mocap_data_dancer1[0]["motion"]["rot_local"][0]

mocap_index = 0
start_pose_index = 1000
pose_count = 1000

create_ref_sequence_anim(mocap_index, start_pose_index, pose_count, "results/anims/ref_dancer1_mocap_{}_start_{}_count_{}.gif".format(mocap_index, start_pose_index, pose_count), "results/anims/ref_dancer2_mocap_{}_start_{}_count_{}.gif".format(mocap_index, start_pose_index, pose_count))
create_pred_sequence_anim(mocap_index, start_pose_index, pose_count, base_pose, "results/anims/pred_dancer2_mocap_{}_start_{}_count_{}_epoch_{}_test.gif".format(mocap_index, start_pose_index, pose_count, epochs))





