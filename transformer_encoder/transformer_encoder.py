"""
dance motion to dance motion translation
employs transformer encoder from official pytorch implementation
network input and output are joint rotations (quaternions)
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg

import os, sys, time, subprocess
import numpy as np
import math

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
Mocap Data
"""

# mocap settings

mocap_file_path = "D:/Data/mocap/stocos/Duets/Amsterdam_2024/bvh_50hz"
mocap_files = [ 
    [ "Recording_JS-001_jason.bvh", "Recording_JS-001_sherise.bvh" ], 
    [ "Recording2_JS-001_jason.bvh", "Recording2_JS-001_sherise.bvh" ],
    [ "Recording3_JS-001_jason.bvh", "Recording3_JS-001_sherise.bvh" ]
    ]

mocap_valid_frame_ranges = [ 
    [ [ 500, 30000 ] ], 
    [ [ 490, 30679] ],
    [ [ 500, 30800] ]
    ]


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

mocap_fps = 50


"""
Model Settings
"""

rnn_layer_dim = 512
rnn_layer_count = 2

save_weights = True
load_weights = False
transformer_encoder_weights_file = "results_rnn_v2/weights/transformer_encoder_weights_epoch_100"

"""
Training settings
"""

batch_size = 32
test_percentage = 0.1

seq_length = 64

learning_rate = 1e-4
norm_loss_scale = 0.1
pos_loss_scale = 0.1
quat_loss_scale = 0.9
model_save_interval = 10

epochs = 100
save_history = True

"""
Visualization settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

"""
Load mocap data
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
        
        for pI in np.arange(frame_range_start, frame_range_end - seq_length - 1):

            sequence_excerpt_dancer1 = pose_sequence_dancer1[pI:pI+seq_length]
            dancer1_data.append(sequence_excerpt_dancer1)
            
            sequence_excerpt_dancer2 = pose_sequence_dancer2[pI:pI+seq_length]
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
Create Transformer Encoder Model
"""

layer_count = 6
head_count = 8
embed_dim = 512
dropout = 0.1

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


transformer_encoder = TransformerEncoder(data_dim=pose_dim, embed_dim=embed_dim, num_heads=head_count, num_encoder_layers=layer_count, dropout_p=dropout).to(device)

print(transformer_encoder)

# test model

batch_x, _ = next(iter(train_loader))
batch_x = batch_x.to(device)

print(batch_x.shape)

test_y2 = transformer_encoder(batch_x)

print(test_y2.shape)

if load_weights == True:
    transformer_encoder.load_state_dict(torch.load(transformer_encoder_weights_file))


"""
Training
"""


optimizer = torch.optim.Adam(transformer_encoder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

def norm_loss(yhat):
    _yhat = yhat.view(-1, 4)
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
    
    #print("pos_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))
    
    #print("_y_rot s ", _y_rot.shape)
    #print("_yhat_rot s ", _yhat_rot.shape)

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)
    
    #print("_y_pos s ", _y_pos.shape)
    #print("_yhat_pos s ", _yhat_pos.shape)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def quat_loss(y, yhat):
    
    #print("quat_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))
    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    
    #print("_y s ", _y.shape)
    #print("_yhat_norm s ", _yhat_norm.shape)
    
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
    
    #print("_abs s ", _abs.shape)
    
    _abs_weighted = _abs * joint_loss_weights
    
    _loss = torch.mean(_abs_weighted)   
    return _loss

# autoencoder loss function
def loss(y, yhat):
    _norm_loss = norm_loss(yhat)
    _pos_loss = pos_loss(y, yhat)
    _quat_loss = quat_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * norm_loss_scale
    _total_loss += _pos_loss * pos_loss_scale
    _total_loss += _quat_loss * quat_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss

def train_step(input_poses, target_poses):

    _pred_poses = transformer_encoder(input_poses)
    
    _loss, _norm_loss, _pos_loss, _quat_loss = loss(target_poses, _pred_poses) 
        
    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss, _norm_loss, _pos_loss, _quat_loss

def test_step(input_poses, target_poses):
    
    transformer_encoder.eval()

    with torch.no_grad():
        _pred_poses = transformer_encoder(input_poses)
        
        _loss, _norm_loss, _pos_loss, _quat_loss = loss(target_poses, _pred_poses) 
    
    transformer_encoder.train()
    
    return _loss, _norm_loss, _pos_loss, _quat_loss


def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []
    loss_history["norm"] = []
    loss_history["pos"] = []
    loss_history["quat"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []
        _norm_loss_per_epoch = []
        _pos_loss_per_epoch = []
        _quat_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_poses = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            _loss, _norm_loss, _pos_loss, _quat_loss = train_step(input_poses, target_poses)
            
            _loss = _loss.detach().cpu().numpy()
            _norm_loss = _norm_loss.detach().cpu().numpy()
            _pos_loss = _pos_loss.detach().cpu().numpy()
            _quat_loss = _quat_loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)
            _norm_loss_per_epoch.append(_norm_loss)
            _pos_loss_per_epoch.append(_pos_loss)
            _quat_loss_per_epoch.append(_quat_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _norm_loss_per_epoch = np.mean(np.array(_norm_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))
        _quat_loss_per_epoch = np.mean(np.array(_quat_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            input_poses = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)

            _loss, _, _, _ = test_step(input_poses, target_poses)
            
            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer_encoder.state_dict(), "results/weights/transformer_encoder_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["norm"].append(_norm_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        loss_history["quat"].append(_quat_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} norm {:01.4f} pos {:01.4f} quat {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _norm_loss_per_epoch, _pos_loss_per_epoch, _quat_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/rnn_history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/rnn_history_{}.png".format(epochs))

# save model weights
torch.save(transformer_encoder.state_dict(), "results/weights/transformer_encoder_weights_epoch_{}".format(epochs))

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

def create_pred_sequence_anim(mocap_index, start_pose_index, pose_count, pose_offset, base_pose, file_name):
    
    transformer_encoder.eval()
    
    seq_env = np.hanning(seq_length)
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    sequence_excerpt_dancer1 = pose_sequence_dancer1[start_pose_index:start_pose_index + pose_count]
    
    #print("sequence_excerpt_dancer1 s ", sequence_excerpt_dancer1.shape)

    """
    start_seq = sequence_excerpt_dancer1[:seq_length, :]
    
    #print("start_seq s ", start_seq.shape)
    
    start_seq = torch.from_numpy(start_seq).to(torch.float32).to(device)
    start_seq = torch.reshape(start_seq, (seq_length, pose_dim))
    
    next_seq = start_seq
    """
    
    gen_sequence = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - seq_length, pose_offset):
        
        start_seq = sequence_excerpt_dancer1[pI:pI + seq_length, :]
        start_seq = torch.from_numpy(start_seq).to(torch.float32).to(device)
        start_seq = torch.reshape(start_seq, (seq_length, pose_dim))
        next_seq = start_seq
        
        print("pI ", pI, " out of ", (pose_count - seq_length))
        
        with torch.no_grad():
            pred_seq = transformer_encoder(torch.unsqueeze(next_seq, axis=0))

        # normalize pred seq
        pred_seq = torch.squeeze(pred_seq)
        pred_seq = pred_seq.reshape((-1, 4))
        pred_seq = nn.functional.normalize(pred_seq, p=2, dim=1)
        pred_seq = pred_seq.reshape((seq_length, pose_dim))
        
        # blend pred seq into gen seq
        pred_seq = pred_seq.detach().cpu().numpy()
        pred_seq = np.reshape(pred_seq, (-1, joint_count, joint_dim))
        
        for si in range(seq_length):
            for ji in range(joint_count): 
                current_quat = gen_sequence[pI + si, ji, :]
                target_quat = pred_seq[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix )
                gen_sequence[pI + si, ji, :] = mix_quat
        
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
    
    transformer_encoder.train()
   
    
mocap_index = 0
start_pose_index = 1000
pose_count = 1000

    
create_ref_sequence_anim(mocap_index, start_pose_index, pose_count, "results/anims/ref_mI_{}_spI_{}_pc_{}_dancer1.gif".format(mocap_index, start_pose_index, pose_count), "results/anims/ref_mI_{}_spI_{}_pc_{}_dancer2.gif".format(mocap_index, start_pose_index, pose_count)) 
    
pose_offset = 16
base_pose = all_mocap_data_dancer1[mocap_index]["motion"]["rot_local"][0]

create_pred_sequence_anim(mocap_index, start_pose_index, pose_count, pose_offset, base_pose, "results/anims/pred_mI_{}_spI_{}_pc_{}_ep_{}_dancer2.gif".format(mocap_index, start_pose_index, pose_count, epochs))

# inference with new mocap file (not from training set)

new_mocap_file_path = "D:/Data/mocap/stocos/Duets/Amsterdam_2024/bvh_50hz"
new_mocap_files = [ [ "Recording_JM-001_jason.bvh", "Recording_JM-001_Muriel.bvh" ] ]
new_mocap_valid_frame_ranges = [ [ [ 500, 29500 ] ] ]

new_mocap_data_dancer1 = []
new_mocap_data_dancer2 = []

for mocap_file_dancer1, mocap_file_dancer2 in new_mocap_files:
    
    print("process file for dancer 1 ", mocap_file_dancer1)
    
    bvh_data_dancer1 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer1)
    mocap_data_dancer1 = mocap_tools.bvh_to_mocap(bvh_data_dancer1)
    mocap_data_dancer1["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data_dancer1["motion"]["rot_local_euler"], mocap_data_dancer1["rot_sequence"])

    new_mocap_data_dancer1.append(mocap_data_dancer1)

    print("process file for dancer 2 ", mocap_file_dancer2)
    
    bvh_data_dancer2 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer2)
    mocap_data_dancer2 = mocap_tools.bvh_to_mocap(bvh_data_dancer2)
    mocap_data_dancer2["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data_dancer2["motion"]["rot_local_euler"], mocap_data_dancer2["rot_sequence"])

    new_mocap_data_dancer2.append(mocap_data_dancer2)

# create new ref pose sequence
def create_new_ref_sequence_anim(mocap_index, start_pose_index, pose_count, file_name1, file_name2):
    
    mocap_data_dancer1 = new_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = new_mocap_data_dancer2[mocap_index]
    
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

def create_new_pred_sequence_anim(mocap_index, start_pose_index, pose_count, pose_offset, base_pose, file_name):
    
    transformer_encoder.eval()
    
    seq_env = np.hanning(seq_length)
    
    mocap_data_dancer1 = new_mocap_data_dancer1[mocap_index]
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    sequence_excerpt_dancer1 = pose_sequence_dancer1[start_pose_index:start_pose_index + pose_count]
    
    #print("sequence_excerpt_dancer1 s ", sequence_excerpt_dancer1.shape)

    """
    start_seq = sequence_excerpt_dancer1[:seq_length, :]
    
    #print("start_seq s ", start_seq.shape)
    
    start_seq = torch.from_numpy(start_seq).to(torch.float32).to(device)
    start_seq = torch.reshape(start_seq, (seq_length, pose_dim))
    
    next_seq = start_seq
    """
    
    gen_sequence = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - seq_length, pose_offset):
        
        start_seq = sequence_excerpt_dancer1[pI:pI + seq_length, :]
        start_seq = torch.from_numpy(start_seq).to(torch.float32).to(device)
        start_seq = torch.reshape(start_seq, (seq_length, pose_dim))
        next_seq = start_seq
        
        print("pI ", pI, " out of ", (pose_count - seq_length))
        
        with torch.no_grad():
            pred_seq = transformer_encoder(torch.unsqueeze(next_seq, axis=0))

        # normalize pred seq
        pred_seq = torch.squeeze(pred_seq)
        pred_seq = pred_seq.reshape((-1, 4))
        pred_seq = nn.functional.normalize(pred_seq, p=2, dim=1)
        pred_seq = pred_seq.reshape((seq_length, pose_dim))
        
        # blend pred seq into gen seq
        pred_seq = pred_seq.detach().cpu().numpy()
        pred_seq = np.reshape(pred_seq, (-1, joint_count, joint_dim))
        
        for si in range(seq_length):
            for ji in range(joint_count): 
                current_quat = gen_sequence[pI + si, ji, :]
                target_quat = pred_seq[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix )
                gen_sequence[pI + si, ji, :] = mix_quat
        
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
    
    transformer_encoder.train()
   
    
new_mocap_index = 0
new_start_pose_index = 2500
new_pose_count = 1000

create_new_ref_sequence_anim(new_mocap_index, new_start_pose_index, new_pose_count, "results/anims/ref_dancer1_new_mocap{}_start{}_count{}.gif".format(new_mocap_index, new_start_pose_index, new_pose_count), "results/anims/ref_dancer2_new_mocap{}_start{}_count{}.gif".format(new_mocap_index, new_start_pose_index, new_pose_count))
create_new_pred_sequence_anim(new_mocap_index, new_start_pose_index, new_pose_count, pose_offset, base_pose, "results/anims/pred_dancer2_new_mocap{}_start{}_count{}_epoch{}.gif".format(new_mocap_index, new_start_pose_index, new_pose_count, epochs))
