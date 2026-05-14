"""
Transformer for Motion-to-Motion Translation 
Upgraded with 6D rotations, FBX/BVH handling, and root trajectory tracking
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as nnF
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import copy
import numpy as np
import csv
import matplotlib.pyplot as plt

# mocap specific imports
from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer
from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

# -------------------------------------------------------------------------------------------------
# Compute Device
# -------------------------------------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# -------------------------------------------------------------------------------------------------
# Mocap Settings
# -------------------------------------------------------------------------------------------------

mocap_file_path = "E:/Data/mocap/stocos/Duets/Amsterdam_2024/fbx_50hz"
mocap_files = [ [ "Jason_Take3.fbx", "Sherise_Take3.fbx" ] ]
mocap_valid_frame_ranges = [ [ [ 500, 30000] ] ]
mocap_fps = 50

train_root_trajectory = False
mocap_pos_scale = 1.0

joint_loss_weights = [ 1.0 ] * 28 # Simplified to match the count of 28 joints

# -------------------------------------------------------------------------------------------------
# Save Paths Settings
# -------------------------------------------------------------------------------------------------

save_path = "results_Jason_Sherise_Take1/"
save_weights_path = save_path + "weights/"
save_history_path = save_path + "history/"
save_anims_path = save_path + "anims/"
save_anim_formats = ["gif", "fbx"]

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_anims_path, exist_ok=True)

# -------------------------------------------------------------------------------------------------
# Model Settings
# -------------------------------------------------------------------------------------------------

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 512
transformer_dropout = 0.1   

# -------------------------------------------------------------------------------------------------
# Training Settings
# -------------------------------------------------------------------------------------------------

seq_length = 64
seq_non_teacherforcing = 10
teacher_forcing_prob = 0.5

batch_size = 32
test_percentage = 0.1

learning_rate = 1e-4
pos_loss_scale = 0.1
rot_loss_scale = 0.9
traj_loss_scale = 0.1

model_save_interval = 10
save_weights = True
epochs = 200

# -------------------------------------------------------------------------------------------------
# Mocap Visualisation Settings
# -------------------------------------------------------------------------------------------------

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

# -------------------------------------------------------------------------------------------------
# Utility: Variable Timestamp Resampling (Time-Range Filtered)
# -------------------------------------------------------------------------------------------------

def resample_mocap_data(mocap_data, target_fps, time_ranges):
    times_dict = mocap_data["motion"].get("times", {})
    joints = mocap_data["skeleton"]["joints"]
    num_joints = len(joints)
    
    pos_local = mocap_data["motion"]["pos_local"]
    rot_local_euler = mocap_data["motion"]["rot_local_euler"]
    
    def get_joint_data(data, j_idx):
        return data[j_idx] if isinstance(data, list) else data[:, j_idx, :]
            
    joint_times_list = []
    for j_idx, j_name in enumerate(joints):
        if j_name in times_dict:
            j_times = times_dict[j_name]
        else:
            j_frames = len(get_joint_data(pos_local, j_idx))
            orig_fps = mocap_data.get("frame_rate", target_fps)
            j_times = np.arange(j_frames) / orig_fps
        joint_times_list.append(j_times)
        
    resampled_segments = []
    for t_range in time_ranges:
        start_time, end_time = t_range[0], t_range[1]
        target_times = np.arange(start_time, end_time, 1.0 / target_fps)
        num_frames = len(target_times)
        
        new_pos_local = np.zeros((num_frames, num_joints, 3))
        new_rot_local_euler = np.zeros((num_frames, num_joints, 3))
        
        for j_idx in range(num_joints):
            j_times = joint_times_list[j_idx]
            j_pos = get_joint_data(pos_local, j_idx)
            j_rot = get_joint_data(rot_local_euler, j_idx)
            
            if len(j_times) == 0: continue
            if len(j_times) == 1:
                new_pos_local[:, j_idx, :] = j_pos[0]
                new_rot_local_euler[:, j_idx, :] = j_rot[0]
                continue
                
            for i in range(3):
                new_pos_local[:, j_idx, i] = np.interp(target_times, j_times, j_pos[:, i])
                
            j_rot_rad = np.deg2rad(j_rot)
            j_rot_rad_unwrapped = np.unwrap(j_rot_rad, axis=0)
            j_rot_deg_unwrapped = np.rad2deg(j_rot_rad_unwrapped)
            
            for i in range(3):
                new_rot_local_euler[:, j_idx, i] = np.interp(target_times, j_times, j_rot_deg_unwrapped[:, i])
                
        segment_data = copy.deepcopy(mocap_data)
        segment_data["motion"]["pos_local"] = new_pos_local
        segment_data["motion"]["rot_local_euler"] = new_rot_local_euler
        segment_data["frame_rate"] = target_fps
        
        if "times" in segment_data["motion"]:
            del segment_data["motion"]["times"]
            
        resampled_segments.append(segment_data)
        
    return resampled_segments

# -------------------------------------------------------------------------------------------------
# Load Data - Mocap
# -------------------------------------------------------------------------------------------------

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data_dancer1 = []
all_mocap_data_dancer2 = []

for i, (mocap_file_dancer1, mocap_file_dancer2) in enumerate(mocap_files):
    
    # Convert valid frame ranges to time ranges for resampling
    time_ranges = [[r[0] / mocap_fps, r[1] / mocap_fps] for r in mocap_valid_frame_ranges[i]]

    print("process file for dancer 1 ", mocap_file_dancer1)
    d1_path = os.path.join(mocap_file_path, mocap_file_dancer1)
    d1_data = mocap_tools.bvh_to_mocap(bvh_tools.load(d1_path)) if d1_path.endswith(".bvh") else mocap_tools.fbx_to_mocap(fbx_tools.load(d1_path))[0]
    segments1 = resample_mocap_data(d1_data, mocap_fps, time_ranges)

    print("process file for dancer 2 ", mocap_file_dancer2)
    d2_path = os.path.join(mocap_file_path, mocap_file_dancer2)
    d2_data = mocap_tools.bvh_to_mocap(bvh_tools.load(d2_path)) if d2_path.endswith(".bvh") else mocap_tools.fbx_to_mocap(fbx_tools.load(d2_path))[0]
    segments2 = resample_mocap_data(d2_data, mocap_fps, time_ranges)

    for seg1, seg2 in zip(segments1, segments2):
        for s in [seg1, seg2]:
            s["skeleton"]["offsets"] *= mocap_pos_scale
            s["motion"]["pos_local"] *= mocap_pos_scale
            
            if not train_root_trajectory:
                s["skeleton"]["offsets"][0, 0] = 0.0
                s["skeleton"]["offsets"][0, 2] = 0.0
                s["motion"]["pos_local"][:, 0, 0] = 0.0
                s["motion"]["pos_local"][:, 0, 2] = 0.0
                
            rot_quat = mocap_tools.euler_to_quat(s["motion"]["rot_local_euler"], s["rot_sequence"])
            s["motion"]["rot_local"] = rot_np.quat_to_r6d(rot_quat)
            
        all_mocap_data_dancer1.append(seg1)
        all_mocap_data_dancer2.append(seg2)

# Retrieve mocap properties
mocap_data = all_mocap_data_dancer1[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = 6
pose_dim = joint_count * joint_dim
input_dim = pose_dim + 3 if train_root_trajectory else pose_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

def get_edge_list(children):
    edge_list = []
    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    return edge_list

edge_list = get_edge_list(children)

# -------------------------------------------------------------------------------------------------
# Create Dataset
# -------------------------------------------------------------------------------------------------

total_seq_length = seq_length + seq_non_teacherforcing

dancer1_data = []
dancer2_data = []

for i in range(len(all_mocap_data_dancer1)):
    mocap_data_dancer1 = all_mocap_data_dancer1[i]
    mocap_data_dancer2 = all_mocap_data_dancer2[i]
    
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"].reshape(-1, pose_dim)
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"].reshape(-1, pose_dim)

    if train_root_trajectory:
        root_positions1 = mocap_data_dancer1["motion"]["pos_local"][:, 0, :]
        root_positions2 = mocap_data_dancer2["motion"]["pos_local"][:, 0, :]
        pose_sequence_dancer1 = np.concatenate((root_positions1, pose_sequence_dancer1), axis=1)
        pose_sequence_dancer2 = np.concatenate((root_positions2, pose_sequence_dancer2), axis=1)
        
    for pI in np.arange(0, len(pose_sequence_dancer1) - total_seq_length - 2):
        dancer1_data.append(pose_sequence_dancer1[pI:pI+total_seq_length + 1])
        dancer2_data.append(pose_sequence_dancer2[pI:pI+total_seq_length + 1])

dancer1_data = np.array(dancer1_data, dtype=np.float32)
dancer2_data = np.array(dancer2_data, dtype=np.float32)

# Pre-normalize root trajectory over the dataset globally
if train_root_trajectory:
    combined_roots = np.concatenate((dancer1_data[:, :, :3], dancer2_data[:, :, :3]), axis=0)
    root_pos_mean = np.mean(combined_roots, axis=(0,1), keepdims=True)
    root_pos_std = np.std(combined_roots, axis=(0,1), keepdims=True)
    root_pos_std[root_pos_std == 0] = 1.0

    dancer1_data[:, :, :3] = (dancer1_data[:, :, :3] - root_pos_mean) / root_pos_std
    dancer2_data[:, :, :3] = (dancer2_data[:, :, :3] - root_pos_mean) / root_pos_std

    root_pos_mean_tensor = torch.from_numpy(root_pos_mean[0]).to(device)
    root_pos_std_tensor = torch.from_numpy(root_pos_std[0]).to(device)

dancer1_data = torch.from_numpy(dancer1_data)
dancer2_data = torch.from_numpy(dancer2_data)

class DuetDataset(Dataset):
    def __init__(self, dancer1_data, dancer2_data):
        self.dancer1_data = dancer1_data
        self.dancer2_data = dancer2_data
    
    def __len__(self): return self.dancer1_data.shape[0]
    def __getitem__(self, idx): return self.dancer1_data[idx, ...], self.dancer2_data[idx, ...]

# Temporal split 
total_samples = len(dancer1_data)
test_size = int(test_percentage * total_samples)
train_size = total_samples - test_size

train_dataset = DuetDataset(dancer1_data[:train_size], dancer2_data[:train_size])
test_dataset = DuetDataset(dancer1_data[train_size:], dancer2_data[train_size:])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------------------------------------------------------------------------
# Create Models
# -------------------------------------------------------------------------------------------------

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

pos_encoding_max_length = seq_length + seq_non_teacherforcing
transformer = Transformer(mocap_dim=input_dim, embed_dim=transformer_embed_dim, num_heads=transformer_head_count, 
                          num_encoder_layers=transformer_layer_count, num_decoder_layers=transformer_layer_count, 
                          dropout_p=transformer_dropout, pos_encoding_max_length=pos_encoding_max_length).to(device)

print(transformer)

# -------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------

optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.336) 
joint_loss_weights_t = torch.tensor(joint_loss_weights, dtype=torch.float32).reshape(1, 1, -1).to(device)

def forward_kinematics(rotation_matrices, root_positions):
    t_offsets = torch.tensor(offsets).to(device)
    expanded_offsets = t_offsets.expand(rotation_matrices.shape[0], rotation_matrices.shape[1], offsets.shape[0], offsets.shape[1]).unsqueeze(-1)
    positions_world = []
    rotations_world = []
    
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotation_matrices[:, :, 0])
        else:
            parent_rot = rotations_world[parents[jI]]
            local_offset = expanded_offsets[:, :, jI]
            rotated_offset = torch.matmul(parent_rot, local_offset).squeeze(-1)
            positions_world.append(rotated_offset + positions_world[parents[jI]])

            if len(children[jI]) > 0:
                new_world_rot = torch.matmul(parent_rot, rotation_matrices[:, :, jI])
                rotations_world.append(new_world_rot)
            else:
                rotations_world.append(parent_rot)
                
    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def pos_loss(y, yhat):
    if train_root_trajectory:
        y_root_traj = (y[:, :, :3] * root_pos_std_tensor) + root_pos_mean_tensor
        yhat_root_traj = (yhat[:, :, :3] * root_pos_std_tensor) + root_pos_mean_tensor
        y_rot_6d = y[:, :, 3:].reshape(y.shape[0], y.shape[1], joint_count, 6)
        yhat_rot_6d = yhat[:, :, 3:].reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)
    else:
        y_root_traj = torch.zeros((y.shape[0], y.shape[1], 3)).to(device)
        yhat_root_traj = torch.zeros((yhat.shape[0], yhat.shape[1], 3)).to(device)
        y_rot_6d = y.reshape(y.shape[0], y.shape[1], joint_count, 6)
        yhat_rot_6d = yhat.reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)

    y_mat = rot_to.r6d_to_mat(y_rot_6d)
    yhat_mat = rot_to.r6d_to_mat(yhat_rot_6d)

    y_pos = forward_kinematics(y_mat, y_root_traj)
    yhat_pos = forward_kinematics(yhat_mat, yhat_root_traj)

    pos_diff = torch.norm(y_pos - yhat_pos, dim=3)
    return torch.mean(pos_diff * joint_loss_weights_t)

def rot_loss(y, yhat):
    if train_root_trajectory:
        y = y[:, :, 3:]
        yhat = yhat[:, :, 3:]

    y_rot_6d = y.reshape(y.shape[0], y.shape[1], joint_count, 6)
    yhat_rot_6d = yhat.reshape(yhat.shape[0], yhat.shape[1], joint_count, 6)
    y_mat = rot_to.r6d_to_mat(y_rot_6d)
    yhat_mat = rot_to.r6d_to_mat(yhat_rot_6d)

    trace = torch.diagonal(torch.matmul(y_mat.transpose(-1, -2), yhat_mat), dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -0.9999, 0.9999))
    return torch.mean(angle * joint_loss_weights_t)

def loss(y, yhat):
    _pos_loss = pos_loss(y, yhat)
    _rot_loss = rot_loss(y, yhat)
    
    _total_loss = (_pos_loss * pos_loss_scale) + (_rot_loss * rot_loss_scale)
    
    if train_root_trajectory:
        traj_mse = torch.mean((y[:, :, :3] - yhat[:, :, :3]) ** 2)
        _total_loss += (traj_mse * traj_loss_scale)
        
    return _total_loss, _pos_loss, _rot_loss

def train_step(dancer1_mocap, dancer2_mocap, teacher_forcing):
    transformer.train()

    if teacher_forcing:
        _dancer1_x = dancer1_mocap[:, :-1, :]
        _dancer2_x = dancer2_mocap[:, :-1, :]
        _dancer2_y = dancer2_mocap[:, 1:, :]

        _dancer2_yhat = transformer(_dancer1_x, _dancer2_x)
    else:
        _dancer1_x = dancer1_mocap[:, :seq_length, :]
        _dancer2_x = dancer2_mocap[:, :seq_length, :]
        _dancer2_y = dancer2_mocap[:, 1:seq_length + seq_non_teacherforcing, :]

        _dancer2_yhat = transformer(_dancer1_x, _dancer2_x)
        __dancer2_yhat_all = _dancer2_yhat

        for i in range(1, seq_non_teacherforcing):
            _dancer1_x = dancer1_mocap[:, i:seq_length+i, :]
            _dancer2_x_step = _dancer2_yhat.detach()
            _dancer2_yhat = transformer(_dancer1_x, _dancer2_x_step)
            __dancer2_yhat_all = torch.cat([__dancer2_yhat_all, _dancer2_yhat[:, -1:, :]], axis=1)

        _dancer2_yhat = __dancer2_yhat_all
    
    _loss, _pos_loss, _rot_loss = loss(_dancer2_y, _dancer2_yhat) 

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss, _pos_loss, _rot_loss

@torch.no_grad()
def test_step(dancer1_mocap, dancer2_mocap, teacher_forcing):
    transformer.eval()

    if teacher_forcing:
        _dancer1_x = dancer1_mocap[:, :-1, :]
        _dancer2_x = dancer2_mocap[:, :-1, :]
        _dancer2_y = dancer2_mocap[:, 1:, :]
        _dancer2_yhat = transformer(_dancer1_x, _dancer2_x)
    else:
        _dancer1_x = dancer1_mocap[:, :seq_length, :]
        _dancer2_x = dancer2_mocap[:, :seq_length, :]
        _dancer2_y = dancer2_mocap[:, 1:seq_length + seq_non_teacherforcing, :]

        _dancer2_yhat = transformer(_dancer1_x, _dancer2_x)
        __dancer2_yhat_all = _dancer2_yhat

        for i in range(1, seq_non_teacherforcing):
            _dancer1_x = dancer1_mocap[:, i:seq_length+i, :]
            _dancer2_x_step = _dancer2_yhat.detach()
            _dancer2_yhat = transformer(_dancer1_x, _dancer2_x_step)
            __dancer2_yhat_all = torch.cat([__dancer2_yhat_all, _dancer2_yhat[:, -1:, :]], axis=1)

        _dancer2_yhat = __dancer2_yhat_all
    
    _loss, _pos_loss, _rot_loss = loss(_dancer2_y, _dancer2_yhat) 
    return _loss, _pos_loss, _rot_loss

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {"train": [], "test": [], "pos": [], "rot": []}

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []
        _pos_loss_per_epoch = []
        _rot_loss_per_epoch = []

        for train_batch in train_dataloader:
            X_batch = train_batch[0].to(device)
            y_batch = train_batch[1].to(device)

            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            _loss, _pos_loss, _rot_loss = train_step(X_batch, y_batch, use_teacher_forcing)
            
            _train_loss_per_epoch.append(_loss.detach().cpu().numpy())
            _pos_loss_per_epoch.append(_pos_loss.detach().cpu().numpy())
            _rot_loss_per_epoch.append(_rot_loss.detach().cpu().numpy())

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))
        _rot_loss_per_epoch = np.mean(np.array(_rot_loss_per_epoch))

        _test_loss_per_epoch = []
        for test_batch in test_dataloader:
            batch_mocap = test_batch[0].to(device)
            batch_mocap_tgt = test_batch[1].to(device) 

            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            _loss, _, _ = test_step(batch_mocap, batch_mocap_tgt, use_teacher_forcing)
            _test_loss_per_epoch.append(_loss.detach().cpu().numpy())
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(transformer.state_dict(), "{}/transformer_weights_epoch_{}".format(save_weights_path, epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        loss_history["rot"].append(_rot_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} pos {:01.4f} rot {:01.4f} time {:01.2f}'.format(
            epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _pos_loss_per_epoch, _rot_loss_per_epoch, time.time()-start))
    
    return loss_history

if save_weights == True:
    loss_history = train(train_loader, test_loader, epochs)

    def save_loss_as_image(loss_history, image_file_name):
        keys = list(loss_history.keys())
        epochs_arr = range(len(loss_history[keys[0]]))
        for key in keys: plt.plot(epochs_arr, loss_history[key], label=key)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(image_file_name)

    def save_loss_as_csv(loss_history, csv_file_name):
        with open(csv_file_name, 'w') as csv_file:
            csv_columns = list(loss_history.keys())
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
            csv_writer.writeheader()
            for row in range(len(loss_history[csv_columns[0]])):
                csv_row = {key: loss_history[key][row] for key in loss_history.keys()}
                csv_writer.writerow(csv_row)

    os.makedirs("results/histories/", exist_ok=True)
    os.makedirs("results/weights/", exist_ok=True)
    os.makedirs("results/anims/", exist_ok=True)

    save_loss_as_csv(loss_history, "{}/history_{}.csv".format(save_history_path, epochs))
    save_loss_as_image(loss_history, "{}/history_{}.png".format(save_history_path, epochs))
    torch.save(transformer.state_dict(), "{}/transformer_weights_epoch_{}".format(save_weights_path, epochs))

# -------------------------------------------------------------------------------------------------
# Inference and Export
# -------------------------------------------------------------------------------------------------

poseRenderer = PoseRenderer(edge_list)

def export_sequence_gif(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    rot_6d = torch.tensor(rot_sequence).reshape(-1, joint_count, 6)
    rot_matrices = rot_to.r6d_to_mat(rot_6d.unsqueeze(0).to(device))
    root_traj_tensor = torch.tensor(root_trajectory).unsqueeze(0).to(device)
    
    skel_sequence = forward_kinematics(rot_matrices, root_traj_tensor).squeeze().cpu().numpy()
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, mocap_template, file_name):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_template.get("frame_rate", mocap_fps),
        "rot_sequence": mocap_template["rot_sequence"],
        "skeleton": mocap_template["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(
        pred_dataset["motion"]["rot_local"], 
        pred_dataset["rot_sequence"]
    )

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, mocap_template, file_name):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_template.get("frame_rate", mocap_fps),
        "rot_sequence": mocap_template["rot_sequence"],
        "skeleton": mocap_template["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(
        pred_dataset["motion"]["rot_local"], 
        pred_dataset["rot_sequence"]
    )

    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    fbx_tools.write(pred_fbx, file_name)

def create_ref_sequence(mocap_index, start_pose_index, pose_count, base_file_name_1, base_file_name_2, export_formats=["gif", "bvh", "fbx"]):
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    
    def extract_sequence(mocap_data):
        pose_sequence = mocap_data["motion"]["rot_local"].reshape(-1, pose_dim)[start_pose_index:start_pose_index + pose_count]
        if train_root_trajectory:
            root_sequence = mocap_data["motion"]["pos_local"][start_pose_index:start_pose_index + pose_count, 0, :]
            return np.concatenate((root_sequence, pose_sequence), axis=1)
        else:
            return pose_sequence
            
    seq_dancer1 = extract_sequence(mocap_data_dancer1)
    seq_dancer2 = extract_sequence(mocap_data_dancer2)
    
    for fmt in export_formats:
        if fmt == "gif":
            export_sequence_gif(seq_dancer1, f"{base_file_name_1}.gif")
            export_sequence_gif(seq_dancer2, f"{base_file_name_2}.gif")
        elif fmt == "bvh":
            export_sequence_bvh(seq_dancer1, mocap_data_dancer1, f"{base_file_name_1}.bvh")
            export_sequence_bvh(seq_dancer2, mocap_data_dancer2, f"{base_file_name_2}.bvh")
        elif fmt == "fbx":
            export_sequence_fbx(seq_dancer1, mocap_data_dancer1, f"{base_file_name_1}.fbx")
            export_sequence_fbx(seq_dancer2, mocap_data_dancer2, f"{base_file_name_2}.fbx")

def create_pred_sequence(mocap_index, start_pose_index, pose_count, base_file_name, export_formats=["gif", "bvh", "fbx"]):
    transformer.eval()
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    
    def extract_sequence(mocap_data):
        pose_seq = mocap_data["motion"]["rot_local"].reshape(-1, pose_dim)
        if train_root_trajectory:
            root_pos = mocap_data["motion"]["pos_local"][:, 0, :]
            pose_seq = np.concatenate((root_pos, pose_seq), axis=1)
        return pose_seq[start_pose_index:start_pose_index + pose_count]

    seq_dancer1_full = extract_sequence(mocap_data_dancer1)
    seq_dancer2_full = extract_sequence(mocap_data_dancer2)
    
    # Normalize trajectory for network input
    if train_root_trajectory:
        seq_dancer1_full[:, :3] = (seq_dancer1_full[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()
        seq_dancer2_full[:, :3] = (seq_dancer2_full[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()

    _input_dancer1 = torch.from_numpy(seq_dancer1_full).to(torch.float32).to(device)
    _input_dancer2 = torch.from_numpy(seq_dancer2_full[:seq_length]).to(torch.float32).to(device)
    
    gen_sequence = []
    
    for pI in range(0, pose_count - seq_length):
        print("Generating frame ", pI, " out of ", (pose_count - seq_length))
        
        _in_d1 = _input_dancer1[pI:seq_length + pI, :].unsqueeze(0)
        _in_d2 = _input_dancer2.unsqueeze(0)

        with torch.no_grad():
            _pred_dancer2 = transformer(_in_d1, _in_d2)
        
        pred_pose = _pred_dancer2[0, -1, :]
        gen_sequence.append(pred_pose.detach().cpu().numpy())
        _input_dancer2 = torch.cat([_input_dancer2[1:, :], pred_pose.unsqueeze(0)], axis=0)
        
    gen_sequence = np.array(gen_sequence)
    
    # De-normalize trajectory for export
    if train_root_trajectory:
        gen_sequence[:, :3] = (gen_sequence[:, :3] * root_pos_std.flatten()) + root_pos_mean.flatten()

    for fmt in export_formats:
        if fmt == "gif":
            export_sequence_gif(gen_sequence, f"{base_file_name}.gif")
        elif fmt == "bvh":
            export_sequence_bvh(gen_sequence, mocap_data_dancer2, f"{base_file_name}.bvh")
        elif fmt == "fbx":
            export_sequence_fbx(gen_sequence, mocap_data_dancer2, f"{base_file_name}.fbx")
            
    transformer.train()

# -------------------------------------------------------------------------------------------------
# Run Exports
# -------------------------------------------------------------------------------------------------

mocap_index = 0
start_pose_index = 1000
pose_count = 1000

# Base names without extensions 
ref_base_1 = f"{save_anims_path}/ref_dancer1_mocap_{mocap_index}_start_{start_pose_index}_count_{pose_count}"
ref_base_2 = f"{save_anims_path}/ref_dancer2_mocap_{mocap_index}_start_{start_pose_index}_count_{pose_count}"
pred_base = f"{save_anims_path}/pred_dancer2_mocap_{mocap_index}_start_{start_pose_index}_count_{pose_count}_epoch_{epochs}_test"

create_ref_sequence(mocap_index, start_pose_index, pose_count, ref_base_1, ref_base_2, export_formats=save_anim_formats)
create_pred_sequence(mocap_index, start_pose_index, pose_count, pred_base, export_formats=save_anim_formats)