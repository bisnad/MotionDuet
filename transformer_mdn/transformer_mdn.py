"""
Dance motion to dance motion translation (Dancer 1 -> Dancer 2)
Employs Full Transformer (Encoder-Decoder), MDN, Physical Constraint Losses, 
and Scheduled Teacher Forcing for robust autoregressive generation.
"""

import os
import sys
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as nnF
from scipy.signal import savgol_filter

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.pose_renderer import PoseRenderer
from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_file_path = "../../../Data/Mocap/Xsens/Stocos/Duets/fbx_50hz"
mocap_files = [ 
    [ "Jason_Take3.fbx", "Sherise_Take3.fbx" ],
    [ "Jason_Take4.fbx", "Sherise_Take4.fbx" ], 
    [ "Jason_Take5.fbx", "Sherise_Take5.fbx" ]
]

# Provide [start_frame, end_frame]
mocap_valid_frame_ranges = [ 
    [ [ 500, 30000 ] ],
    [ [ 500, 27000 ] ], 
    [ [ 670, 30000 ] ]
]

mocap_fps = 50
mocap_pos_scale = 1.0
train_root_trajectory = True

"""
Model Settings
"""

num_mixtures = 5
temperature = 1.0

layer_count = 6
head_count = 8
embed_dim = 512
dropout = 0.1

"""
Training Settings
"""

batch_size = 32
test_percentage = 0.1
seq_input_length = 64
seq_output_length = 4
seq_offset = 4

teacher_forcing_prob = 0.5
learning_rate = 1e-4

pos_loss_scale = 0.1
rot_loss_scale = 1.0
traj_loss_scale = 0.1
nll_loss_scale = 1.0

model_save_interval = 10

epochs = 2
save_weights = True
load_weights = False
transformer_weights_file = "results/weights/transformer_mdn_weights_epoch_100.pt"


# -------------------------------------------------------------------------------------------------
# Render Settings
# -------------------------------------------------------------------------------------------------

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0


"""
Save Settings
"""

save_path = "results_Jason_Sherise_Takes_all"
save_weights_path = f"{save_path}/weights/"
save_history_path = f"{save_path}/history/"
save_anims_path = f"{save_path}/anims/"
save_anim_formats = ["gif", "fbx", "bvh"]

os.makedirs(save_weights_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_anims_path, exist_ok=True)

"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data_dancer1 = []
all_mocap_data_dancer2 = []

def process_mocap_file(file_name, valid_ranges):
    
    print("Processing file:", file_name)
    
    file_path = os.path.join(mocap_file_path, file_name)
    if file_name.endswith('.bvh') or file_name.endswith('.BVH'):
        raw_data = bvh_tools.load(file_path)
        mocap_data = mocap_tools.bvh_to_mocap(raw_data)
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    else:
        raw_data = fbx_tools.load(file_path)
        mocap_data = mocap_tools.fbx_to_mocap(raw_data)[0]
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale

    if not train_root_trajectory:
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 0] = 0.0
        mocap_data["motion"]["pos_local"][:, 0, 2] = 0.0

    mocap_data["motion"]["rot_local_6d"] = rot_np.quat_to_r6d(mocap_data["motion"]["rot_local"])
    
    extracted_seqs = []
    for (start, end) in valid_ranges:
        seq_data = {
            "rot_local_6d": mocap_data["motion"]["rot_local_6d"][start:end],
            "pos_local": mocap_data["motion"]["pos_local"][start:end]
        }
        extracted_seqs.append(seq_data)
        
    return extracted_seqs, mocap_data

for i, (file1, file2) in enumerate(mocap_files):
    seqs1, mocap_data = process_mocap_file(file1, mocap_valid_frame_ranges[i])
    seqs2, _= process_mocap_file(file2, mocap_valid_frame_ranges[i])
    
    if i == 0:
        global_skeleton = mocap_data["skeleton"]
        global_rot_sequence = mocap_data["rot_sequence"]
        
    all_mocap_data_dancer1.extend(seqs1)
    all_mocap_data_dancer2.extend(seqs2)

joint_count = all_mocap_data_dancer1[0]["rot_local_6d"].shape[1]
joint_dim = 6
pose_dim = (joint_count * joint_dim) + (3 if train_root_trajectory else 0)

offsets = global_skeleton["offsets"].astype(np.float32)
parents = global_skeleton["parents"]
children = global_skeleton["children"]
joint_loss_weights_t = torch.ones((1, 1, joint_count)).to(device)

def get_edge_list(children):
    edge_list = []
    for p_idx in range(len(children)):
        for c_idx in children[p_idx]:
            edge_list.append([p_idx, c_idx])
    return edge_list
edge_list = get_edge_list(children)

"""
Create Dataset
"""

dancer1_data = []
dancer2_data = []

for i in range(len(all_mocap_data_dancer1)):
    m1, m2 = all_mocap_data_dancer1[i], all_mocap_data_dancer2[i]
    
    seq1_rot = np.reshape(m1["rot_local_6d"], (-1, joint_count * joint_dim))
    seq2_rot = np.reshape(m2["rot_local_6d"], (-1, joint_count * joint_dim))
    
    if train_root_trajectory:
        seq1 = np.concatenate([m1["pos_local"][:, 0, :], seq1_rot], axis=1)
        seq2 = np.concatenate([m2["pos_local"][:, 0, :], seq2_rot], axis=1)
    else:
        seq1, seq2 = seq1_rot, seq2_rot

    for pI in np.arange(0, seq1.shape[0] - seq_input_length - 1, seq_offset):
        dancer1_data.append(seq1[pI : pI + seq_input_length])
        dancer2_data.append(seq2[pI : pI + seq_output_length]) # concurrent sequence for Dancer 2

dancer1_data = np.array(dancer1_data, dtype=np.float32)
dancer2_data = np.array(dancer2_data, dtype=np.float32)

if train_root_trajectory:
    root_pos_mean = np.mean(dancer1_data[:, :, :3], axis=(0, 1), keepdims=True)
    root_pos_std = np.std(dancer1_data[:, :, :3], axis=(0, 1), keepdims=True)
    root_pos_std[root_pos_std == 0] = 1.0
    
    dancer1_data[:, :, :3] = (dancer1_data[:, :, :3] - root_pos_mean) / root_pos_std
    dancer2_data[:, :, :3] = (dancer2_data[:, :, :3] - root_pos_mean) / root_pos_std

    root_pos_mean_tensor = torch.from_numpy(root_pos_mean).to(device)
    root_pos_std_tensor = torch.from_numpy(root_pos_std).to(device)

class DuetDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
    def __len__(self):
        return self.data1.shape[0]
    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

full_dataset = DuetDataset(dancer1_data, dancer2_data)
test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Model Architecture
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * div_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * div_term)
        self.register_buffer("pos_encoding", pos_encoding.unsqueeze(0))
        
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])

class MDNLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.out_dim = out_dim
        self.pi_head = nn.Linear(hidden_dim, num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, num_mixtures * out_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_mixtures * out_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        log_pi = nnF.log_softmax(self.pi_head(x), dim=-1)
        mu = self.mu_head(x).view(batch_size, seq_len, self.num_mixtures, self.out_dim)
        sigma = nnF.elu(self.sigma_head(x).view(batch_size, seq_len, self.num_mixtures, self.out_dim)) + 1.0 + 1e-6
        return log_pi, mu, sigma

class TransformerSeq2SeqMDN(nn.Module):
    def __init__(self, data_dim, embed_dim, num_heads, num_enc_layers, num_dec_layers, dropout_p, num_mixtures):
        super().__init__()
        self.src_proj = nn.Linear(data_dim, embed_dim)
        self.tgt_proj = nn.Linear(data_dim, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, dropout_p)
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_enc_layers, 
            num_decoder_layers=num_dec_layers, dropout=dropout_p, batch_first=True
        )
        self.mdn = MDNLayer(embed_dim, data_dim, num_mixtures)
       
    def get_tgt_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, src, tgt):
        tgt_mask = self.get_tgt_mask(tgt.shape[1]).to(tgt.device)
        src_emb = self.pos_enc(self.src_proj(src) * math.sqrt(embed_dim))
        tgt_emb = self.pos_enc(self.tgt_proj(tgt) * math.sqrt(embed_dim))
        return self.mdn(self.transformer(src=src_emb, tgt=tgt_emb, tgt_mask=tgt_mask))

model = TransformerSeq2SeqMDN(pose_dim, embed_dim, head_count, layer_count, layer_count, dropout, num_mixtures).to(device)

if load_weights:
    model.load_state_dict(torch.load(transformer_weights_file))

"""
Loss Functions
"""

def forward_kinematics(rot_matrices, root_positions):
    t_offsets = torch.tensor(offsets).to(device)
    
    # Reshape t_offsets to [1, 1, joint_count, 3, 1] before expanding
    t_offsets = t_offsets.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    
    # Now it can safely be expanded
    expanded_offsets = t_offsets.expand(rot_matrices.shape[0], rot_matrices.shape[1], offsets.shape[0], 3, 1)
    
    positions_world, rotations_world = [], []
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rot_matrices[:, :, 0])
        else:
            p_rot = rotations_world[parents[jI]]
            p_pos = positions_world[parents[jI]]
            local_offset = expanded_offsets[:, :, jI]
            
            rotated_offset = torch.matmul(p_rot, local_offset).squeeze(-1)
            positions_world.append(p_pos + rotated_offset)
            
            if len(children[jI]) > 0:
                rotations_world.append(torch.matmul(p_rot, rot_matrices[:, :, jI]))
            else:
                rotations_world.append(p_rot)
                
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

def mdn_nll_loss(log_pi, mu, sigma, target):
    target = target.unsqueeze(2)
    var = (sigma ** 2)
    log_normal = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((target - mu) ** 2) / var
    log_normal = torch.sum(log_normal, dim=-1)
    return -torch.logsumexp(log_pi + log_normal, dim=-1).mean()

def sample_mdn(log_pi, mu, sigma, temp=1.0):
    batch_size, seq_len, K = log_pi.shape
    pi_probs = nnF.softmax(log_pi / temp, dim=-1)
    chosen_idx = torch.multinomial(pi_probs.reshape(-1, K), 1).view(batch_size, seq_len, 1, 1)
    
    idx_expanded = chosen_idx.expand(batch_size, seq_len, 1, mu.size(-1))
    chosen_mu = torch.gather(mu, 2, idx_expanded).squeeze(2)
    chosen_sigma = torch.gather(sigma, 2, idx_expanded).squeeze(2)
    
    epsilon = torch.randn_like(chosen_mu) * temp
    sampled_pose = chosen_mu + chosen_sigma * epsilon

    if train_root_trajectory:
        root_pred = sampled_pose[..., :3]
        rot_pred = sampled_pose[..., 3:]
    else:
        rot_pred = sampled_pose

    rot_pred_6d = rot_pred.reshape(batch_size, seq_len, joint_count, 6)
    rot_mat = rot_to.r6d_to_mat(rot_pred_6d)
    rot_pred_norm = rot_to.mat_to_r6d(rot_mat).reshape(batch_size, seq_len, -1)

    if train_root_trajectory:
        return torch.cat([root_pred, rot_pred_norm], dim=-1)
    return rot_pred_norm

def compute_total_loss(log_pi, mu, sigma, tgt_poses):
    # Predict the poses based on the MDN distributions
    pred_poses = sample_mdn(log_pi, mu, sigma, temp=1.0)
    
    # CRITICAL FIX: Align sequence lengths to prevent broadcasting failures
    # We take the minimum length to ensure exact matching between predictions and targets
    min_seq_len = min(tgt_poses.shape[1], pred_poses.shape[1])
    tgt_poses_aligned = tgt_poses[:, :min_seq_len, :]
    pred_poses_aligned = pred_poses[:, :min_seq_len, :]
    log_pi_aligned = log_pi[:, :min_seq_len, :]
    mu_aligned = mu[:, :min_seq_len, :, :]
    sigma_aligned = sigma[:, :min_seq_len, :, :]

    # Calculate losses on the aligned tensors
    nll = mdn_nll_loss(log_pi_aligned, mu_aligned, sigma_aligned, tgt_poses_aligned)
    p_loss = pos_loss(tgt_poses_aligned, pred_poses_aligned)
    r_loss = rot_loss(tgt_poses_aligned, pred_poses_aligned)
    
    total = (nll * nll_loss_scale) + (p_loss * pos_loss_scale) + (r_loss * rot_loss_scale)
    
    if train_root_trajectory:
        traj_mse = torch.mean((tgt_poses_aligned[:, :, :3] - pred_poses_aligned[:, :, :3])**2)
        total += traj_mse * traj_loss_scale
        
    return total, nll, p_loss, r_loss

"""
Training Functions
"""

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.336)

def train_step(src_poses, tgt_poses, teacher_forcing=True):
    model.train()
    start_token = torch.zeros((tgt_poses.size(0), 1, pose_dim), device=device)
    
    if teacher_forcing:
        decoder_input = torch.cat([start_token, tgt_poses[:, :-1, :]], dim=1)
        log_pi, mu, sigma = model(src_poses, decoder_input)
        loss, nll, p_loss, r_loss = compute_total_loss(log_pi, mu, sigma, tgt_poses)
    else:
        decoder_input = start_token
        log_pi_list, mu_list, sigma_list = [], [], []
        
        for oi in range(seq_output_length):
            log_pi, mu, sigma = model(src_poses, decoder_input)
            
            curr_log_pi = log_pi[:, -1:, :]
            curr_mu = mu[:, -1:, :, :]
            curr_sigma = sigma[:, -1:, :, :]
            
            log_pi_list.append(curr_log_pi)
            mu_list.append(curr_mu)
            sigma_list.append(curr_sigma)
            
            # Predict next frame and append
            pred_pose = sample_mdn(curr_log_pi, curr_mu, curr_sigma, temp=1.0)
            decoder_input = torch.cat([decoder_input, pred_pose.detach().clone()], dim=1)

        log_pi_all = torch.cat(log_pi_list, dim=1)
        mu_all = torch.cat(mu_list, dim=1)
        sigma_all = torch.cat(sigma_list, dim=1)
        
        loss, nll, p_loss, r_loss = compute_total_loss(log_pi_all, mu_all, sigma_all, tgt_poses)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), nll.item(), p_loss.item(), r_loss.item()

@torch.no_grad()
def test_step(src_poses, tgt_poses, is_training, teacher_forcing=True):
    model.eval()
    start_token = torch.zeros((tgt_poses.size(0), 1, pose_dim), device=device)
    
    if teacher_forcing:
        decoder_input = torch.cat([start_token, tgt_poses[:, :-1, :]], dim=1)
        log_pi, mu, sigma = model(src_poses, decoder_input)
        loss, nll, p_loss, r_loss = compute_total_loss(log_pi, mu, sigma, tgt_poses)
    else:
        decoder_input = start_token
        log_pi_list, mu_list, sigma_list = [], [], []
        
        for oi in range(seq_output_length):
            log_pi, mu, sigma = model(src_poses, decoder_input)
            
            curr_log_pi = log_pi[:, -1:, :]
            curr_mu = mu[:, -1:, :, :]
            curr_sigma = sigma[:, -1:, :, :]
            
            log_pi_list.append(curr_log_pi)
            mu_list.append(curr_mu)
            sigma_list.append(curr_sigma)
            
            # Predict next frame and append
            pred_pose = sample_mdn(curr_log_pi, curr_mu, curr_sigma, temp=1.0)
            decoder_input = torch.cat([decoder_input, pred_pose.detach().clone()], dim=1)

        log_pi_all = torch.cat(log_pi_list, dim=1)
        mu_all = torch.cat(mu_list, dim=1)
        sigma_all = torch.cat(sigma_list, dim=1)
        
        loss, nll, p_loss, r_loss = compute_total_loss(log_pi_all, mu_all, sigma_all, tgt_poses)

    return loss.item(), nll.item(), p_loss.item(), r_loss.item()

def train(train_dataloader, test_dataloader, epochs):
    loss_history = {"train": [], "test": [], "nll": [], "pos": [], "rot": []}

    for epoch in range(epochs):
        start = time.time()
        _train_loss_per_epoch = []
        _nll_loss_per_epoch = []
        _pos_loss_per_epoch = []
        _rot_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)

            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            _loss, _nll_loss, _pos_loss, _rot_loss = train_step(input_pose_sequences, target_poses, use_teacher_forcing)

            # FIX: Just append the floats, remove .item()
            _train_loss_per_epoch.append(_loss)
            _nll_loss_per_epoch.append(_nll_loss)
            _pos_loss_per_epoch.append(_pos_loss)
            _rot_loss_per_epoch.append(_rot_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _nll_loss_per_epoch = np.mean(np.array(_nll_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))
        _rot_loss_per_epoch = np.mean(np.array(_rot_loss_per_epoch))

        _test_loss_per_epoch = []

        for test_batch in test_dataloader:
            input_pose_sequences = test_batch[0].to(device)
            target_poses = test_batch[1].to(device)
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            # FIX: Just extract the first float (total loss)
            _loss, _, _, _ = test_step(input_pose_sequences, target_poses, is_training=False, teacher_forcing=use_teacher_forcing)
            
            # FIX: Just append the float, remove .item()
            _test_loss_per_epoch.append(_loss)

        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_weights:
            torch.save(model.state_dict(), f"{save_weights_path}transformer_weights_epoch_{epoch}.pt")

        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["nll"].append(_nll_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        loss_history["rot"].append(_rot_loss_per_epoch)

        scheduler.step()

        print ('epoch {} : train: {:01.4f} test: {:01.4f} nll {:01.4f} pos {:01.4f} rot {:01.4f} time {:01.2f}'.format(
            epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _nll_loss_per_epoch, _pos_loss_per_epoch, _rot_loss_per_epoch, time.time()-start))

    return loss_history

# -------------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------------

def plot_training_history(loss_history, file_name):
    epochs_range = range(1, len(loss_history["train"]) + 1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Transformer MDN 6D Training History', fontsize=16, y=0.92)

    axes[0].plot(epochs_range, loss_history["train"], label='Train Total Loss', color='blue', linewidth=2)
    axes[0].plot(epochs_range, loss_history["test"], label='Test Total Loss', color='orange', linewidth=2, linestyle='--')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Overall Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(epochs_range, loss_history["nll"], label='Train MDN NLL', color='purple', linewidth=2)
    axes[1].set_ylabel('NLL')
    axes[1].set_title('Negative Log-Likelihood (Distribution Learning)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    axes[2].plot(epochs_range, loss_history["pos"], label='Train Pos Loss (Scaled)', color='green', linewidth=2)
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Position Loss')
    axes[2].set_title('Physical Constraints (Positional Error)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle=':', alpha=0.6)

    axes[3].plot(epochs_range, loss_history["rot"], label='Train Rot Loss (Scaled)', color='red', linewidth=2)
    axes[3].set_xlabel('Epochs')
    axes[3].set_ylabel('Rotation Loss')
    axes[3].set_title('Physical Constraints (Rotational Error)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {file_name}")


# -------------------------------------------------------------------------------------------------
# Run Training
# -------------------------------------------------------------------------------------------------

if save_weights == True:
    # fit model
    loss_history = train(train_loader, test_loader, epochs)

    # save history
    utils.save_loss_as_csv(loss_history, f"{save_history_path}history_{epochs}.csv")

    # plot history
    plot_training_history(loss_history, f"{save_history_path}history_{epochs}.png")

    # save model weights
    torch.save(model.state_dict(), f"{save_weights_path}transformer_weights_epoch_{epochs}.pt")


"""
Inference & Rendering
"""

poseRenderer = PoseRenderer(edge_list)

def sample_mdn(log_pi, mu, sigma, gaussian_temp=1.0):
    batch_size, seq_len, K = log_pi.shape
    pi_probs = nnF.softmax(log_pi / gaussian_temp, dim=-1)
    pi_probs_flat = pi_probs.reshape(-1, K)
    
    chosen_idx = torch.multinomial(pi_probs_flat, 1)
    chosen_idx = chosen_idx.view(batch_size, seq_len, 1, 1)
    
    idx_expanded = chosen_idx.expand(batch_size, seq_len, 1, mu.size(-1))
    chosen_mu = torch.gather(mu, 2, idx_expanded).squeeze(2)
    chosen_sigma = torch.gather(sigma, 2, idx_expanded).squeeze(2)
    
    epsilon = torch.randn_like(chosen_mu) * gaussian_temp
    sampled_pose = chosen_mu + chosen_sigma * epsilon

    # Re-orthogonalize generated 6D rotations
    if train_root_trajectory:
        root_pred = sampled_pose[..., :3]
        rot_pred = sampled_pose[..., 3:]
    else:
        rot_pred = sampled_pose

    rot_pred_6d = rot_pred.reshape(batch_size, seq_len, joint_count, 6)
    rot_mat = rot_to.r6d_to_mat(rot_pred_6d)
    rot_pred_norm = rot_to.mat_to_r6d(rot_mat).reshape(batch_size, seq_len, -1)

    if train_root_trajectory:
        return torch.cat([root_pred, rot_pred_norm], dim=-1)
    return rot_pred_norm

def smooth_motion(pred_poses, window_length=9, polyorder=3):
    pose_count = pred_poses.shape[0]
    if pose_count > window_length:
        pred_poses = savgol_filter(pred_poses, window_length, polyorder, axis=0)

    pred_poses_t = torch.from_numpy(pred_poses)
    
    if train_root_trajectory:
        root_smoothed = pred_poses_t[:, :3]
        rot_smoothed = pred_poses_t[:, 3:]
    else:
        rot_smoothed = pred_poses_t
        
    rot_6d = rot_smoothed.reshape(pose_count, joint_count, 6)
    rot_mat = rot_to.r6d_to_mat(rot_6d)
    rot_6d_norm = rot_to.mat_to_r6d(rot_mat).reshape(pose_count, -1)

    if train_root_trajectory:
        return torch.cat([root_smoothed, rot_6d_norm], dim=-1).numpy()
    return rot_6d_norm.numpy()

def generate_sequence(src_sequence, seq_len, temp=1.0):
    model.eval()
    tgt_seq = torch.zeros((1, 1, pose_dim), device=device)
    pred_poses = []
    
    for i in range(seq_len):
        with torch.no_grad():
            log_pi, mu, sigma = model(src_sequence, tgt_seq)
            next_frame = sample_mdn(log_pi[:, -1:, :], mu[:, -1:, :, :], sigma[:, -1:, :, :], gaussian_temp=temp)
            pred_poses.append(next_frame)
            tgt_seq = torch.cat([tgt_seq, next_frame], dim=1)
            
    pred_numpy = torch.cat(pred_poses, dim=1).squeeze(0).cpu().numpy()

    if train_root_trajectory:
        pred_numpy[:, :3] = (pred_numpy[:, :3] * root_pos_std.flatten()) + root_pos_mean.flatten()
        
    return smooth_motion(pred_numpy)

def export_sequence_anim(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    rot_sequence = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    rot_sequence_tensor = torch.tensor(np.expand_dims(rot_sequence, axis=0)).to(device)
    rot_matrices = rot_to.r6d_to_mat(rot_sequence_tensor)

    root_trajectory = torch.tensor(np.expand_dims(root_trajectory, axis=0)).to(device)
    skel_sequence = forward_kinematics(rot_matrices, root_trajectory)
    skel_sequence = skel_sequence.detach().cpu().numpy().squeeze()

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_data["frame_rate"],
        "rot_sequence": mocap_data["rot_sequence"],
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }

    # set joint local positions
    # the root joint gets its local position from the trajectory, all other joints from the offsets
    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    # Convert 6D network output to Quaternions, then to Euler Angles 
    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    
    # Use the euler conversion work-around specifically designed for BVHs in this mocap_tools version
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(
        pred_dataset["motion"]["rot_local"], 
        pred_dataset["rot_sequence"]
    )

    # Use the internal mocap_to_bvh compiler
    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    # Write the BVH to disk using the standard bvh_tools script structure
    bvh_tools.write(pred_bvh, file_name) 

def export_sequence_fbx(pose_sequence, file_name):
    pose_count = pose_sequence.shape[0]
    if train_root_trajectory:
        root_trajectory = pose_sequence[:, :3]
        rot_sequence = pose_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pose_sequence

    pred_dataset = {
        "frame_rate": mocap_data["frame_rate"],
        "rot_sequence": mocap_data["rot_sequence"],
        "skeleton": mocap_data["skeleton"],
        "motion": {}
    }

    pos_local = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    if train_root_trajectory: pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local

    rot_seq_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_seq_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    fbx_tools.write(pred_fbx, file_name)


def export_sequence(pred_sequence, filename):
    pose_count = pred_sequence.shape[0]
    
    if train_root_trajectory:
        root_trajectory = pred_sequence[:, :3]
        rot_sequence = pred_sequence[:, 3:]
    else:
        root_trajectory = np.zeros((pose_count, 3), dtype=np.float32)
        rot_sequence = pred_sequence
        
    rot_6d = np.reshape(rot_sequence, (pose_count, joint_count, 6))
    
    pred_dataset = {
        "framerate": mocap_fps,
        "rot_sequence": global_rot_sequence,
        "skeleton": global_skeleton,
        "motion": {}
    }
    
    pos_local = np.repeat(np.expand_dims(global_skeleton["offsets"], axis=0), pose_count, axis=0)
    pos_local[:, 0, :] = root_trajectory
    pred_dataset["motion"]["pos_local"] = pos_local
    
    # Convert back to Quaternions and Euler for export
    pred_dataset["motion"]["rot_local"] = rot_np.r6d_to_quat(rot_6d)
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], global_rot_sequence)
    
    if filename.endswith(".fbx"):
        pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
        fbx_tools.write(pred_fbx, filename)
    elif filename.endswith(".bvh"):
        pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
        bvh_tools.write(pred_bvh, filename)

# -------------------------------------------------------------------------------------------------
# Run Inference and Export
# -------------------------------------------------------------------------------------------------

model.eval()


# Example sequence
mocap_index = 0
start_idx = 1000
pose_count = 500

# Orig Sequence

dancer1_mocap = all_mocap_data_dancer1[mocap_index]
dancer2_mocap = all_mocap_data_dancer2[mocap_index]

dancer1_rot = dancer1_mocap["rot_local_6d"][start_idx:start_idx+pose_count, ...].astype(np.float32)
dancer2_rot = dancer2_mocap["rot_local_6d"][start_idx:start_idx+pose_count, ...].astype(np.float32)
dancer1_root_pos = dancer1_mocap["pos_local"][start_idx:start_idx+pose_count, 0, ...].astype(np.float32)
dancer2_root_pos = dancer2_mocap["pos_local"][start_idx:start_idx+pose_count, 0, ...].astype(np.float32)

dancer1_rot = np.reshape(dancer1_rot, (-1, joint_count * 6))
dancer2_rot = np.reshape(dancer2_rot, (-1, joint_count * 6))

if train_root_trajectory:
    dancer1_seq = np.concatenate([dancer1_root_pos, dancer1_rot], axis=1)
    dancer2_seq = np.concatenate([dancer2_root_pos, dancer2_rot], axis=1)
    # And you normalized it
    #dancer1_seq[:, :3] = (dancer1_seq[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()
    #dancer2_seq[:, :3] = (dancer2_seq[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()
else:
    dancer1_seq = dancer1_rot
    dancer2_seq = dancer2_rot
    
if "gif" in save_anim_formats:
    export_sequence_anim(dancer1_seq, f"{save_anims_path}orig_dancer1.gif") 
    export_sequence_anim(dancer2_seq, f"{save_anims_path}orig_dancer2.gif") 
if "fbx" in save_anim_formats:
    export_sequence_fbx(dancer1_seq, f"{save_anims_path}orig_dancer1.fbx")
    export_sequence_fbx(dancer2_seq, f"{save_anims_path}orig_dancer2.fbx")
if "bvh" in save_anim_formats:
    export_sequence_bvh(dancer1_seq, f"{save_anims_path}orig_dancer1.bvh")
    export_sequence_bvh(dancer2_seq, f"{save_anims_path}orig_dancer2.bvh")
    
    
# Generated Sequence

dancer1_mocap = all_mocap_data_dancer1[mocap_index]
dancer1_rot = dancer1_mocap["rot_local_6d"][start_idx:start_idx+pose_count, ...].astype(np.float32)
dancer1_root_pos = dancer1_mocap["pos_local"][start_idx:start_idx+pose_count, 0, ...].astype(np.float32)

dancer1_rot = np.reshape(dancer1_rot, (-1, joint_count * 6))

if train_root_trajectory:
    dancer1_seq = np.concatenate([dancer1_root_pos, dancer1_rot], axis=1)
    # normalized it
    dancer1_seq[:, :3] = (dancer1_seq[:, :3] - root_pos_mean.flatten()) / root_pos_std.flatten()
else:
    dancer1_seq = dancer1_rot


dancer1_seq = torch.from_numpy(dancer1_seq).unsqueeze(0).to(device).float()
dancer2_seq = generate_sequence(dancer1_seq, pose_count, temp=temperature)

if "gif" in save_anim_formats:
    export_sequence_anim(dancer2_seq, f"{save_anims_path}pred_mdn_dancer2_ep{epochs}.gif")
if "fbx" in save_anim_formats:
    export_sequence_fbx(dancer2_seq, f"{save_anims_path}pred_mdn_dancer2_ep{epochs}.fbx")
if "bvh" in save_anim_formats:
    export_sequence_bvh(dancer2_seq, f"{save_anims_path}pred_mdn_dancer2_ep{epochs}.bvh")

