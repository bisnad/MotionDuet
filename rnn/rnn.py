"""
motion to motion translation
using a simple bidirectional LSTM
this is for motion capture data that stores joint rotations and recorded in BVH or FBX format
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
from common import fbx_tools as fbx
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

# important: the skeleton needs to be identical in all mocap recordings

mocap_file_path = "D:/Data/mocap/stocos/Duets/Amsterdam_2024/fbx_50hz"
mocap_files = [ [ "Jason_Take4.fbx", "Sherise_Take4.fbx" ] ]
mocap_valid_frame_ranges = [ [ [ 490, 30679] ] ]
mocap_pos_scale = 1.0
mocap_fps = 50


"""
# debug
mocap_file_path = "D:/Data/mocap/Daniel/Zed/fbx/"
mocap_files = [ [ "daniel_zed_solo1.fbx", "daniel_zed_solo1.fbx" ] ]
mocap_valid_frame_ranges = [ [ [ 0, 9100 ] ] ]
mocap_pos_scale = 1.0
mocap_fps = 30
"""

"""
Model Settings
"""

rnn_layer_dim = 512
rnn_layer_count = 2

save_weights = True
load_weights = False
rnn_weights_file = "results_JasonSherise_Take4/weights/rnn_weights_epoch_100"

"""
Training settings
"""

batch_size = 32
test_percentage = 0.1
seq_input_length = 64
learning_rate = 1e-4
norm_loss_scale = 0.1
pos_loss_scale = 0.1
quat_loss_scale = 0.9
model_save_interval = 10
epochs = 200
save_history = True

"""
# zed body34 specific joint loss weights
# todo: this information should be stored in config files
joint_loss_weights = [
    1.0, # PELVIS
    1.0, # NAVAL SPINE
    1.0, # CHEST SPINE
    1.0, # RIGHT CLAVICLE
    1.0, # RIGHT SHOULDER
    1.0, # RIGHT ELBOW
    1.0, # RIGHT WRIST
    1.0, # RIGHT HAND
    0.1, # RIGHT HANDTIP
    0.1, # RIGHT THUMB
    1.0, # NECK
    1.0, # HEAD
    0.1, # NOSE
    0.1, # LEFT EYE
    0.1, # LEFT EAR
    0.1, # RIGHT EYE
    0.1, # RIGHT EAR
    1.0, # LEFT CLAVICLE
    1.0, # LEFT SHOULDER
    1.0, # LEFT ELBOW
    1.0, # LEFT WRIST
    1.0, # LEFT HAND
    0.1, # LEFT HANDTIP
    0.1, # LEFT THUMB
    1.0, # LEFT HIP
    1.0, # LEFT KNEE
    1.0, # LEFT ANKLE
    1.0, # LEFT FOOT
    1.0, # LEFT HEEL
    1.0, # RIGHT HIP
    1.0, # RIGHT KNEE
    1.0, # RIGHT ANKLE
    1.0, # RIGHT FOOT
    1.0 # RIGHT HEEL
    ]
"""

joint_loss_weights = [ 1.0 ] # assign individual weights to joints if not all the weights are identical


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
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data_dancer1 = []
all_mocap_data_dancer2 = []

for mocap_file_dancer1, mocap_file_dancer2 in mocap_files:
    
    print("process file for dancer 1 ", mocap_file_dancer1)
    
    if mocap_file_dancer1.endswith(".bvh") or mocap_file_dancer1.endswith(".BVH"):
        bvh_data_dancer1 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer1)
        mocap_data_dancer1 = mocap_tools.bvh_to_mocap(bvh_data_dancer1)
    elif mocap_file_dancer1.endswith(".fbx") or mocap_file_dancer1.endswith(".FBX"):
        fbx_data_dancer1 = fbx_tools.load(mocap_file_path + "/" + mocap_file_dancer1)
        mocap_data_dancer1 = mocap_tools.fbx_to_mocap(fbx_data_dancer1)[0] # first skeleton only
   
    mocap_data_dancer1["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data_dancer1["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data_dancer1["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data_dancer1["skeleton"]["offsets"][0, 2] = 0.0 
    
    if mocap_file_dancer1.endswith(".bvh") or mocap_file_dancer1.endswith(".BVH"):
        mocap_data_dancer1["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data_dancer1["motion"]["rot_local_euler"], mocap_data_dancer1["rot_sequence"])
    elif mocap_file_dancer1.endswith(".fbx") or mocap_file_dancer1.endswith(".FBX"):
        mocap_data_dancer1["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data_dancer1["motion"]["rot_local_euler"], mocap_data_dancer1["rot_sequence"])
    
    all_mocap_data_dancer1.append(mocap_data_dancer1)

    print("process file for dancer 2 ", mocap_file_dancer2)
    
    if mocap_file_dancer2.endswith(".bvh") or mocap_file_dancer2.endswith(".BVH"):
        bvh_data_dancer2 = bvh_tools.load(mocap_file_path + "/" + mocap_file_dancer2)
        mocap_data_dancer2 = mocap_tools.bvh_to_mocap(bvh_data_dancer2)
    elif mocap_file_dancer2.endswith(".fbx") or mocap_file_dancer2.endswith(".FBX"):
        fbx_data_dancer2 = fbx_tools.load(mocap_file_path + "/" + mocap_file_dancer2)
        mocap_data_dancer2 = mocap_tools.fbx_to_mocap(fbx_data_dancer2)[0] # first skeleton only
        
    mocap_data_dancer2["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data_dancer2["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data_dancer2["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data_dancer2["skeleton"]["offsets"][0, 2] = 0.0 
    
    if mocap_file_dancer2.endswith(".bvh") or mocap_file_dancer2.endswith(".BVH"):
        mocap_data_dancer2["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data_dancer2["motion"]["rot_local_euler"], mocap_data_dancer2["rot_sequence"])
    elif mocap_file_dancer2.endswith(".fbx") or mocap_file_dancer2.endswith(".FBX"):
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
        
        for pI in np.arange(frame_range_start, frame_range_end - seq_input_length - 1):

            sequence_excerpt_dancer1 = pose_sequence_dancer1[pI:pI+seq_input_length]
            dancer1_data.append(sequence_excerpt_dancer1)
            
            sequence_excerpt_dancer2 = pose_sequence_dancer2[pI:pI+seq_input_length]
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
Reccurent Model
"""

class Reccurent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count):
        super(Reccurent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.output_dim = output_dim
            
        rnn_layers = []
        
        rnn_layers.append(("rnn", nn.LSTM(self.input_dim, self.hidden_dim, self.layer_count, batch_first=True, bidirectional=True)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        dense_layers = []
        dense_layers.append(("dense", nn.Linear(self.hidden_dim * 2, self.output_dim)))
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x1 s ", x.shape)
        
        xs = x.shape
    
        x = x.reshape(-1, self.hidden_dim * 2)
        
        #print("x2 s ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x3 s ", x.shape)
        
        x = x.reshape(-1, xs[1], self.output_dim)

        #print("x4 s ", x.shape)

        return x

rnn = Reccurent(pose_dim, rnn_layer_dim, pose_dim, rnn_layer_count).to(device)
print(rnn)

# test Reccurent model

batch_x, _ = next(iter(train_loader))
batch_x = batch_x.to(device)

print(batch_x.shape)

test_y2 = rnn(batch_x)

print(test_y2.shape)

if load_weights == True:
    rnn.load_state_dict(torch.load(rnn_weights_file))


"""
Training
"""

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) # reduce the learning every 20 epochs by a factor of 10

# extend joint loss array if necessary
if len(joint_loss_weights) == 1:
    joint_loss_weights *= joint_count

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

    _pred_poses = rnn(input_poses)
    
    _loss, _norm_loss, _pos_loss, _quat_loss = loss(target_poses, _pred_poses) 
        
    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss, _norm_loss, _pos_loss, _quat_loss

def test_step(input_poses, target_poses):
    
    rnn.eval()

    with torch.no_grad():
        _pred_poses = rnn(input_poses)
        
        _loss, _norm_loss, _pos_loss, _quat_loss = loss(target_poses, _pred_poses) 
    
    rnn.train()
    
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
            torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epoch))
        
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
torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epochs))

"""
Inference and Rendering 
"""

poseRenderer = PoseRenderer(edge_list)

# From here on TODO

def export_sequence_anim(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=1000.0 / mocap_fps, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]

    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    
    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    
    fbx_tools.write(pred_fbx, file_name)


def create_pred_sequence(pose_sequence, pose_offset, base_pose):
    
    pose_count = pose_sequence.shape[0]
    
    rnn.eval()
    
    seq_env = np.hanning(seq_input_length)
    
    seq_dancer1 = pose_sequence
    gen_sequence_dancer2 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose, dtype=np.float32)
    
    for pI in range(0, pose_count - seq_input_length, pose_offset):

        start_seq_dancer1 = seq_dancer1[pI:pI + seq_input_length, :]
        start_seq_dancer1 = torch.from_numpy(start_seq_dancer1).to(torch.float32).to(device)
        start_seq_dancer1 = torch.reshape(start_seq_dancer1, (seq_input_length, pose_dim))
        next_seq_dancer1 = start_seq_dancer1
        
        with torch.no_grad():
            pred_seq_dancer2 = rnn(torch.unsqueeze(next_seq_dancer1, axis=0))

        # normalize pred seq
        pred_seq_dancer2 = torch.squeeze(pred_seq_dancer2)
        pred_seq_dancer2 = pred_seq_dancer2.reshape((-1, 4))
        pred_seq_dancer2 = nn.functional.normalize(pred_seq_dancer2, p=2, dim=1)
        pred_seq_dancer2 = pred_seq_dancer2.reshape((seq_input_length, pose_dim))

        # blend pred seq into gen seq
        pred_seq_dancer2 = pred_seq_dancer2.detach().cpu().numpy()
        pred_seq_dancer2 = np.reshape(pred_seq_dancer2, (-1, joint_count, joint_dim))
        
        for si in range(seq_input_length):
            for ji in range(joint_count): 
                current_quat = gen_sequence_dancer2[pI + si, ji, :]
                target_quat = pred_seq_dancer2[si, ji, :]
                quat_mix = seq_env[si]
                mix_quat = slerp(current_quat, target_quat, quat_mix )
                gen_sequence_dancer2[pI + si, ji, :] = mix_quat
                
    # fix quaternions in gen sequence
    gen_sequence_dancer2 = gen_sequence_dancer2.reshape((-1, 4))
    gen_sequence_dancer2 = gen_sequence_dancer2 / np.linalg.norm(gen_sequence_dancer2, ord=2, axis=1, keepdims=True)
    gen_sequence_dancer2 = gen_sequence_dancer2.reshape((pose_count, joint_count, joint_dim))
    gen_sequence_dancer2 = qfix(gen_sequence_dancer2)
    gen_sequence_dancer2 = np.reshape(gen_sequence_dancer2, (pose_count, joint_count, joint_dim))

    rnn.train()
    
    return gen_sequence_dancer2

# create original sequences

seq_index = 0
seq_start = 1000
seq_length = 1000

orig_sequence_dancer1 = all_mocap_data_dancer1[seq_index]["motion"]["rot_local"].astype(np.float32)
orig_sequence_dancer2 = all_mocap_data_dancer2[seq_index]["motion"]["rot_local"].astype(np.float32)

export_sequence_anim(orig_sequence_dancer1[seq_start:seq_start+seq_length], "results/anims/orig_sequence_dancer1_index_{}_seq_start_{}_length_{}.gif".format(seq_index, seq_start, seq_length))
export_sequence_fbx(orig_sequence_dancer1[seq_start:seq_start+seq_length], "results/anims/orig_sequence_dancer1_index_{}_seq_start_{}_length_{}.fbx".format(seq_index, seq_start, seq_length))

export_sequence_anim(orig_sequence_dancer2[seq_start:seq_start+seq_length], "results/anims/orig_sequence_dancer2_index_{}_seq_start_{}_length_{}.gif".format(seq_index, seq_start, seq_length))
export_sequence_fbx(orig_sequence_dancer2[seq_start:seq_start+seq_length], "results/anims/orig_sequence_dancer2_index_{}_seq_start_{}_length_{}.fbx".format(seq_index, seq_start, seq_length))

# create predicted sequences

seq_index = 0
seq_start = 1000
seq_length = 1000
pose_offset = 16
base_pose = all_mocap_data_dancer1[seq_index]["motion"]["rot_local"][0]

orig_sequence_dancer1 = all_mocap_data_dancer1[seq_index]["motion"]["rot_local"].astype(np.float32)
pred_sequence_dancer2 = create_pred_sequence(orig_sequence_dancer1[seq_start:seq_start+seq_length], pose_offset, base_pose)

export_sequence_anim(pred_sequence_dancer2, "results/anims/pred_sequence_dancer2_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(pred_sequence_dancer2, "results/anims/pred_sequence_dancer2_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))
