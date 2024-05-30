"""
motion to motion translation
using a variational autoencoder
the approach is based on deepfakes, i.e. have two autoencoders, one for dancer1 and one for dancer2. 
The two autoencoders share the same encoder but have different decoders
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Data
"""

mocap_file_path = "../../../../../../Data/mocap/stocos/duets"
mocap_files = [ [ "Jason_Take4.bvh", "Sherise_Take4.bvh" ] ]
mocap_valid_frame_ranges = [ [ [ 490, 30679] ] ]

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

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 256
ae_rnn_bidirectional = True
ae_dense_layer_sizes = [ 512 ]

save_weights = True
load_weights = True

encoder_weights_file = "results_deepfake_jason_sherise/weights/encoder_weights_epoch_600"
decoder1_weights_file = "results_deepfake_jason_sherise/weights/decoder1_weights_epoch_600"
decoder2_weights_file = "results_deepfake_jason_sherise/weights/decoder2_weights_epoch_600"

"""
Training Settings
"""

sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 16
train_percentage = 0.8 # train / test split
test_percentage  = 0.2
dp_learning_rate = 5e-4
ae_learning_rate = 1e-4
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_kld_loss_scale = 0.0 # will be calculated
kld_scale_cycle_duration = 100
kld_scale_min_const_duration = 20
kld_scale_max_const_duration = 20
min_kld_scale = 0.0
max_kld_scale = 0.1

epochs = 600
model_save_interval = 50
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
        
        for pI in np.arange(frame_range_start, frame_range_end - sequence_length - 1):

            sequence_excerpt_dancer1 = pose_sequence_dancer1[pI:pI+sequence_length]
            dancer1_data.append(sequence_excerpt_dancer1)
            
            sequence_excerpt_dancer2 = pose_sequence_dancer2[pI:pI+sequence_length]
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

# create models

# create encoder model

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
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x 3 ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " std s ", std.shape)
    
        return mu, std
    
encoder = Encoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_rnn_bidirectional, ae_dense_layer_sizes).to(device)

print(encoder)

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    
# test encoder

encoder_test_input1, encoder_test_input2 = next(iter(train_loader))

encoder_test_input1 = encoder_test_input1.to(device)
encoder_test_input2 = encoder_test_input2.to(device)

encoder_test_output_mu1, encoder_test_output_std1 = encoder(encoder_test_input1)
encoder_test_output_mu2, encoder_test_output_std2 = encoder(encoder_test_input2)

print("encoder_test_input1 s ", encoder_test_input1.shape)
print("encoder_test_input2 s ", encoder_test_input2.shape)
print("encoder_test_output_mu1 s ", encoder_test_output_mu1.shape)
print("encoder_test_output_mu2 s ", encoder_test_output_mu2.shape)
print("encoder_test_output_std1 s ", encoder_test_output_std1.shape)
print("encoder_test_output_std2 s ", encoder_test_output_std2.shape)
    
# create decoder model

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
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        dense_input_dim = self.rnn_layer_size if self.rnn_bidirectional == False else self.rnn_layer_size  * 2

        x_reshaped = x.contiguous().view(-1, dense_input_dim)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        #print("yhat 2 ", yhat.size())

        return yhat

ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()

decoder1 = Decoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_rnn_bidirectional, ae_dense_layer_sizes_reversed).to(device)
decoder2 = Decoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_rnn_bidirectional, ae_dense_layer_sizes_reversed).to(device)

print(decoder1)

if load_weights and decoder1_weights_file:
    decoder1.load_state_dict(torch.load(decoder1_weights_file, map_location=device))
if load_weights and decoder2_weights_file:
    decoder2.load_state_dict(torch.load(decoder2_weights_file, map_location=device))

# test decoder

decoder1_test_input = encoder_test_output_mu1
decoder2_test_input = encoder_test_output_mu2

decoder1_test_output = decoder1(decoder1_test_input)
decoder2_test_output = decoder1(decoder2_test_input)

print("decoder1_test_input s ", decoder1_test_input.shape)
print("decoder2_test_input s ", decoder2_test_input.shape)
print("decoder1_test_output s ", decoder1_test_output.shape)
print("decoder2_test_output s ", decoder2_test_output.shape)
    
# Training

def calc_kld_scales():
    
    kld_scales = []

    for e in range(epochs):
        
        cycle_step = e % kld_scale_cycle_duration
        
        #print("cycle_step ", cycle_step)


        if cycle_step < kld_scale_min_const_duration:
            kld_scale = min_kld_scale
            kld_scales.append(kld_scale)
        elif cycle_step > kld_scale_cycle_duration - kld_scale_max_const_duration:
            kld_scale = max_kld_scale
            kld_scales.append(kld_scale)
        else:
            lin_step = cycle_step - kld_scale_min_const_duration
            kld_scale = min_kld_scale + (max_kld_scale - min_kld_scale) * lin_step / (kld_scale_cycle_duration - kld_scale_min_const_duration - kld_scale_max_const_duration)
            kld_scales.append(kld_scale)
            
    return kld_scales

kld_scales = calc_kld_scales()

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()), lr=ae_learning_rate)
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=100, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# KL Divergence

def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl
   
def variational_loss2(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #alternative: mean squared distance from ideal mu=0 and std=1:
    vl=torch.mean(mu.pow(2)+(1-std).pow(2))
    return vl

def reparameterize(mu, std):
    z = mu + std*torch.randn_like(std)
    return z

# joint loss weights

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

# function returning normal distributed random data 
# serves as reference for the discriminator to distinguish the encoders prior from
def sample_normal(shape):
    return torch.tensor(np.random.normal(size=shape), dtype=torch.float32).to(device)

# discriminator prior loss function
def disc_prior_loss(disc_real_output, disc_fake_output):
    ones = torch.ones_like(disc_real_output).to(device)
    zeros = torch.zeros_like(disc_fake_output).to(device)

    real_loss = cross_entropy(disc_real_output, ones)
    fake_loss = cross_entropy(disc_fake_output, zeros)

    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

def ae_norm_loss(yhat):
    
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

def ae_pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim

    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))

    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def ae_quat_loss(y, yhat):
    _y_rot = y.view(-1, 4)
    _yhat_rot = yhat.view(-1, 4)
    
    _yhat_norm = nn.functional.normalize(_yhat_rot, p=2, dim=1)
    
    _loss = mse_loss(_yhat_norm, _y_rot)

    return _loss

"""
def ae_quat_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))

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
    _loss = torch.mean(_abs)   
    return _loss
"""

def ae_quat_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    # normalize quaternion
    
    _y = y.view((-1, 4))
    _yhat = yhat.view((-1, 4))

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
    
    _abs = _abs.reshape(-1, sequence_length, joint_count)
    
    _abs_weighted = _abs * joint_loss_weights
    
    #print("_abs s ", _abs.shape)
    
    _loss = torch.mean(_abs_weighted)   
    return _loss


# autoencoder loss function
def ae_loss(y, yhat, mu, std):
    # function parameters
    # y: encoder input
    # yhat: decoder output (i.e. reconstructed encoder input)
    # disc_fake_output: discriminator output for encoder generated prior
    
    _norm_loss = ae_norm_loss(yhat)
    _pos_loss = ae_pos_loss(y, yhat)
    _quat_loss = ae_quat_loss(y, yhat)

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * ae_norm_loss_scale
    _total_loss += _pos_loss * ae_pos_loss_scale
    _total_loss += _quat_loss * ae_quat_loss_scale
    _total_loss += _ae_kld_loss * ae_kld_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _quat_loss, _ae_kld_loss

def ae_train_step(target_poses1, target_poses2):
    
    #print("train step target_poses ", target_poses.shape)
 
    encoder_output1 = encoder(target_poses1)
    encoder_output1_mu = encoder_output1[0]
    encoder_output1_std = encoder_output1[1]
    mu1 = torch.tanh(encoder_output1_mu)
    std1 = torch.abs(torch.tanh(encoder_output1_std)) + 0.00001
    decoder1_input = reparameterize(mu1, std1)
    pred_poses1 = decoder1(decoder1_input)
    
    encoder_output2 = encoder(target_poses2)
    encoder_output2_mu = encoder_output2[0]
    encoder_output2_std = encoder_output2[1]
    mu2 = torch.tanh(encoder_output2_mu)
    std2 = torch.abs(torch.tanh(encoder_output2_std)) + 0.00001
    decoder2_input = reparameterize(mu2, std2)
    pred_poses2 = decoder2(decoder2_input)
    
    _ae_loss1, _ae_norm_loss1, _ae_pos_loss1, _ae_quat_loss1, _ae_kld_loss1 = ae_loss(target_poses1, pred_poses1, mu1, std1) 
    _ae_loss2, _ae_norm_loss2, _ae_pos_loss2, _ae_quat_loss2, _ae_kld_loss2 = ae_loss(target_poses2, pred_poses2, mu2, std2) 

    _ae_loss = _ae_loss1 + _ae_loss2
    _ae_norm_loss = _ae_norm_loss1 + _ae_norm_loss2
    _ae_pos_loss = _ae_pos_loss1 + _ae_pos_loss2
    _ae_quat_loss = _ae_quat_loss1 + _ae_quat_loss2
    _ae_kld_loss = _ae_kld_loss1 + _ae_kld_loss2
    
    #print("_ae_pos_loss ", _ae_pos_loss)
    
    # Backpropagation
    ae_optimizer.zero_grad()
    _ae_loss.backward()

    ae_optimizer.step()
    
    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss

def ae_test_step(target_poses1, target_poses2):
    
    #print("train step target_poses ", target_poses.shape)
    with torch.no_grad():
        encoder_output1 = encoder(target_poses1)
        encoder_output1_mu = encoder_output1[0]
        encoder_output1_std = encoder_output1[1]
        mu1 = torch.tanh(encoder_output1_mu)
        std1 = torch.abs(torch.tanh(encoder_output1_std)) + 0.00001
        decoder1_input = reparameterize(mu1, std1)
        pred_poses1 = decoder1(decoder1_input)
        
        encoder_output2 = encoder(target_poses2)
        encoder_output2_mu = encoder_output2[0]
        encoder_output2_std = encoder_output2[1]
        mu2 = torch.tanh(encoder_output2_mu)
        std2 = torch.abs(torch.tanh(encoder_output2_std)) + 0.00001
        decoder2_input = reparameterize(mu2, std2)
        pred_poses2 = decoder2(decoder2_input)
        
        _ae_loss1, _ae_norm_loss1, _ae_pos_loss1, _ae_quat_loss1, _ae_kld_loss1 = ae_loss(target_poses1, pred_poses1, mu1, std1) 
        _ae_loss2, _ae_norm_loss2, _ae_pos_loss2, _ae_quat_loss2, _ae_kld_loss2 = ae_loss(target_poses2, pred_poses2, mu2, std2) 
    
        _ae_loss = _ae_loss1 + _ae_loss2
        _ae_norm_loss = _ae_norm_loss1 + _ae_norm_loss2
        _ae_pos_loss = _ae_pos_loss1 + _ae_pos_loss2
        _ae_quat_loss = _ae_quat_loss1 + _ae_quat_loss2
        _ae_kld_loss = _ae_kld_loss1 + _ae_kld_loss2

    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss

def train(train_dataloader, test_dataloader, epochs):
    
    global ae_kld_loss_scale
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae norm"] = []
    loss_history["ae pos"] = []
    loss_history["ae quat"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_kld_loss_scale = kld_scales[epoch]
        
        #print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        ae_train_loss_per_epoch = []
        ae_norm_loss_per_epoch = []
        ae_pos_loss_per_epoch = []
        ae_quat_loss_per_epoch = []
        ae_prior_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for target_poses1, target_poses2 in train_dataloader:
            
            target_poses1 = target_poses1.to(device)
            target_poses2 = target_poses2.to(device)
            
            _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss = ae_train_step(target_poses1, target_poses2)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_norm_loss = _ae_norm_loss.detach().cpu().numpy()
            _ae_pos_loss = _ae_pos_loss.detach().cpu().numpy()
            _ae_quat_loss = _ae_quat_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_norm_loss_per_epoch.append(_ae_norm_loss)
            ae_pos_loss_per_epoch.append(_ae_pos_loss)
            ae_quat_loss_per_epoch.append(_ae_quat_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_norm_loss_per_epoch = np.mean(np.array(ae_norm_loss_per_epoch))
        ae_pos_loss_per_epoch = np.mean(np.array(ae_pos_loss_per_epoch))
        ae_quat_loss_per_epoch = np.mean(np.array(ae_quat_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))

        ae_test_loss_per_epoch = []
        
        for target_poses1, target_poses2 in test_dataloader:
            
            target_poses1 = target_poses1.to(device)
            target_poses2 = target_poses2.to(device)
            
            _ae_loss, _, _, _, _ = ae_test_step(target_poses1, target_poses2)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            ae_test_loss_per_epoch.append(_ae_loss)
        
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder1.state_dict(), "results/weights/decoder1_weights_epoch_{}".format(epoch))
            torch.save(decoder2.state_dict(), "results/weights/decoder2_weights_epoch_{}".format(epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae norm"].append(ae_norm_loss_per_epoch)
        loss_history["ae pos"].append(ae_pos_loss_per_epoch)
        loss_history["ae quat"].append(ae_quat_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} norm {:01.4f} pos {:01.4f} quat {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, ae_norm_loss_per_epoch, ae_pos_loss_per_epoch, ae_quat_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        ae_scheduler.step()
        
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(decoder1.state_dict(), "results/weights/decoder1_weights_epoch_{}".format(epochs))
torch.save(decoder2.state_dict(), "results/weights/decoder2_weights_epoch_{}".format(epochs))

# inference and rendering 

poseRenderer = PoseRenderer(edge_list)

def create_ref_sequence_anim(mocap_index, pose_index, pose_count, file_name1, file_name2):
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]
    
    seq_excerpt_dancer1 = torch.tensor(np.expand_dims(seq_excerpt_dancer1, axis=0)).to(torch.float32).to(device)
    seq_excerpt_dancer2 = torch.tensor(np.expand_dims(seq_excerpt_dancer2, axis=0)).to(torch.float32).to(device)

    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence_dancer1 = forward_kinematics(seq_excerpt_dancer1, zero_trajectory)
    skel_sequence_dancer2 = forward_kinematics(seq_excerpt_dancer2, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer1[0].save(file_name1, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0)

    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer2[0].save(file_name2, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0)

def create_pred_sequence_anim_dancer1_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer1 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer1 = seq_excerpt_dancer1[pI:pI+sequence_length]
            encoder_input_dancer1 = torch.from_numpy(encoder_input_dancer1).to(torch.float32).to(device)
            encoder_input_dancer1 = torch.reshape(encoder_input_dancer1, (1, sequence_length, pose_dim))

            encoder_output_dancer1 = encoder(encoder_input_dancer1)
            encoder_output_mu_dancer1 = encoder_output_dancer1[0]
            encoder_output_std_dancer1 = encoder_output_dancer1[1]
            mu_dancer1 = torch.tanh(encoder_output_mu_dancer1)
            std_dancer1 = torch.abs(torch.tanh(encoder_output_std_dancer1)) + 0.00001
        
            decoder_input_dancer1 = reparameterize(mu_dancer1, std_dancer1)

            pred_seq_dancer1 = decoder1(decoder_input_dancer1)

            # normalize pred seq
            pred_seq_dancer1 = torch.squeeze(pred_seq_dancer1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((-1, 4))
            pred_seq_dancer1 = nn.functional.normalize(pred_seq_dancer1, p=2, dim=1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer1 = pred_seq_dancer1.detach().cpu().numpy()
            pred_seq_dancer1 = np.reshape(pred_seq_dancer1, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
                for ji in range(joint_count): 
                    current_quat = gen_sequence_dancer1[pI + si, ji, :]
                    target_quat = pred_seq_dancer1[si, ji, :]
                    quat_mix = seq_env[si]
                    mix_quat = slerp(current_quat, target_quat, quat_mix )
                    gen_sequence_dancer1[pI + si, ji, :] = mix_quat
            
    # fix quaternions in gen sequence
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((-1, 4))
    gen_sequence_dancer1 = gen_sequence_dancer1 / np.linalg.norm(gen_sequence_dancer1, ord=2, axis=1, keepdims=True)
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((pose_count, joint_count, joint_dim))
    gen_sequence_dancer1 = qfix(gen_sequence_dancer1)
    gen_sequence_dancer1 = np.expand_dims(gen_sequence_dancer1, axis=0)
    gen_sequence_dancer1 = torch.from_numpy(gen_sequence_dancer1).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer1 = forward_kinematics(gen_sequence_dancer1, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer1[0].save(file_name, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0) 

def create_pred_sequence_anim_dancer2_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer2 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer2 = seq_excerpt_dancer2[pI:pI+sequence_length]
            encoder_input_dancer2 = torch.from_numpy(encoder_input_dancer2).to(torch.float32).to(device)
            encoder_input_dancer2 = torch.reshape(encoder_input_dancer2, (1, sequence_length, pose_dim))

            encoder_output_dancer2 = encoder(encoder_input_dancer2)
            encoder_output_mu_dancer2 = encoder_output_dancer2[0]
            encoder_output_std_dancer2 = encoder_output_dancer2[1]
            mu_dancer2 = torch.tanh(encoder_output_mu_dancer2)
            std_dancer2 = torch.abs(torch.tanh(encoder_output_std_dancer2)) + 0.00001
        
            decoder_input_dancer2 = reparameterize(mu_dancer2, std_dancer2)

            pred_seq_dancer2 = decoder2(decoder_input_dancer2)

            # normalize pred seq
            pred_seq_dancer2 = torch.squeeze(pred_seq_dancer2)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((-1, 4))
            pred_seq_dancer2 = nn.functional.normalize(pred_seq_dancer2, p=2, dim=1)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer2 = pred_seq_dancer2.detach().cpu().numpy()
            pred_seq_dancer2 = np.reshape(pred_seq_dancer2, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
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
    gen_sequence_dancer2 = np.expand_dims(gen_sequence_dancer2, axis=0)
    gen_sequence_dancer2 = torch.from_numpy(gen_sequence_dancer2).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer2 = forward_kinematics(gen_sequence_dancer2, zero_trajectory)
    
    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer2[0].save(file_name, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0) 

def create_pred_sequence_anim_dancer1_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer1 = all_mocap_data_dancer1[mocap_index]
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer2 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer1 = seq_excerpt_dancer1[pI:pI+sequence_length]
            encoder_input_dancer1 = torch.from_numpy(encoder_input_dancer1).to(torch.float32).to(device)
            encoder_input_dancer1 = torch.reshape(encoder_input_dancer1, (1, sequence_length, pose_dim))

            encoder_output_dancer1 = encoder(encoder_input_dancer1)
            encoder_output_mu_dancer1 = encoder_output_dancer1[0]
            encoder_output_std_dancer1 = encoder_output_dancer1[1]
            mu_dancer1 = torch.tanh(encoder_output_mu_dancer1)
            std_dancer1 = torch.abs(torch.tanh(encoder_output_std_dancer1)) + 0.00001
        
            decoder_input_dancer1 = reparameterize(mu_dancer1, std_dancer1)

            pred_seq_dancer2 = decoder2(decoder_input_dancer1)

            # normalize pred seq
            pred_seq_dancer2 = torch.squeeze(pred_seq_dancer2)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((-1, 4))
            pred_seq_dancer2 = nn.functional.normalize(pred_seq_dancer2, p=2, dim=1)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer2 = pred_seq_dancer2.detach().cpu().numpy()
            pred_seq_dancer2 = np.reshape(pred_seq_dancer2, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
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
    gen_sequence_dancer2 = np.expand_dims(gen_sequence_dancer2, axis=0)
    gen_sequence_dancer2 = torch.from_numpy(gen_sequence_dancer2).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer2 = forward_kinematics(gen_sequence_dancer2, zero_trajectory)
    
    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer2[0].save(file_name, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0) 

def create_pred_sequence_anim_dancer2_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer2 = all_mocap_data_dancer2[mocap_index]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer1 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer2 = seq_excerpt_dancer2[pI:pI+sequence_length]
            encoder_input_dancer2 = torch.from_numpy(encoder_input_dancer2).to(torch.float32).to(device)
            encoder_input_dancer2 = torch.reshape(encoder_input_dancer2, (1, sequence_length, pose_dim))

            encoder_output_dancer2 = encoder(encoder_input_dancer2)
            encoder_output_mu_dancer2 = encoder_output_dancer2[0]
            encoder_output_std_dancer2 = encoder_output_dancer2[1]
            mu_dancer2 = torch.tanh(encoder_output_mu_dancer2)
            std_dancer2 = torch.abs(torch.tanh(encoder_output_std_dancer2)) + 0.00001
        
            decoder_input_dancer2 = reparameterize(mu_dancer2, std_dancer2)

            pred_seq_dancer1 = decoder1(decoder_input_dancer2)

            # normalize pred seq
            pred_seq_dancer1 = torch.squeeze(pred_seq_dancer1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((-1, 4))
            pred_seq_dancer1 = nn.functional.normalize(pred_seq_dancer1, p=2, dim=1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer1 = pred_seq_dancer1.detach().cpu().numpy()
            pred_seq_dancer1 = np.reshape(pred_seq_dancer1, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
                for ji in range(joint_count): 
                    current_quat = gen_sequence_dancer1[pI + si, ji, :]
                    target_quat = pred_seq_dancer1[si, ji, :]
                    quat_mix = seq_env[si]
                    mix_quat = slerp(current_quat, target_quat, quat_mix )
                    gen_sequence_dancer1[pI + si, ji, :] = mix_quat
            
    # fix quaternions in gen sequence
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((-1, 4))
    gen_sequence_dancer1 = gen_sequence_dancer1 / np.linalg.norm(gen_sequence_dancer1, ord=2, axis=1, keepdims=True)
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((pose_count, joint_count, joint_dim))
    gen_sequence_dancer1 = qfix(gen_sequence_dancer1)
    gen_sequence_dancer1 = np.expand_dims(gen_sequence_dancer1, axis=0)
    gen_sequence_dancer1 = torch.from_numpy(gen_sequence_dancer1).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer1 = forward_kinematics(gen_sequence_dancer1, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer1[0].save(file_name, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0) 


mocap_index = 0
pose_index = 1000
pose_count = 1000

create_ref_sequence_anim(mocap_index, pose_index, pose_count, "results/anims/ref_dancer1_mocap_{}_poses_{}-{}.gif".format(mocap_index, pose_index, (pose_index + pose_count)), "results/anims/ref_dancer2_ocap_{}_poses_{}-{}.gif".format(mocap_index, pose_index, (pose_index + pose_count)))


base_pose = all_mocap_data_dancer1[0]["motion"]["rot_local"][0]
pose_offset = 16 # 2 for 8, 32 for 128


create_pred_sequence_anim_dancer1_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, "results/anims/pred_dancer1_to_dancer1_mocap_{}_poses_{}-{}_epoch_{}.gif".format(mocap_index, pose_index, (pose_index + pose_count), epochs))
create_pred_sequence_anim_dancer2_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, "results/anims/pred_dancer2_to_dancer2_mocap_{}_poses_{}-{}_epoch_{}.gif".format(mocap_index, pose_index, (pose_index + pose_count), epochs))
create_pred_sequence_anim_dancer1_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, "results/anims/pred_dancer1_to_dancer2_mocap_{}_poses_{}-{}_epoch_{}.gif".format(mocap_index, pose_index, (pose_index + pose_count), epochs))
create_pred_sequence_anim_dancer2_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, "results/anims/pred_dancer2_to_dancer1_mocap_{}_poses_{}-{}_epoch_{}.gif".format(mocap_index, pose_index, (pose_index + pose_count), epochs))

# inference with new mocap file (not from training set)

new_mocap_file_path = "E:/Data/mocap/stocos/Duets/Amsterdam_2024/bvh_50hz"
new_mocap_files = [ [ "Recording3_JS-001_jason.bvh", "Recording3_JS-001_sherise.bvh" ] ]
new_mocap_valid_frame_ranges = [ [ [ 500, 30800] ] ]

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

def create_new_ref_sequence_anim(mocap_index, pose_index, pose_count, file_name1, file_name2):
    
    mocap_data_dancer1 = new_mocap_data_dancer1[mocap_index]
    mocap_data_dancer2 = new_mocap_data_dancer2[mocap_index]
    
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]
    
    seq_excerpt_dancer1 = torch.tensor(np.expand_dims(seq_excerpt_dancer1, axis=0)).to(torch.float32).to(device)
    seq_excerpt_dancer2 = torch.tensor(np.expand_dims(seq_excerpt_dancer2, axis=0)).to(torch.float32).to(device)

    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence_dancer1 = forward_kinematics(seq_excerpt_dancer1, zero_trajectory)
    skel_sequence_dancer2 = forward_kinematics(seq_excerpt_dancer2, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer1[0].save(file_name1, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0)

    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images_dancer2[0].save(file_name2, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0)

def create_new_pred_sequence_anim_dancer1_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer1 = new_mocap_data_dancer1[mocap_index]
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer1 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer1 = seq_excerpt_dancer1[pI:pI+sequence_length]
            encoder_input_dancer1 = torch.from_numpy(encoder_input_dancer1).to(torch.float32).to(device)
            encoder_input_dancer1 = torch.reshape(encoder_input_dancer1, (1, sequence_length, pose_dim))

            encoder_output_dancer1 = encoder(encoder_input_dancer1)
            encoder_output_mu_dancer1 = encoder_output_dancer1[0]
            encoder_output_std_dancer1 = encoder_output_dancer1[1]
            mu_dancer1 = torch.tanh(encoder_output_mu_dancer1)
            std_dancer1 = torch.abs(torch.tanh(encoder_output_std_dancer1)) + 0.00001
        
            decoder_input_dancer1 = reparameterize(mu_dancer1, std_dancer1)

            pred_seq_dancer1 = decoder1(decoder_input_dancer1)

            # normalize pred seq
            pred_seq_dancer1 = torch.squeeze(pred_seq_dancer1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((-1, 4))
            pred_seq_dancer1 = nn.functional.normalize(pred_seq_dancer1, p=2, dim=1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer1 = pred_seq_dancer1.detach().cpu().numpy()
            pred_seq_dancer1 = np.reshape(pred_seq_dancer1, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
                for ji in range(joint_count): 
                    current_quat = gen_sequence_dancer1[pI + si, ji, :]
                    target_quat = pred_seq_dancer1[si, ji, :]
                    quat_mix = seq_env[si]
                    mix_quat = slerp(current_quat, target_quat, quat_mix )
                    gen_sequence_dancer1[pI + si, ji, :] = mix_quat
            
    # fix quaternions in gen sequence
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((-1, 4))
    gen_sequence_dancer1 = gen_sequence_dancer1 / np.linalg.norm(gen_sequence_dancer1, ord=2, axis=1, keepdims=True)
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((pose_count, joint_count, joint_dim))
    gen_sequence_dancer1 = qfix(gen_sequence_dancer1)
    gen_sequence_dancer1 = np.expand_dims(gen_sequence_dancer1, axis=0)
    gen_sequence_dancer1 = torch.from_numpy(gen_sequence_dancer1).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer1 = forward_kinematics(gen_sequence_dancer1, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer1[0].save(file_name, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0) 

def create_new_pred_sequence_anim_dancer2_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer2 = new_mocap_data_dancer2[mocap_index]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer2 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer2 = seq_excerpt_dancer2[pI:pI+sequence_length]
            encoder_input_dancer2 = torch.from_numpy(encoder_input_dancer2).to(torch.float32).to(device)
            encoder_input_dancer2 = torch.reshape(encoder_input_dancer2, (1, sequence_length, pose_dim))

            encoder_output_dancer2 = encoder(encoder_input_dancer2)
            encoder_output_mu_dancer2 = encoder_output_dancer2[0]
            encoder_output_std_dancer2 = encoder_output_dancer2[1]
            mu_dancer2 = torch.tanh(encoder_output_mu_dancer2)
            std_dancer2 = torch.abs(torch.tanh(encoder_output_std_dancer2)) + 0.00001
        
            decoder_input_dancer2 = reparameterize(mu_dancer2, std_dancer2)

            pred_seq_dancer2 = decoder2(decoder_input_dancer2)

            # normalize pred seq
            pred_seq_dancer2 = torch.squeeze(pred_seq_dancer2)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((-1, 4))
            pred_seq_dancer2 = nn.functional.normalize(pred_seq_dancer2, p=2, dim=1)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer2 = pred_seq_dancer2.detach().cpu().numpy()
            pred_seq_dancer2 = np.reshape(pred_seq_dancer2, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
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
    gen_sequence_dancer2 = np.expand_dims(gen_sequence_dancer2, axis=0)
    gen_sequence_dancer2 = torch.from_numpy(gen_sequence_dancer2).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer2 = forward_kinematics(gen_sequence_dancer2, zero_trajectory)
    
    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer2[0].save(file_name, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0) 

def create_new_pred_sequence_anim_dancer1_to_dancer2(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer1 = new_mocap_data_dancer1[mocap_index]
    pose_seq_dancer1 = mocap_data_dancer1["motion"]["rot_local"]
    seq_excerpt_dancer1 = pose_seq_dancer1[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer2 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer1 = seq_excerpt_dancer1[pI:pI+sequence_length]
            encoder_input_dancer1 = torch.from_numpy(encoder_input_dancer1).to(torch.float32).to(device)
            encoder_input_dancer1 = torch.reshape(encoder_input_dancer1, (1, sequence_length, pose_dim))

            encoder_output_dancer1 = encoder(encoder_input_dancer1)
            encoder_output_mu_dancer1 = encoder_output_dancer1[0]
            encoder_output_std_dancer1 = encoder_output_dancer1[1]
            mu_dancer1 = torch.tanh(encoder_output_mu_dancer1)
            std_dancer1 = torch.abs(torch.tanh(encoder_output_std_dancer1)) + 0.00001
        
            decoder_input_dancer1 = reparameterize(mu_dancer1, std_dancer1)

            pred_seq_dancer2 = decoder2(decoder_input_dancer1)

            # normalize pred seq
            pred_seq_dancer2 = torch.squeeze(pred_seq_dancer2)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((-1, 4))
            pred_seq_dancer2 = nn.functional.normalize(pred_seq_dancer2, p=2, dim=1)
            pred_seq_dancer2 = pred_seq_dancer2.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer2 = pred_seq_dancer2.detach().cpu().numpy()
            pred_seq_dancer2 = np.reshape(pred_seq_dancer2, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
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
    gen_sequence_dancer2 = np.expand_dims(gen_sequence_dancer2, axis=0)
    gen_sequence_dancer2 = torch.from_numpy(gen_sequence_dancer2).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer2 = forward_kinematics(gen_sequence_dancer2, zero_trajectory)
    
    skel_sequence_dancer2 = skel_sequence_dancer2.detach().cpu().numpy()
    skel_sequence_dancer2 = np.squeeze(skel_sequence_dancer2)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer2)
    skel_images_dancer2 = poseRenderer.create_pose_images(skel_sequence_dancer2, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer2[0].save(file_name, save_all=True, append_images=skel_images_dancer2[1:], optimize=False, duration=33.0, loop=0) 

def create_new_pred_sequence_anim_dancer2_to_dancer1(mocap_index, pose_index, pose_count, pose_offset, base_pose, file_name):
    
    seq_env = np.hanning(sequence_length)
    
    mocap_data_dancer2 = new_mocap_data_dancer2[mocap_index]
    pose_seq_dancer2 = mocap_data_dancer2["motion"]["rot_local"]
    seq_excerpt_dancer2 = pose_seq_dancer2[pose_index:pose_index+pose_count, ...]

    gen_sequence_dancer1 = np.full(shape=(pose_count, joint_count, joint_dim), fill_value=base_pose)
    
    for pI in range(0, pose_count - sequence_length, pose_offset):
        
        print("pI ", pI, " out of ", (pose_count - sequence_length))

        with torch.no_grad():

            encoder_input_dancer2 = seq_excerpt_dancer2[pI:pI+sequence_length]
            encoder_input_dancer2 = torch.from_numpy(encoder_input_dancer2).to(torch.float32).to(device)
            encoder_input_dancer2 = torch.reshape(encoder_input_dancer2, (1, sequence_length, pose_dim))

            encoder_output_dancer2 = encoder(encoder_input_dancer2)
            encoder_output_mu_dancer2 = encoder_output_dancer2[0]
            encoder_output_std_dancer2 = encoder_output_dancer2[1]
            mu_dancer2 = torch.tanh(encoder_output_mu_dancer2)
            std_dancer2 = torch.abs(torch.tanh(encoder_output_std_dancer2)) + 0.00001
        
            decoder_input_dancer2 = reparameterize(mu_dancer2, std_dancer2)

            pred_seq_dancer1 = decoder1(decoder_input_dancer2)

            # normalize pred seq
            pred_seq_dancer1 = torch.squeeze(pred_seq_dancer1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((-1, 4))
            pred_seq_dancer1 = nn.functional.normalize(pred_seq_dancer1, p=2, dim=1)
            pred_seq_dancer1 = pred_seq_dancer1.reshape((sequence_length, pose_dim))
            
            # blend pred seq into gen seq
            pred_seq_dancer1 = pred_seq_dancer1.detach().cpu().numpy()
            pred_seq_dancer1 = np.reshape(pred_seq_dancer1, (-1, joint_count, joint_dim))
            
            for si in range(sequence_length):
                for ji in range(joint_count): 
                    current_quat = gen_sequence_dancer1[pI + si, ji, :]
                    target_quat = pred_seq_dancer1[si, ji, :]
                    quat_mix = seq_env[si]
                    mix_quat = slerp(current_quat, target_quat, quat_mix )
                    gen_sequence_dancer1[pI + si, ji, :] = mix_quat
            
    # fix quaternions in gen sequence
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((-1, 4))
    gen_sequence_dancer1 = gen_sequence_dancer1 / np.linalg.norm(gen_sequence_dancer1, ord=2, axis=1, keepdims=True)
    gen_sequence_dancer1 = gen_sequence_dancer1.reshape((pose_count, joint_count, joint_dim))
    gen_sequence_dancer1 = qfix(gen_sequence_dancer1)
    gen_sequence_dancer1 = np.expand_dims(gen_sequence_dancer1, axis=0)
    gen_sequence_dancer1 = torch.from_numpy(gen_sequence_dancer1).to(torch.float32).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32))
    zero_trajectory = zero_trajectory.to(device)
    
    
    skel_sequence_dancer1 = forward_kinematics(gen_sequence_dancer1, zero_trajectory)
    
    skel_sequence_dancer1 = skel_sequence_dancer1.detach().cpu().numpy()
    skel_sequence_dancer1 = np.squeeze(skel_sequence_dancer1)
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence_dancer1)
    skel_images_dancer1 = poseRenderer.create_pose_images(skel_sequence_dancer1, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)

    skel_images_dancer1[0].save(file_name, save_all=True, append_images=skel_images_dancer1[1:], optimize=False, duration=33.0, loop=0) 


new_mocap_index = 0
new_pose_index = 1000
new_pose_count = 1000

create_new_ref_sequence_anim(new_mocap_index, new_pose_index, new_pose_count, "results/anims/ref_dancer1_new_mocap_{}_poses_{}-{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count)), "results/anims/ref_dancer2_new_mocap_{}_poses_{}-{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count)))


create_new_pred_sequence_anim_dancer1_to_dancer1(new_mocap_index, new_pose_index, new_pose_count, pose_offset, base_pose, "results/anims/pred_dancer1_to_dancer1_new_mocap_{}_poses_{}-{}_epoch_{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count), epochs))
create_new_pred_sequence_anim_dancer2_to_dancer2(new_mocap_index, new_pose_index, new_pose_count, pose_offset, base_pose, "results/anims/pred_dancer2_to_dancer2_new_mocap_{}_poses_{}-{}_epoch_{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count), epochs))
create_new_pred_sequence_anim_dancer1_to_dancer2(new_mocap_index, new_pose_index, new_pose_count, pose_offset, base_pose, "results/anims/pred_dancer1_to_dancer2_new_mocap_{}_poses_{}-{}_epoch_{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count), epochs))
create_new_pred_sequence_anim_dancer2_to_dancer1(new_mocap_index, new_pose_index, new_pose_count, pose_offset, base_pose, "results/anims/pred_dancer2_to_dancer1_new_mocap_{}_poses_{}-{}_epoch_{}.gif".format(new_mocap_index, new_pose_index, (new_pose_index + new_pose_count), epochs))
