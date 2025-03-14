"""
motion to motion translation
using a variational autoencoder
"""

import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control

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
import pickle
from time import sleep

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
Mocap and Training Settings
"""

mocap_file_path = "data/mocap"
mocap_files = [ [ "Jason_Take4.fbx", "Sherise_Take4.fbx" ] ]
mocap_pos_scale = 1.0
mocap_fps = 50

encoder_weights_file = "data/results/weights/encoder_weights_epoch_600"
decoder1_weights_file = "data/results/weights/decoder1_weights_epoch_600"
decoder2_weights_file = "data/results/weights/decoder2_weights_epoch_600"


"""
mocap_file_path = "../../../Data/Mocap/XSens/Stocos/Duets/fbx_50hz"
mocap_files = [ [ "Jason_Take4.fbx", "Sherise_Take4.fbx" ] ]
mocap_pos_scale = 1.0
mocap_fps = 50

encoder_weights_file = "../../../Data/Models/MotionDuet/vae-rnn/results_vae_deepfake_XSens_SheriseJason_Take4/weights/encoder_weights_epoch_600"
decoder1_weights_file = "../../../Data/Models/MotionDuet/vae-rnn/results_vae_deepfake_XSens_SheriseJason_Take4/weights/decoder1_weights_epoch_600"
decoder2_weights_file = "../../../Data/Models/MotionDuet/vae-rnn/results_vae_deepfake_XSens_SheriseJason_Take4/weights/decoder2_weights_epoch_600"
"""

"""
Model Settings
"""

seq_length = 64
latent_dim = 32
ae_rnn_layer_count = 2
ae_rnn_layer_size = 256
ae_rnn_bidirectional = True
ae_dense_layer_sizes = [ 512  ]


"""
OSC Settings
"""

osc_send_ip = "127.0.0.1"
osc_send_port = 9004

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9002



"""
Load Mocap Data
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
    
all_pose_sequences_dancer1 = []
all_pose_sequences_dancer2 = []

for mocap_data_dancer1, mocap_data_dancer2 in zip(all_mocap_data_dancer1, all_mocap_data_dancer2):
    
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences_dancer1.append(pose_sequence_dancer1)
    
    pose_sequence_dancer2 = mocap_data_dancer2["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences_dancer2.append(pose_sequence_dancer2)

joint_count = all_pose_sequences_dancer1[0].shape[1]
joint_dim = all_pose_sequences_dancer1[0].shape[2]
pose_dim = joint_count * joint_dim

"""
Load Model
"""

motion_model.config = {
    "seq_length": seq_length,
    "pose_dim": pose_dim,
    "latent_dim": latent_dim,
    "ae_rnn_layer_count": ae_rnn_layer_count,
    "ae_rnn_layer_size": ae_rnn_layer_size,
    "ae_rnn_bidirectional": ae_rnn_bidirectional,
    "ae_dense_layer_sizes": ae_dense_layer_sizes,
    "device": device,
    "encoder_weights_path": encoder_weights_file,
    "decoder1_weights_path": decoder1_weights_file,
    "decoder2_weights_path": decoder2_weights_file
    }

encoder, decoder1, decoder2 = motion_model.createModel(motion_model.config) 


"""
Setup Motion Synthesis
"""

seq_overlap = 48

motion_synthesis.config = {
    "skeleton": all_mocap_data_dancer1[0]["skeleton"],
    "model_encoder": encoder,
    "model_decoder1": decoder1,
    "model_decoder2": decoder2,
    "device": device,
    "seq_window_length": seq_length,
    "seq_window_overlap": seq_overlap,
    "orig_sequences1": all_pose_sequences_dancer1,
    "orig_sequences2": all_pose_sequences_dancer2,
    "orig_seq1_index": 0,
    "orig_seq2_index": 0,
    }

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)


"""
OSC Sender
"""

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender
motion_gui.config["update_interval"] = 1.0 / mocap_fps

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
"""

motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()


osc_control.stop()
