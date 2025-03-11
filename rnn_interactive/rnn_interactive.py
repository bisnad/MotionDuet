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

# Example: XSens Mocap Recording
mocap_file_path = "../../../Data/Mocap/XSens/Stocos/Duets/fbx_50hz"
mocap_files = [ "Jason_Take4.fbx" ]
mocap_valid_frame_ranges = [ [ [ 490, 30679] ] ]
mocap_pos_scale = 1.0
mocap_fps = 50

"""
Model Settings
"""

sequence_length = 64
rnn_layer_dim = 512
rnn_layer_count = 2

"""
Training Settings
"""

# Example: XSens Mocap Recording
rnn_weights_file = "../rnn/results_XSens_SheriseJason_Take4/weights/rnn_weights_epoch_200"

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

for mocap_file_dancer1 in mocap_files:
    
    print("process file ", mocap_file_dancer1)
    
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

all_pose_sequences_dancer1 = []

for mocap_data_dancer1 in all_mocap_data_dancer1:
    
    pose_sequence_dancer1 = mocap_data_dancer1["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences_dancer1.append(pose_sequence_dancer1)

joint_count = all_pose_sequences_dancer1[0].shape[1]
joint_dim = all_pose_sequences_dancer1[0].shape[2]
pose_dim = joint_count * joint_dim

"""
Load Model
"""

motion_model.config["seq_length"] = sequence_length
motion_model.config["data_dim"] = pose_dim
motion_model.config["embed_dim"] = rnn_layer_dim
motion_model.config["layer_count"] = rnn_layer_count
motion_model.config["device"] = device
motion_model.config["weights_path"] = rnn_weights_file

model = motion_model.createModel(motion_model.config) 


"""
Setup Motion Synthesis
"""

sequence_overlap = sequence_length // 4 * 3

synthesis_config  = motion_synthesis.config
synthesis_config["skeleton"] = all_mocap_data_dancer1[0]["skeleton"]
synthesis_config["model"] = model
synthesis_config["seq_window_length"] = sequence_length
synthesis_config["seq_window_overlap"] = sequence_overlap
synthesis_config["orig_sequences"] = all_pose_sequences_dancer1
synthesis_config["orig_seq_index"] = 0
synthesis_config["device"] = device

synthesis = motion_synthesis.MotionSynthesis(synthesis_config)


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

motion_control.config["motion_seq"] = all_pose_sequences_dancer1[0]
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
