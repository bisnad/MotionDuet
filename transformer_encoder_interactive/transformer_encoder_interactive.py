"""
same as rnn_ntf_pos_weighted_joints.py
but only for inference and interactive control and sending
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

mocap_path = "D:/Data/mocap/stocos/Duets/Amsterdam_2024/bvh_50hz"
mocap_files = [ "Recording_JS-001_jason.bvh" ]

seq_length = 64
seq_overlap = 48
mocap_fps = 50

"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    bvh_data = bvh_tools.load(mocap_path + "/" + mocap_file)
    mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    all_mocap_data.append(mocap_data)
    
all_pose_sequences = []

for mocap_data in all_mocap_data:
    
    pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences.append(pose_sequence)

joint_count = all_pose_sequences[0].shape[1]
joint_dim = all_pose_sequences[0].shape[2]
pose_dim = joint_count * joint_dim

all_pose_sequences[0].shape


"""
Load Model
"""

motion_model.config = {
    "seq_length": seq_length,
    "data_dim": pose_dim,
    "embed_dim": 512,
    "head_count": 8,
    "layer_count": 6,
    "dropout": 0.1,
    "device": device,
    "weights_path": "../transformer_encoder/results_recordings_1-3/weights/transformer_encoder_weights_epoch_100"
    }

model = motion_model.createModel(motion_model.config) 


"""
Setup Motion Synthesis
"""

synthesis_config  = motion_synthesis.config
synthesis_config["skeleton"] = all_mocap_data[0]["skeleton"]
synthesis_config["model"] = model
synthesis_config["seq_window_length"] = seq_length
synthesis_config["seq_window_overlap"] = seq_overlap
synthesis_config["orig_sequences"] = all_pose_sequences
synthesis_config["orig_seq_index"] = 0
synthesis_config["device"] = device

synthesis = motion_synthesis.MotionSynthesis(synthesis_config)


"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9005

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

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["ip"] = "127.0.0.1"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()


osc_control.stop()
