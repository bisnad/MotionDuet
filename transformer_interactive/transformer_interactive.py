import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control

import torch
import sys
import numpy as np

from common import fbx_tools as fbx
from common import mocap_tools as mocap

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load a skeleton template
mocap_file_path = "data/mocap/Jason_Take3.fbx"
transformer_weights_file = "data/results/weights/transformer_weights_epoch_200"

# Mocap & Transformer Properties
seq_length = 64
seq_non_teacherforcing = 10 # Set this to match your training script
pos_encoding_max_length = seq_length + seq_non_teacherforcing

osc_send_ip = "127.0.0.1"
osc_send_port = 9005

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

# 1. Load Skeleton Data
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()
fbx_data = fbx_tools.load(mocap_file_path)
mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0]

mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
mocap_data["skeleton"]["offsets"][0, 2] = 0.0 

# Extract joint properties dynamically from the loaded FBX
joint_count = mocap_data["skeleton"]["offsets"].shape[0]
pose_dim = joint_count * 6 

# 2. Setup Model
motion_model.config["mocap_dim"] = pose_dim
motion_model.config["embed_dim"] = 256 # 512
motion_model.config["num_heads"] = 4 # 8
motion_model.config["num_encoder_layers"] = 3 # 6
motion_model.config["num_decoder_layers"] = 3 # 6
motion_model.config["dropout_p"] = 0.1
motion_model.config["seq_length"] = seq_length
motion_model.config["pos_encoding_max_length"] = pos_encoding_max_length
motion_model.config["device"] = device
motion_model.config["weights_path"] = transformer_weights_file

transformer = motion_model.createModel(motion_model.config)

# 3. Setup Synthesis
motion_synthesis.config["skeleton"] = mocap_data["skeleton"]
motion_synthesis.config["model_transformer"] = transformer
motion_synthesis.config["device"] = device
motion_synthesis.config["seq_length"] = seq_length
motion_synthesis.config["joint_count"] = joint_count

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

# 4. Setup OSC Sender
motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port
osc_sender = motion_sender.OscSender(motion_sender.config)

# 5. Setup GUI
from PyQt5 import QtWidgets

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent)

# 6. Setup OSC Receiver
motion_control.config["synthesis"] = synthesis
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

# Start Application
osc_control.start()
gui.show()
app.exec_()
osc_control.stop()