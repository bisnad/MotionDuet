import torch
import numpy as np

from common.rotation_utils_numpy import RotationUtilsNumpy as rot_np
from common.rotation_utils_torch import RotationUtilsTorch as rot_to

config = {
    "skeleton": None,
    "model_transformer": None,
    "device": "cuda",
    "seq_length": 64,
    "joint_count": 28,
}

class MotionSynthesis():
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model_transformer"]
        self.device = config["device"]
        self.seq_length = config["seq_length"]
        self.joint_count = config["joint_count"]
        
        self.pose_dim = self.joint_count * 6 # 6D rotations
        
        self.joint_offsets = self.skeleton["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton["parents"]
        self.joint_children = self.skeleton["children"]
        
        # Buffers for real-time inference (stored as 6D rotations)
        self.dancer1_seq = torch.zeros((self.seq_length, self.pose_dim), dtype=torch.float32).to(self.device)
        self.dancer2_seq = torch.zeros((self.seq_length, self.pose_dim), dtype=torch.float32).to(self.device)
        
        # Initialize sequence with identity 6D rotations (1, 0, 0, 1, 0, 0)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).repeat(self.joint_count).to(self.device)
        for i in range(self.seq_length):
            self.dancer1_seq[i] = identity_6d
            self.dancer2_seq[i] = identity_6d
        
        self.live_pose_d1 = None
        self.live_pose_changed = False
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
    def setLiveSeq(self, rotLocal):
        # rotLocal arrives as flat Quaternions via OSC
        rotLocal = np.asarray(rotLocal, dtype=np.float32).reshape(self.joint_count, 4)
        
        # Convert incoming Quaternions to 6D for the model
        rotLocal_6d = rot_np.quat_to_r6d(rotLocal)
        self.live_pose_d1 = torch.tensor(rotLocal_6d, dtype=torch.float32).flatten().to(self.device)
        self.live_pose_changed = True

    def update(self):
        # Only predict if we have received a new OSC frame to prevent desync
        if not self.live_pose_changed:
            return 
            
        # Shift Dancer 1 buffer and append the newly received live frame
        self.dancer1_seq = torch.cat([self.dancer1_seq[1:], self.live_pose_d1.unsqueeze(0)], dim=0)
        self.live_pose_changed = False
        
        # Autoregressive generation
        _in_d1 = self.dancer1_seq.unsqueeze(0) # (1, seq_length, pose_dim)
        _in_d2 = self.dancer2_seq.unsqueeze(0) # (1, seq_length, pose_dim)
        
        with torch.no_grad():
            _pred_dancer2 = self.model(_in_d1, _in_d2)
        
        # Extract the next predicted pose (last frame in the sequence)
        pred_pose_6d = _pred_dancer2[0, -1, :]
        
        # Shift Dancer 2 buffer and append the prediction
        self.dancer2_seq = torch.cat([self.dancer2_seq[1:], pred_pose_6d.unsqueeze(0)], dim=0)
        
        # Prepare Output: Convert 6D -> Matrix -> Forward Kinematics
        pred_rot_6d = pred_pose_6d.reshape(self.joint_count, 6)
        pred_rot_mat = rot_to.r6d_to_mat(pred_rot_6d.unsqueeze(0)) # Shape: (1, J, 3, 3)
        zero_trajectory = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
        
        pos_world, rot_world = self._forward_kinematics(pred_rot_mat, zero_trajectory)
        
        self.synth_pose_wpos = pos_world.squeeze(0).detach().cpu().numpy() # Shape: (J, 3)
        
        # Convert 6D back to Quaternions for the GUI / OSC Sender
        pred_rot_6d_np = pred_rot_6d.detach().cpu().numpy()
        pred_rot_quat_np = rot_np.r6d_to_quat(pred_rot_6d_np)
        self.synth_pose_wrot = pred_rot_quat_np # Shape: (J, 4)
        
    def _forward_kinematics(self, rotation_matrices, root_positions):
        t_offsets = torch.tensor(self.joint_offsets).to(self.device)
        expanded_offsets = t_offsets.expand(rotation_matrices.shape[0], self.joint_offsets.shape[0], self.joint_offsets.shape[1]).unsqueeze(-1)
        
        positions_world = []
        rotations_world = []
        
        for jI in range(self.joint_offsets.shape[0]):
            if self.joint_parents[jI] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotation_matrices[:, jI])
            else:
                parent_rot = rotations_world[self.joint_parents[jI]]
                local_offset = expanded_offsets[:, jI]
                rotated_offset = torch.matmul(parent_rot, local_offset).squeeze(-1)
                positions_world.append(rotated_offset + positions_world[self.joint_parents[jI]])

                if len(self.joint_children[jI]) > 0:
                    new_world_rot = torch.matmul(parent_rot, rotation_matrices[:, jI])
                    rotations_world.append(new_world_rot)
                else:
                    rotations_world.append(parent_rot)
                    
        return torch.stack(positions_world, dim=1), torch.stack(rotations_world, dim=1)