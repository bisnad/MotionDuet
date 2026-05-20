import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl

from threading import Thread, Event
import time
from time import sleep
import datetime

from common import fbx_tools as fbx  # Added for FBX exporting

config = {
    "synthesis": None,
    "sender": None,
    "update_interval": 0.02, # 50 Hz matching your dataset
    "view_min": np.array([-100, -100, -100], dtype=np.float32),
    "view_max": np.array([100, 100, 100], dtype=np.float32),
    "view_ele": 90,
    "view_azi": -90,
    "view_dist": 250,
    "view_line_width": 2.0,
    "osc_ip": "127.0.0.1",
    "osc_port": 9005,
    "mocap_fps": 50
}

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()

class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opts['rotationMethod'] = 'quaternion'

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
        diff = lpos - self.mousePos
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.LeftButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self.pan(diff.x(), diff.y(), 0, relative='view')
            else:
                self.orbit(-diff.x(), diff.y())
        elif ev.buttons() == QtCore.Qt.MiddleButton:
            if ev.modifiers() & QtCore.Qt.ControlModifier:
                self.pan(diff.x(), 0, diff.y(), relative='view-upright')
            else:
                self.pan(diff.x(), diff.y(), 0, relative='view-upright')
        else:
            super().mouseMoveEvent(ev)

class MotionGui(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        
        self.synthesis = config["synthesis"]
        self.sender = config["sender"]
        self.mocap_fps = config.get("mocap_fps", 50)
        
        # Build edge list manually from the skeleton in synthesis
        self.edges = []
        if hasattr(self.synthesis, 'joint_children'):
            for pI in range(len(self.synthesis.joint_children)):
                for cI in self.synthesis.joint_children[pI]:
                    self.edges.append([pI, cI])
        
        self.pose_thread_interval = config["update_interval"]
        self.view_dist = config["view_dist"]
        self.view_azi = config["view_azi"]
        self.view_ele = config["view_ele"]
        self.view_line_width = config["view_line_width"]
        
        # Recording state
        self.is_recording = False
        self.record_buffer_pos = []
        self.record_buffer_rot = []
        
        # dynamic canvas
        self.pose_canvas = CustomGLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)
        self.pose_canvas.setCameraParams(distance=self.view_dist, azimuth=self.view_azi, elevation=self.view_ele)

        # Buttons
        self.q_start_buttom = QtWidgets.QPushButton("Start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("Stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  
        
        self.q_record_button = QtWidgets.QPushButton("Record", self)
        self.q_record_button.setCheckable(True)
        self.q_record_button.clicked.connect(self.toggle_recording)
        
        self.q_exit_button = QtWidgets.QPushButton("Exit", self)
        self.q_exit_button.clicked.connect(self.exit_application)
        
        self.q_button_grid = QtWidgets.QHBoxLayout()
        self.q_button_grid.addWidget(self.q_start_buttom)
        self.q_button_grid.addWidget(self.q_stop_buttom)
        self.q_button_grid.addWidget(self.q_record_button)
        self.q_button_grid.addWidget(self.q_exit_button)

        # OSC IP and Port Layout
        self.q_osc_layout = QtWidgets.QFormLayout()
        
        self.q_osc_ip = QtWidgets.QLineEdit(config.get("osc_ip", "127.0.0.1"))
        self.q_osc_ip.textChanged.connect(self.change_osc_ip)
        self.q_osc_layout.addRow("OSC IP:", self.q_osc_ip)
        
        self.q_osc_port = QtWidgets.QSpinBox()
        self.q_osc_port.setRange(1024, 65535)
        self.q_osc_port.setValue(config.get("osc_port", 9005))
        self.q_osc_port.valueChanged.connect(self.change_osc_port)
        self.q_osc_layout.addRow("OSC Port:", self.q_osc_port)

        # Wrap OSC layout in a container so it aligns to the top right without stretching
        osc_container = QtWidgets.QWidget()
        osc_vbox = QtWidgets.QVBoxLayout()
        osc_vbox.addLayout(self.q_osc_layout)
        osc_vbox.addStretch()  
        osc_container.setLayout(osc_vbox)
        osc_container.setMinimumWidth(200)

        # Canvas Grid: [Pose Canvas | OSC Container]
        self.canvas_grid = QtWidgets.QGridLayout()
        self.canvas_grid.addWidget(self.pose_canvas, 0, 0)
        self.canvas_grid.addWidget(osc_container, 0, 1)
        
        self.canvas_grid.setColumnStretch(0, 1) 
        self.canvas_grid.setColumnStretch(1, 0) 

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.canvas_grid, 0, 0)
        self.q_grid.addLayout(self.q_button_grid, 1, 0)
        
        self.q_grid.setRowStretch(0, 1)
        self.q_grid.setRowStretch(1, 0)
        
        self.setLayout(self.q_grid)
        self.setGeometry(50, 50, 512 + 200, 612)
        self.setWindowTitle("Motion Duet")

        self.poseCanvasUpdater = PoseCanvasUpdater()
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)
        
    # --- Feature Addition Methods ---
    def change_osc_ip(self, text):
        self.sender.config["ip"] = text
        if hasattr(self.sender, 'client'):
            try:
                from pythonosc import udp_client
                self.sender.client = udp_client.SimpleUDPClient(text, self.q_osc_port.value())
            except ImportError:
                pass

    def change_osc_port(self, val):
        self.sender.config["port"] = val
        if hasattr(self.sender, 'client'):
            try:
                from pythonosc import udp_client
                self.sender.client = udp_client.SimpleUDPClient(self.q_osc_ip.text(), val)
            except ImportError:
                pass

    def toggle_recording(self):
        self.is_recording = self.q_record_button.isChecked()
        if self.is_recording:
            self.q_record_button.setText("Stop Recording")
            self.record_buffer_pos = []
            self.record_buffer_rot = []
            print("Recording started...")
        else:
            self.q_record_button.setText("Record")
            self.save_recording()

    def save_recording(self):
        if len(self.record_buffer_rot) == 0:
            print("No frames were recorded.")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_duet_motion_{timestamp}.fbx"
        
        rot_local_quat = np.array(self.record_buffer_rot)
        rot_local_euler = np.zeros((*rot_local_quat.shape[:-1], 3))
        
        try:
            from common.mocap_tools import Mocap_Tools
            mocap_t = Mocap_Tools()
            if hasattr(mocap_t, 'quat_to_euler'):
                rot_local_euler = mocap_t.quat_to_euler(rot_local_quat, [0,1,2])
            else:
                from scipy.spatial.transform import Rotation
                flat_quats = rot_local_quat.reshape(-1, 4)
                flat_eulers = Rotation.from_quat(flat_quats).as_euler('xyz', degrees=True)
                rot_local_euler = flat_eulers.reshape((*rot_local_quat.shape[:-1], 3))
        except Exception as e:
            print(f"Warning: Could not properly convert quaternions to euler angles. ({e})")
            
        from common.fbx_tools import FBX_Mocap_Data, FBX_Tools
        
        fbx_data = FBX_Mocap_Data()
        skel = getattr(self.synthesis, 'skeleton', {})
        
        fbx_data.skeleton_joints = skel.get("joints", [])
        fbx_data.skeleton_children = skel.get("children", [])
        fbx_data.skeleton_parents = skel.get("parents", [])
        fbx_data.skeleton_joint_offsets = skel.get("offsets", [])
        fbx_data.skeleton_root_node = None 
        fbx_data.skeleton_nodes = []
        
        fbx_data.motion_rot_sequence = [0, 1, 2] 
        fbx_data.motion_frame_rate = float(self.mocap_fps)
        fbx_data.motion_frame_count = len(self.record_buffer_pos)
        fbx_data.motion_pos_local = np.array(self.record_buffer_pos)
        fbx_data.motion_rot_local_euler = rot_local_euler
        fbx_data.system_unit = "cm" 

        exporter = fbx.FBX_Tools()
        try:
            exporter.write([fbx_data], filename)
            print(f"Successfully saved recording to: {filename}")
        except Exception as e:
            print(f"Error saving FBX file: {e}")

    def exit_application(self):
        if hasattr(self, 'pose_thread_event') and not self.pose_thread_event.is_set():
            self.stop()
        self.close()

    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        self.pose_thread.start()
        
    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()
                
    def update(self):
        while self.pose_thread_event.is_set() == False:
            start_time = time.time()            

            self.update_pred_seq()
            self.poseCanvasUpdater.request_canvas_update.emit() 
            self.update_osc()
            
            end_time = time.time()   
            next_update_interval = max(self.pose_thread_interval - (end_time - start_time), 0.0)
            sleep(next_update_interval)
            
    def update_pred_seq(self):
        # Trigger the transformer only if new OSC data arrived
        self.synthesis.update()       
        self.synth_pose_wpos = getattr(self.synthesis, 'synth_pose_wpos', None)
        self.synth_pose_wrot = getattr(self.synthesis, 'synth_pose_wrot', None)
        # Attempt to retrieve local rotations if they exist for exporting
        self.synth_pose_lrot = getattr(self.synthesis, 'synth_pose_lrot', None)
        
    def update_osc(self):
        if self.synth_pose_wpos is None or self.synth_pose_wrot is None:
            return

        # convert from left handed bvh coordinate system to right handed standard coordinate system
        self.synth_pose_wpos_rh = np.copy(self.synth_pose_wpos)
        self.synth_pose_wpos_rh[:, 0] = self.synth_pose_wpos[:, 0] / 100.0
        self.synth_pose_wpos_rh[:, 1] = -self.synth_pose_wpos[:, 2] / 100.0
        self.synth_pose_wpos_rh[:, 2] = self.synth_pose_wpos[:, 1] / 100.0

        self.synth_pose_wrot_rh = np.copy(self.synth_pose_wrot)
        self.synth_pose_wrot_rh[:, 1] = self.synth_pose_wrot[:, 1]
        self.synth_pose_wrot_rh[:, 2] = -self.synth_pose_wrot[:, 3]
        self.synth_pose_wrot_rh[:, 3] = self.synth_pose_wrot[:, 2]
        
        # Dispatch the predicted pose out over OSC
        self.sender.send("/mocap/0/joint/pos_world", self.synth_pose_wpos_rh)
        self.sender.send("/mocap/0/joint/rot_world", self.synth_pose_wrot_rh)

        # Buffer frames when recording
        if hasattr(self, 'is_recording') and self.is_recording:
            # We reconstruct pos_local. The root gets absolute position, children get zero vectors
            pos_local = np.zeros_like(self.synth_pose_wpos)
            pos_local[0] = self.synth_pose_wpos[0] 
            
            self.record_buffer_pos.append(pos_local)
            
            # Use local rotation if exposed by synthesis module, else fallback to world rot safely
            if self.synth_pose_lrot is not None:
                self.record_buffer_rot.append(np.copy(self.synth_pose_lrot))
            else:
                self.record_buffer_rot.append(np.copy(self.synth_pose_wrot))

    def update_pose_plot(self):
        if self.synth_pose_wpos is None:
            return

        pose = self.synth_pose_wpos
        points_data = pose
        lines_data = pose[np.array(self.edges).flatten()]
        
        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        self.pose_canvas_points.setData(pos=pose, color=(1.0, 1.0, 1.0, 0.5))