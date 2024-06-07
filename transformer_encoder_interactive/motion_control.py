import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"motion_seq": None,
          "synthesis": None,
          "gui": None,
          "input_length": 64,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.motion_seq = config["motion_seq"]
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.input_length = config["input_length"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/seqindex", self.setSeqIndex)
        self.dispatcher.map("/mocap/seqframeindex", self.setSeqFrameIndex)
        self.dispatcher.map("/mocap/seqframerange", self.setSeqFrameRange)
        self.dispatcher.map("/mocap/seqframeincr", self.setSeqFrameIncrement)
    
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()
        
    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        
        self.th2 = threading.Thread(target=self.stop_server)
        self.th2.start()
        
    def setSeqIndex(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeqIndex(index)
        
    def setSeqFrameIndex(self, address, *args):
        
        index = args[0]
        
        self.synthesis.setSeqFrameIndex(index)
            
    def setSeqFrameRange(self, address, *args):
        
        startFrame = args[0]
        endFrame = args[1]
        
        self.synthesis.setSeqFrameRange(startFrame, endFrame)
    
    def setSeqFrameIncrement(self, address, *args):
    
        incr = args[0]
        
        self.synthesis.setSeqFrameIncrement(incr)