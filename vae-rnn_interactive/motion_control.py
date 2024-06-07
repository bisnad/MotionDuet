import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server


config = {"synthesis": None,
          "gui": None,
          "input_length": 64,
          "ip": "127.0.0.1",
          "port": 9004}

class MotionControl():
    
    def __init__(self, config):
        
        self.synthesis = config["synthesis"]
        self.gui = config["gui"]
        self.input_length = config["input_length"]
        self.ip = config["ip"]
        self.port = config["port"]
        
         
        self.dispatcher = dispatcher.Dispatcher()
        
        self.dispatcher.map("/mocap/seq1index", self.setSeq1Index)
        self.dispatcher.map("/mocap/seq2index", self.setSeq2Index)
        self.dispatcher.map("/mocap/deepfakemode", self.setDeepFakeMode)
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
        
    def setSeq1Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq1Index(index)

    def setSeq2Index(self, address, *args):
        
        index = args[0]
        self.synthesis.setSeq2Index(index)
        
    def setDeepFakeMode(self, address, *args):
        
        mode = args[0]
        self.synthesis.setDeepFakeMode(mode)

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