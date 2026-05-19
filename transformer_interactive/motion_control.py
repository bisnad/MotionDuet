import threading
from pythonosc import dispatcher
from pythonosc import osc_server

config = {
    "synthesis": None,
    "ip": "127.0.0.1",
    "port": 9004
}

class MotionControl():
    def __init__(self, config):
        self.synthesis = config["synthesis"]
        self.ip = config["ip"]
        self.port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/joint/rot_local", self.setLiveSeq)
        self.dispatcher.map("/mocap/*/joint/rot_local", self.setLiveSeq)
        
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
                
    def start_server(self):
        self.server.serve_forever()

    def start(self):
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()
        
    def setLiveSeq(self, address, *args):
        self.synthesis.setLiveSeq(args)