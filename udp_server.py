import socket
import time
import numpy as np 


UDP_IP = "127.0.0.1"
UDP_PORT = 9003
x = np.arange(200,dtype=np.uint16).reshape((2,100))
MESSAGE = x.tobytes()

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP

while True:
    time.sleep(0.1)
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
    # print('send ', MESSAGE)
    print('.')