from collections import deque

import serial
import numpy as np
from collections import deque
import time

PORT = 'COM7'

class LidarSensor:
    def __init__(self):
        self.angles = []        
        self.distances = []
        self.angles = deque(maxlen=360) 
        self.distances = deque(maxlen=360)
        pass

    def init_lidar(self):
        ser = serial.Serial(PORT, 115200, timeout=1, dsrdtr=False, rtscts=False)
        ser.dtr = False
        ser.rts = False
        return ser

    def stop(self, ser):
        ser.write(bytearray([0xA5, 0x25])) # Stop command
        ser.close()

    def start(self,ser):
        # Send Start Command
        ser.reset_input_buffer()
        ser.write(bytearray([0xA5, 0x20]))
        time.sleep(0.5)
        ser.read(7) # Skip descriptor

    def measure(self, ser):
        # We need at least 5 bytes to start parsing
        if ser.in_waiting >= 5:
            # Read one byte at a time to find the start of a packet
            sync_byte = ser.read(1)
            
            # Check start bits: S (bit 0) and !S (bit 1)
            # The protocol requires bit 0 and bit 1 of the first byte to be inverse
            start_node = sync_byte[0] & 0x03
            if start_node == 0x01 or start_node == 0x02: 
                # This looks like a valid start! Read the remaining 4 bytes
                raw = ser.read(4)
                
                # Reconstruct the logic with the first byte (sync_byte) included
                # raw[0] is now the 2nd byte of the packet (raw[1] in your old code)
                angle = ((raw[1] << 7) | (raw[0] >> 1)) / 64.0
                dist = ((raw[3] << 8) | raw[2]) / 4.0
                
                if dist > 0 and dist < 1000: # Filter out invalid readings
                    if angle > 180 and angle < 360:
                        angle_rad = np.deg2rad(angle)
                        self.angles.append(angle_rad)
                        self.distances.append(dist)
            else:
                # Not a start byte, discard and let the next loop iteration find the sync
                pass
                    
            
            
            
                        
        
