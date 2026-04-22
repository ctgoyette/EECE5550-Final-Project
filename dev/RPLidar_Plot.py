import serial
import numpy as np
import matplotlib.pyplot as plt
import time

PORT = '/dev/ttyUSB0' # Set port connection from device manager

def start_lidar_plot():
    # Setup Serial Connection
    ser = serial.Serial(PORT, 115200, timeout=1, dsrdtr=False, rtscts=False)
    ser.dtr = False
    ser.rts = False
    
    
    plt.ion() # Interactive mode ON
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_rmax(5000)
    ax.set_title("RPLidar A1M8 Live Map")
    
    
    # Send Start Command
    ser.reset_input_buffer()
    ser.write(bytearray([0xA5, 0x20]))
    time.sleep(0.5)
    ser.read(7)

    angles = []
    distances = []
    
    print("Plotting... Close the window to stop.")
    
    try:
        while True:
            # Read a batch of points
            if ser.in_waiting >= 5:
                raw = ser.read(5)
                
                # Protocol Math
                dist = (raw[4] << 8 | raw[3]) / 4.0
                angle = ((raw[2] << 7 | (raw[1] >> 1))) / 64.0
                
                # Convert Angle to Radians for Matplotlib
                angle_rad = np.deg2rad(angle)
                
                if dist > 0:
                    angles.append(angle_rad)
                    distances.append(dist)
                
                
                if len(angles) > 300:
                    ax.clear()
                    ax.set_theta_zero_location('N') # Front of Lidar is North
                    ax.set_theta_direction(-1)      # Clockwise
                    
                    ax.scatter(angles, distances, s=5, c='red')
                    
                    plt.pause(0.001) # Refresh the plot
                    angles = []
                    distances = []
                    
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        ser.write(bytearray([0xA5, 0x25])) # Stop command
        ser.close()
        plt.close()

if __name__ == '__main__':
    start_lidar_plot()
