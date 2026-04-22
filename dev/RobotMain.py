import RobotWorkspace
import LidarSensor
import matplotlib.pyplot as plt
import time
import csv
import numpy as np

lidar = LidarSensor.LidarSensor()
lidar_ser = lidar.init_lidar()
#workspace = RobotWorkspace.RobotWorkspace()
time.sleep(2)  # Give some time for the serial connection to stabilize
lidar.start(lidar_ser)


# plt.ion() # Interactive mode ON
# fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_rmax(5000)
ax.set_title("RPLidar A1M8 Live Map")
ax.set_theta_zero_location('N') # Front of Lidar is North
ax.set_theta_direction(-1)      # Clockwise



while True:
    lidar.measure(lidar_ser)
    if len(lidar.angles) > 150:
                ax.clear()    
                flipped_angles = [2 * np.pi - a for a in lidar.angles]
                ax.scatter(flipped_angles, lidar.distances, s=10, c='red')
                # ax.scatter(lidar.angles, lidar.distances, s=10, c='red')
                            
                # plt.pause(0.01) # Refresh the plot
                
                
                lidar.distances = []
                lidar.angles = []  
    

