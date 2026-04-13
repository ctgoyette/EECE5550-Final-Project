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


plt.ion() # Interactive mode ON
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_rmax(5000)
ax.set_title("RPLidar A1M8 Live Map")
ax.set_theta_zero_location('N') # Front of Lidar is North
ax.set_theta_direction(-1)      # Clockwise



while True:
    choice = input("Press Enter to take a measurement (or Ctrl+C to exit)...")
    if choice.strip() == "":
        for i in range(10000): # Take 360 measurements for a full scan
            lidar.measure(lidar_ser)
            if len(lidar.angles) > 300:
                ax.clear()    
                ax.scatter(lidar.angles, lidar.distances, s=10, c='red')
                            
                plt.pause(0.01) # Refresh the plot
                
                with open('output.csv', 'w', newline='') as file:
                    fieldnames = ['Angle', 'Distance']
                    writer = csv.DictWriter(file, fieldnames=fieldnames)

                    writer.writeheader()  # Writes the first row as column names
                    writer.writerows({'Angle': np.rad2deg(angle), 'Distance': dist} for angle, dist in zip(lidar.angles, lidar.distances))

                lidar.distances = []
                lidar.angles = []

    elif choice.strip().lower() == "exit":
        lidar.stop(lidar_ser)
        print("Exiting...")
        break
    

