#!/usr/bin/env python3
from hw_drivers.dv_rplidar import RPLidarSensor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


try:
    print("Getting a single scan...")
    
    lidar = RPLidarSensor(18, '/dev/ttyAMA0')
    scan = lidar.read_scan_frame()

    angles = []
    distances = []

    for (_, angle, distance) in scan:
        angles.append(angle)
        distances.append(distance)

    # Convert to radians for polar plot
    angles = np.radians(angles)

    # Plot
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.scatter(angles, distances, s=5)

    ax.set_title("LiDAR Scan")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    plt.savefig('test.png')

    with open("scan.csv", "w") as f:
        f.write("angle_deg,distance_mm\n")  # header
        
        for (_, angle, distance) in scan:
            if distance > 0:
                f.write(f"{angle},{distance}\n")

finally:
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()

