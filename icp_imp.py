from hw_drivers.dv_rplidar import RPLidarSensor
import numpy as np
from robot_pose import RobotPoseTracker
import time

lidar   = RPLidarSensor(18, '/dev/ttyAMA0')
tracker = RobotPoseTracker()

try:
    while True:
        scan = lidar.read_scan_frame()
        if not scan:
            continue
        x, y, theta = tracker.update_pose_incremental(scan)
        print(f"Pose → x={x:.3f} mm  y={y:.3f} mm  θ={np.degrees(theta):.1f}°")
        from time import sleep
        sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    lidar.stop()
    lidar.disconnect()