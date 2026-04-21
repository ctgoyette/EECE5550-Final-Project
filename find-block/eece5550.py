import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import LidarSensor

lidar = LidarSensor.LidarSensor()
lidar_ser = lidar.init_lidar()
time.sleep(2)
lidar.start(lidar_ser)

mlp = joblib.load("mlp_model.pkl")
sc = joblib.load("scaler.pkl")
block_class_idx = list(mlp.classes_).index(1)

plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
ax.set_rmax(5000)
ax.set_title("RPLidar A1M8 Live Map")
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

def predict_block(rows):
    binned = np.zeros(360)
    counts = np.zeros(360)
    for a, d in rows:
        idx = int(a) % 360
        binned[idx] += d
        counts[idx] += 1
    mask = counts > 0
    binned[mask] /= counts[mask]
    binned[~mask] = np.mean(binned[mask])
    binned = (binned - binned.mean()) / (binned.std() + 1e-8)
    X = np.array([[binned[(i + j) % 360] for j in range(-15, 16)] for i in range(360)])
    probs = mlp.predict_proba(sc.transform(X))[:, block_class_idx]
    return np.argmax(probs)

while True:
    lidar.measure(lidar_ser)
    if len(lidar.angles) <= 150:
        continue
    angles = np.array(lidar.angles)
    distances = np.array(lidar.distances)
    rows = list(zip(np.degrees(angles) % 360, distances))
    block_angle = predict_block(rows)

    ax.clear()
    ax.scatter(2 * np.pi - angles, distances, s=10, c='red')
    ax.plot(2 * np.pi - np.radians(block_angle), ax.get_rmax(), 'go', markersize=15)
    plt.pause(0.01)

    lidar.distances = []
    lidar.angles = []
