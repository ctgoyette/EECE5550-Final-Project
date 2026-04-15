import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

LIDAR_DATA = "../data/Lidar Data (With Block Against Wall) - lidar_data.csv.csv"
df = pd.read_csv(LIDAR_DATA, header=None, names=["angles(rad)", "distance(mm)"])
df["angles(deg)"] = np.degrees(df["angles(rad)"])
pd.set_option('display.max_rows', None)

resets = df.index[df["angles(rad)"].diff() < -5].tolist()

for i in range(len(resets)-1):
    df_part = df.iloc[resets[i]:resets[i+1]]
    valid = df_part[df_part['distance(mm)'] > 0].copy()
    valid['angle_bin'] = (valid['angles(deg)'] // 1).astype(int)
    profile = valid.groupby('angle_bin')['distance(mm)'].median().reset_index()
    profile.columns = ['angle', 'dist']
    margin = 10
    detections = []

    for i in range(margin, len(profile) - margin):
        left = profile['dist'].iloc[i - margin:i].median()
        right = profile['dist'].iloc[i + 1:i + 1 + margin].median()
        center = profile['dist'].iloc[i]
        
        if np.isnan(left) or np.isnan(right):
            continue
        
        if min(left, right) / max(left, right) < 0.85:
            continue
        
        surround = (left + right) / 2
        if center / surround < 0.8:
            detections.append(profile.iloc[i])

    result = pd.DataFrame(detections)
    print(result)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(df_part["angles(rad)"], df_part["distance(mm)"], s=1)
plt.show()

