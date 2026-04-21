def find_block(angles_rad, distances_mm):
    angles_deg = np.degrees(angles_rad)
    valid = distances_mm > 0
    a = angles_deg[valid]
    d = distances_mm[valid]

    angle_bin = (a // 1).astype(int)
    profile = pd.DataFrame({'angle': angle_bin, 'dist': d}).groupby('angle')['dist'].median().reset_index()

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

    blkpos = pd.DataFrame(detections)
    result = blkpos[['angle', 'dist']].mean()
    return np.array([result['angle'], result['dist']])