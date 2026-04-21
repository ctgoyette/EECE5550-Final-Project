import numpy as np
class ICP_2D:
    def __init__(self, dmax = 15, num_ICP_iters = 30, tolerance=0.0001, min_points=3):
        self.dmax = float(dmax)
        self.num_iters = int(num_ICP_iters)
        self.tolerance = float(tolerance)
        self.min_points = int(min_points)

    def rmse(self, X, Y, C, R, t):
        if len(C) == 0: return np.inf
        X_tf = self.transform_points(X, R, t) # X_tf=(X @ R.T + t)
        err = X_tf[C[:, 0]] - Y[C[:, 1]]
        return np.sqrt(np.mean(np.sum(err * err, axis=1)))

    def transform_points(self, points, R, t):
        points = np.asarray(points, dtype=float)
        return points @ R.T + t

    def polar_to_cartesian(self, scan):
        scan = np.asarray(scan, dtype=float)
        if scan.size == 0: return np.empty((0, 2), dtype=float)
        if scan.ndim != 2: raise ValueError(f'Dim Error. Got dim={scan.ndim}')
        if scan.shape[1] != 2: raise ValueError(f"Shape Error. Got [angle(rad), distance] = {scan.shape[0]}, {scan.shape[1]}")

        angles = scan[:, 0]
        distances = scan[:, 1]

        finite_mask = np.isfinite(angles) & np.isfinite(distances) & (distances > 0)
        # Create a boolean mask to filter out invalid LiDAR measurements
        # 'isfinite(angles)' -> angles must be finite (not NaN or inf)
        # 'isfinite(dists)' -> distances must be finite (not NaN or inf)
        # '(distances > 0)' -> distances must be strictly positive (physically valid range)
        angles = angles[finite_mask]
        distances = distances[finite_mask]

        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        return np.stack((x, y), axis=1)

    def estimate_correspondences(self, X, Y, R, t):
        # for each transformed point in X
        # if the distance is less than dmax, find the closest point in Y and keep the pair 
        
        dmax = self.dmax
        X_tf = self.transform_points(X, R, t) # X_tf=(X @ R.T + t)
        correspondences = []

        for i, x_i in enumerate(X_tf):
            diff = Y - x_i
            dist2 = np.sum(diff * diff, axis=1)
            j = int(np.argmin(dist2))
            if np.sqrt(dist2[j]) < dmax:
                correspondences.append((i, j))

        if not correspondences:
            print('Fail in estimate_correspondences()')
            return np.empty((0, 2), dtype=int)
        
        return np.asarray(correspondences, dtype=int)

    def compute_optimal_rigid_registration(self, X, Y, C):

        if len(C) < self.min_points:
            print("[WARN] Registration skipped: not enough valid correspondences.")
            print("Returning identity transform instead of computed result.")
            return np.eye(2), np.zeros(2)
        
        K = len(C)

        Xc = X[C[:, 0]];         Yc = Y[C[:, 1]]
        x_bar = Xc.mean(axis=0); y_bar = Yc.mean(axis=0)
        Xp = Xc - x_bar;         Yp = Yc - y_bar

        W = Yp.T @ Xp / K
        U, _, VT = np.linalg.svd(W)

        R = U @ np.diag([1, np.linalg.det(U @ VT)]) @ VT # rotation hat_R
        t = y_bar - R @ x_bar
        return R, t
    
    def icp(self, X, Y, R0=None, t0=None):
        # Align source X to target Y
        # Returns R, t, C, rmse

        num_iters = self.num_iters

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if R0 is None: R = np.eye(2)
        else: R = np.asarray(R0, dtype=float).copy()

        if t0 is None: t = np.zeros(2, dtype=float)
        else: t = np.asarray(t0, dtype=float).reshape(2).copy()

        prev_rmse = np.inf
        C = np.empty((0, 2), dtype=int)

        for _ in range(num_iters):
            C = self.estimate_correspondences(X, Y, R, t)
            X_tf = self.transform_points(X, R, t)
            dR, dt = self.compute_optimal_rigid_registration(X_tf, Y, C)

            # Compose transforms: new = dT ∘ current
            R = dR @ R
            t = dR @ t + dt

            curr_rmse = self.rmse(X, Y, C, R, t)
            if abs(prev_rmse - curr_rmse) < self.tolerance: break
            prev_rmse = curr_rmse

        return R, t, C

    def wrap_to_pi(self, angle): return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def run_icp(self, actual, virtual):
        # actual_scan: ndarray (N, 2). Real lidar scan in [angle, distance]
        # virtual_scan:: ndarray (M, 2). Expected scan from map in [angle, distance]
        # Returns: delta_x, delta_y, delta_theta type float

        X = self.polar_to_cartesian(actual)
        Y = self.polar_to_cartesian(virtual)

        if len(X) < self.min_points or len(Y) < self.min_points: return 0, 0, 0

        R, t, _ = self.icp(X, Y)
        delta_theta = np.arctan2(R[1, 0], R[0, 0])
        #return float(t[0]), float(t[1]), float(self.wrap_to_pi(delta_theta))
        return float(t[0]), float(t[1]), float(delta_theta)

