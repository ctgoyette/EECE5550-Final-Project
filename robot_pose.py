import numpy as np
from ICP import ICP_2D  # your file
from pure_pursuit_package import Pose

class RobotPoseTracker:
    def __init__(self):
        self.icp = ICP_2D(
            dmax=0.05,          # 0.5m threshold — tune for your environment
            num_ICP_iters=30,
            tolerance=0.0001,
            min_points=5
        )

        # Accumulated global pose
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        # Global rotation/translation (composed across all frames)
        self.R_global = np.eye(2)
        self.t_global = np.zeros(2)

        self.reference_scan = None   # first scan = "map"
        self.prev_scan      = None   # for frame-to-frame mode

    # ------------------------------------------------------------------
    # RPLidar raw → [[angle_rad, distance_m], ...]
    # ------------------------------------------------------------------
    @staticmethod
    def parse_rplidar(measurements):
        if not measurements:
            return np.empty((0, 2), dtype=float)
        
        rows = []
        for measurement in measurements:
            if measurement is None:          # skip None entries within the scan
                continue
            quality, angle_deg, distance_mm = measurement
            if quality == 0 or distance_mm <= 0:
                continue
            rows.append([np.deg2rad(angle_deg), distance_mm / 1000])
        
        return np.array(rows, dtype=float) if rows else np.empty((0, 2), dtype=float)

    # ------------------------------------------------------------------
    # Option A: map-to-frame  (stable but drifts if env changes)
    # ------------------------------------------------------------------
    def update_pose_vs_map(self, raw_measurements):
        """
        Always aligns the new scan to the very first (reference) scan.
        Returns (x, y, theta_rad) relative to starting pose.
        """
        scan = self.parse_rplidar(raw_measurements)

        if self.reference_scan is None:
            self.reference_scan = scan
            return 0.0, 0.0, 0.0   # we're at the origin by definition

        dx, dy, dtheta = self.icp.run_icp(
            actual=scan,                  # source  X
            virtual=self.reference_scan   # target  Y
        )
        return dx, dy, dtheta

    # ------------------------------------------------------------------
    # Option B: frame-to-frame  (works over large motions, accumulates drift)
    # ------------------------------------------------------------------
    def update_pose_incremental(self, raw_measurements):
        """
        Aligns each new scan to the previous one and accumulates the pose.
        Returns (x, y, theta_rad) relative to starting pose.
        """
        scan = self.parse_rplidar(raw_measurements)

        if self.prev_scan is None:
            self.prev_scan = scan
            return 0.0, 0.0, 0.0

        dx, dy, dtheta = self.icp.run_icp(
            actual=scan,
            virtual=self.prev_scan
        )

        dx_mm = dx * 1000.0
        dy_mm = dy * 1000.0

        # Compose: rotate increment into global frame then add
        c, s = np.cos(self.theta), np.sin(self.theta)
        R_cur = np.array([[c, -s], [s, c]])

        delta_pos  = R_cur @ np.array([dx_mm, dy_mm])
        self.x      += delta_pos[0]
        self.y      += delta_pos[1]
        self.theta  += dtheta
        self.theta   = (self.theta + np.pi) % (2 * np.pi) - np.pi  # wrap

        self.prev_scan = scan
        return self.x, self.y, self.theta
    
    def to_pose(self):
        return Pose(self.x, self.y, self.theta)