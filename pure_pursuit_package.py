import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from RobotWorkspace import RobotWorkspace


def wrap2pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose:
    x: float
    y: float
    theta: float  # rad


class PoseSource:
    def get_pose(self) -> Pose:
        raise NotImplementedError


class SimulatedPoseSource(PoseSource):
    def __init__(self, initial_pose: Pose):
        self.pose = initial_pose

    def get_pose(self) -> Pose:
        return self.pose

    def set_pose(self, pose: Pose) -> None:
        self.pose = pose


class LowLevelController:
    def __init__(self, nominal_speed: float):
        self.nominal_speed = nominal_speed

    def command_from_curvature(self, curvature: float) -> Tuple[float, float]:
        v = self.nominal_speed
        omega = v * curvature
        return v, omega


class PurePursuitTracker:
    def __init__(self, path: np.ndarray, lookahead: float):
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("path must have shape (N, 2)")
        if len(path) < 2:
            raise ValueError("path must contain at least 2 points")
        if lookahead <= 0:
            raise ValueError("lookahead must be positive")

        self.path = path.astype(float)
        self.lookahead = float(lookahead)

    def find_closest_point_index(self, pose: Pose) -> int:
        pos = np.array([pose.x, pose.y], dtype=float)
        dists = np.linalg.norm(self.path - pos, axis=1)
        return int(np.argmin(dists))

    def find_goal_point_index(self, pose: Pose, closest_idx: int) -> int:
        pos = np.array([pose.x, pose.y], dtype=float)
        for i in range(closest_idx, len(self.path)):
            if np.linalg.norm(self.path[i] - pos) >= self.lookahead:
                return i
        return len(self.path) - 1

    def transform_goal_to_vehicle_frame(self, pose: Pose, goal_global: np.ndarray) -> Tuple[float, float]:
        dx = float(goal_global[0] - pose.x)
        dy = float(goal_global[1] - pose.y)

        c = math.cos(pose.theta)
        s = math.sin(pose.theta)

        x_r = c * dx + s * dy
        y_r = -s * dx + c * dy
        return x_r, y_r

    def compute_curvature(self, x_r: float, y_r: float) -> float:
        ld2 = x_r * x_r + y_r * y_r
        if ld2 < 1e-12:
            return 0.0
        return 2.0 * y_r / ld2

    def step(self, pose: Pose):
        closest_idx = self.find_closest_point_index(pose)
        goal_idx = self.find_goal_point_index(pose, closest_idx)
        goal_global = self.path[goal_idx]

        x_r, y_r = self.transform_goal_to_vehicle_frame(pose, goal_global)
        curvature = self.compute_curvature(x_r, y_r)

        return {
            "closest_idx": closest_idx,
            "goal_idx": goal_idx,
            "goal_global": goal_global,
            "goal_local": (x_r, y_r),
            "curvature": curvature,
        }


def simulate_unicycle_step(pose: Pose, v: float, omega: float, dt: float) -> Pose:
    new_x = pose.x + v * math.cos(pose.theta) * dt
    new_y = pose.y + v * math.sin(pose.theta) * dt
    new_theta = wrap2pi(pose.theta + omega * dt)
    return Pose(new_x, new_y, new_theta)


def build_workspace_and_paths(block_xy=(20, 80), goal_xy=(90, 90)):
    ws = RobotWorkspace()

    block_x, block_y = block_xy
    goal_x, goal_y = goal_xy

    ws.set_block(block_x, block_y)
    ws.set_goal(goal_x, goal_y)

    path1 = ws.find_path((ws.x0, ws.y0), (block_x, block_y), ws.path_block_line)
    path2 = ws.find_path((block_x, block_y), (goal_x, goal_y), ws.path_goal_line)

    if path1 is None:
        raise ValueError(f"path1 not found: start=({ws.x0}, {ws.y0}), block=({block_x}, {block_y})")
    if path2 is None:
        raise ValueError(f"path2 not found: block=({block_x}, {block_y}), goal=({goal_x}, {goal_y})")

    path1 = np.asarray(path1, dtype=float)
    path2 = np.asarray(path2, dtype=float)
    path_full = np.vstack([path1, path2[1:]])

    return {
        "ws": ws,
        "block": np.array([block_x, block_y], dtype=float),
        "goal": np.array([goal_x, goal_y], dtype=float),
        "path1": path1,
        "path2": path2,
        "path_full": path_full,
    }


def run_pure_pursuit(
    block_xy=(20, 80),
    goal_xy=(90, 90),
    dt=0.05,
    total_time=30.0,
    nominal_speed=80.0,
    lookahead_phase1=20.0,
    lookahead_phase2=60.0,
    block_tolerance=5.0,
    goal_tolerance=5.0,
    pickup_pause_steps=5,
    initial_pose=None,
    enable_workspace_update=False,
    verbose=True,
):
    env = build_workspace_and_paths(block_xy=block_xy, goal_xy=goal_xy)

    ws = env["ws"]
    block = env["block"]
    goal = env["goal"]
    path1 = env["path1"]
    path2 = env["path2"]
    path_full = env["path_full"]

    if initial_pose is None:
        initial_pose = Pose(x=ws.x0, y=ws.y0, theta=0.0)

    pose_source = SimulatedPoseSource(initial_pose)
    low_level = LowLevelController(nominal_speed=nominal_speed)

    phase = 1
    picked_up = False
    paused_steps = 0

    tracker = PurePursuitTracker(path=path1, lookahead=lookahead_phase1)

    traj = []
    step_logs = []

    for step_idx in range(int(total_time / dt)):
        pose = pose_source.get_pose()

        if phase == 1:
            active_path = path1
            phase_name = "PHASE 1: TO BLOCK"
            final_target = block
        else:
            active_path = path2
            phase_name = "PHASE 2: TO GOAL"
            final_target = goal

        if paused_steps > 0:
            paused_steps -= 1
            v = 0.0
            omega = 0.0
            curvature = 0.0
            info = {
                "closest_idx": None,
                "goal_idx": None,
                "goal_global": final_target,
                "goal_local": (0.0, 0.0),
                "curvature": 0.0,
            }
            new_pose = pose
            event = "pickup_pause"

            if verbose:
                print(f"STEP={step_idx}, {phase_name}, STOPPED for pickup... remaining={paused_steps}")
        else:
            tracker.path = active_path

            info = tracker.step(pose)
            goal_pt = info["goal_global"]
            x_r, y_r = info["goal_local"]

            heading_to_goal = math.atan2(goal_pt[1] - pose.y, goal_pt[0] - pose.x)
            heading_error = wrap2pi(heading_to_goal - pose.theta)

            if abs(heading_error) > math.radians(25):
                v = 0.0
                omega = 2.0 * heading_error
                curvature = 0.0
                event = "turn_in_place"
            else:
                curvature = info["curvature"]
                v, omega = low_level.command_from_curvature(curvature)

                if abs(curvature) > 0.03:
                    v *= 0.7
                    omega = v * curvature

                event = "track_path"

            new_pose = simulate_unicycle_step(pose, v, omega, dt)
            pose_source.set_pose(new_pose)

            if verbose:
                print(
                    f"STEP={step_idx}, {phase_name}, "
                    f"POSE=({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.theta):.2f} deg), "
                    f"CLOSEST_IDX={info['closest_idx']}, GOAL_IDX={info['goal_idx']}, "
                    f"GOAL_LOCAL=({x_r:.2f}, {y_r:.2f}), "
                    f"KAPPA={curvature:.5f}, V={v:.2f}, OMEGA={omega:.5f}"
                )

        traj.append([new_pose.x, new_pose.y, new_pose.theta])

        if enable_workspace_update:
            try:
                ws.update_robot(new_pose.x, new_pose.y)
            except Exception:
                pass

        dist_to_target = np.linalg.norm(np.array([new_pose.x, new_pose.y]) - final_target)
        switch_event = None

        if phase == 1 and dist_to_target < block_tolerance and not picked_up:
            picked_up = True
            paused_steps = pickup_pause_steps
            phase = 2
            tracker = PurePursuitTracker(path=path2, lookahead=lookahead_phase2)
            switch_event = "reached_block"

            if verbose:
                print("Reached block. Stop and pick up block.")

        elif phase == 2 and dist_to_target < goal_tolerance:
            switch_event = "reached_goal"

            if verbose:
                print("Reached goal region.")
            step_logs.append({
                "step_idx": step_idx,
                "phase": phase_name,
                "x": new_pose.x,
                "y": new_pose.y,
                "theta_deg": math.degrees(new_pose.theta),
                "v": v,
                "omega": omega,
                "curvature": curvature,
                "event": event,
                "switch_event": switch_event,
            })
            break

        step_logs.append({
            "step_idx": step_idx,
            "phase": phase_name,
            "x": new_pose.x,
            "y": new_pose.y,
            "theta_deg": math.degrees(new_pose.theta),
            "v": v,
            "omega": omega,
            "curvature": curvature,
            "event": event,
            "switch_event": switch_event,
        })

    traj = np.asarray(traj, dtype=float)

    return {
        "ws": ws,
        "block": block,
        "goal": goal,
        "path1": path1,
        "path2": path2,
        "path_full": path_full,
        "trajectory": traj,         # shape (N, 3): x, y, theta
        "step_logs": step_logs,     # list of dict
        "final_pose": Pose(*traj[-1]) if len(traj) > 0 else initial_pose,
        "success": len(step_logs) > 0 and step_logs[-1]["switch_event"] == "reached_goal",
    }


def plot_pure_pursuit_result(result, save_path=None, show=True):
    path1 = result["path1"]
    path2 = result["path2"]
    path_full = result["path_full"]
    traj = result["trajectory"]
    block = result["block"]
    goal = result["goal"]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(path_full[:, 0], path_full[:, 1], "--", linewidth=2, label="Full Path")
    ax.plot(path1[:, 0], path1[:, 1], "-", linewidth=2, label="Path to Block")
    ax.plot(path2[:, 0], path2[:, 1], "-", linewidth=2, label="Path to Goal")

    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label="Trajectory")
        ax.scatter(traj[0, 0], traj[0, 1], marker="D", s=80, label="Start Pose")
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=80, label="Final Pose")

    ax.scatter(block[0], block[1], marker="o", s=100, label="Block")
    ax.scatter(goal[0], goal[1], marker="*", s=140, label="Goal")

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_title("Pure Pursuit Result")

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_step_logs_csv(step_logs, csv_path):
    import csv

    if not step_logs:
        return

    fieldnames = list(step_logs[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_logs)


if __name__ == "__main__":
    result = run_pure_pursuit(
        block_xy=(20, 80),
        goal_xy=(90, 90),
        dt=0.05,
        total_time=30.0,
        nominal_speed=80.0,
        lookahead_phase1=20.0,
        lookahead_phase2=60.0,
        block_tolerance=5.0,
        goal_tolerance=5.0,
        pickup_pause_steps=5,
        verbose=True,
    )

    plot_pure_pursuit_result(result, save_path=None, show=True)
    save_step_logs_csv(result["step_logs"], "pure_pursuit_log.csv")

    print("success =", result["success"])
    print("final_pose =", result["final_pose"])
