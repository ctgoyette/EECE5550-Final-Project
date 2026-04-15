import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from RobotWorkspace import RobotWorkspace
from pure_pursuit_package import Pose, PurePursuitTracker, wrap2pi


# Utilities

def deg2rad(deg: float) -> float: return math.radians(deg)
def rad2deg(rad: float) -> float: return math.degrees(rad)
def distance_xy(pose: Pose, pt: np.ndarray) -> float:
    return float(np.linalg.norm(np.array([pose.x, pose.y], dtype=float) - pt))
def heading_to_point(pose: Pose, pt: np.ndarray) -> float:
    return math.atan2(float(pt[1] - pose.y), float(pt[0] - pose.x))
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


# Workspace / path setup

def build_workspace_and_paths(block_xy=(20, 80), goal_xy=(90, 90), show_plot: bool = True):
    ws = RobotWorkspace()
    if not show_plot: plt.ioff()

    block_x, block_y = block_xy; goal_x, goal_y = goal_xy

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


# Chunk representation
@dataclass
class Chunk:
    cmd: str  # 'turn_to_heading' | 'drive_for' | 'pause' | 'stop'
    heading_deg: Optional[float] = None
    distance_mm: Optional[float] = None
    turn_speed_pct: float = 25
    drive_speed_pct: float = 20
    wait_s: float = 0
    note: str = ""

# Controller: convert current pose + path geometry -> short chunk
def compute_chunk_from_pose(
    pose: Pose, info: Dict,
    heading_error_rad: float,
    nominal_speed: float = 80,
    turn_threshold_deg: float = 25,
    drive_chunk_mm: float = 5,
    min_drive_mm: float = 1,
) -> Tuple[Chunk, Dict]:
    """
    Convert the current pure-pursuit geometry into one short executable chunk.

    Policy:
      - If heading error is large, issue one short turn chunk.
      - Otherwise issue one short drive chunk whose length is limited.
    """
    goal_pt = info["goal_global"]
    x_r, y_r = info["goal_local"]
    curvature = float(info["curvature"])

    heading_err_deg = abs(rad2deg(heading_error_rad))
    target_heading_deg = rad2deg(heading_to_point(pose, goal_pt)) % 360

    if heading_err_deg > turn_threshold_deg:
        chunk = Chunk(
            cmd="turn_to_heading",
            heading_deg=round(target_heading_deg, 2),
            turn_speed_pct=25,
            note="large heading error -> turn first",
        )
        debug = {
            "event": "turn_in_place",
            "curvature": 0,
            "goal_local": (x_r, y_r),
            "target_heading_deg": target_heading_deg,
            "heading_error_deg": rad2deg(heading_error_rad),
            "v_cmd": 0,
            "omega_cmd": 2 * heading_error_rad,
        }
        return chunk, debug

    v = nominal_speed
    omega = v * curvature
    if abs(curvature) > 0.03:
        v *= 0.7; omega = v * curvature

    # Use only a short horizon so we can re-localize and re-plan frequently.
    drive_dist = clamp(v * 0.05, min_drive_mm, drive_chunk_mm)

    chunk = Chunk(
        cmd="drive_for",
        distance_mm=round(drive_dist, 2),
        drive_speed_pct=25,
        note="small heading error -> drive short chunk",
    )
    debug = {
        "event": "track_path",
        "curvature": curvature,
        "goal_local": (x_r, y_r),
        "target_heading_deg": target_heading_deg,
        "heading_error_deg": rad2deg(heading_error_rad),
        "v_cmd": v,
        "omega_cmd": omega,
    }
    return chunk, debug


# Executors

class PCChunkExecutor:
    """
    PC simulation executor.

    This is directly runnable with the existing Python files
    It simulates short chunks using simple kinematics and snaps heading from
    the executed turn chunk, which mirrors the intended real-world use of an
    inertial-corrected heading estimate.
    """

    def __init__(self, turn_rate_deg_per_s: float = 120):
        self.turn_rate_deg_per_s = float(turn_rate_deg_per_s)

    def execute(self, pose: Pose, chunk: Chunk) -> Tuple[Pose, Dict]:
        if chunk.cmd == "turn_to_heading":
            assert chunk.heading_deg is not None
            target = deg2rad(chunk.heading_deg)
            current = wrap2pi(pose.theta)
            dtheta = wrap2pi(target - current)
            duration = abs(rad2deg(dtheta)) / max(self.turn_rate_deg_per_s, 1e-6)
            theta_noise = 0
            theta_noise = np.deg2rad(np.random.randn() * 5)  # ±5 deg noise
            
            new_pose = Pose(pose.x, pose.y, target+theta_noise)
            meas = {
                "theta_meas": target,
                "distance_executed": 0,
                "duration_s": duration,
            }
            return new_pose, meas

        if chunk.cmd == "drive_for":
            assert chunk.distance_mm is not None
            d = float(chunk.distance_mm)
            d = d * (1 + 0.1 * np.random.randn())  # ±10% noise
            new_x = pose.x + d * math.cos(pose.theta)
            new_y = pose.y + d * math.sin(pose.theta)
            new_pose = Pose(new_x, new_y, pose.theta)
            meas = {
                "theta_meas": pose.theta,
                "distance_executed": d,
                "duration_s": d / 50,
            }
            return new_pose, meas

        if chunk.cmd == "pause":
            return Pose(pose.x, pose.y, pose.theta), {
                "theta_meas": pose.theta,
                "distance_executed": 0,
                "duration_s": chunk.wait_s,
            }

        if chunk.cmd == "stop":
            return Pose(pose.x, pose.y, pose.theta), {
                "theta_meas": pose.theta,
                "distance_executed": 0,
                "duration_s": 0,
            }

        raise ValueError(f"Unsupported chunk cmd: {chunk.cmd}")



# Pose estimation update

def update_pose_estimate_from_measurement(prev_pose: Pose, executed_chunk: Chunk, meas: Dict) -> Pose:
    """
    Minimal pose estimator:
      - heading always corrected from measured heading (e.g. inertial)
      - distance comes from executed chunk / odometry-like measurement
    """
    theta = wrap2pi(float(meas["theta_meas"]))

    if executed_chunk.cmd == "turn_to_heading":
        return Pose(prev_pose.x, prev_pose.y, theta)

    if executed_chunk.cmd == "drive_for":
        d = float(meas["distance_executed"])
        x = prev_pose.x + d * math.cos(theta)
        y = prev_pose.y + d * math.sin(theta)
        return Pose(x, y, theta)

    return Pose(prev_pose.x, prev_pose.y, theta)



# Closed-loop main
def run_chunked_closed_loop(
    block_xy=(20, 80),
    goal_xy=(90, 90),
    lookahead_phase1: float = 20,
    lookahead_phase2: float = 60,
    nominal_speed: float = 80,
    turn_threshold_deg: float = 25,
    drive_chunk_mm: float = 5,
    block_tolerance: float = 5,
    goal_tolerance: float = 5,
    pickup_pause_iters: int = 5,
    max_iters: int = 300,
    enable_workspace_update: bool = True,
    sleep_per_iter: float = 1,
    verbose: bool = True,
):
    env = build_workspace_and_paths(block_xy=block_xy, goal_xy=goal_xy)

    ws = env["ws"]
    block = env["block"]
    goal = env["goal"]
    path1 = env["path1"]
    path2 = env["path2"]

    pose_est = Pose(x=float(ws.x0), y=float(ws.y0), theta=0)
    pose_exec = Pose(x=float(ws.x0), y=float(ws.y0), theta=0)

    tracker = PurePursuitTracker(path1, lookahead_phase1)
    executor = PCChunkExecutor()

    phase = 1
    picked_up = False
    pickup_pause_remaining = 0

    traj_est = []
    traj_exec = []
    loop_logs = []
    success = False

    for it in range(max_iters):
        active_path = path1 if phase == 1 else path2
        tracker.path = active_path
        phase_name = "PHASE 1: TO BLOCK" if phase == 1 else "PHASE 2: TO GOAL"
        final_target = block if phase == 1 else goal

        if pickup_pause_remaining > 0:
            pickup_pause_remaining -= 1
            chunk = Chunk(cmd="pause", wait_s=sleep_per_iter, note="pickup pause")
            new_pose_exec, meas = executor.execute(pose_exec, chunk)
            pose_exec = new_pose_exec
            pose_est = update_pose_estimate_from_measurement(pose_est, chunk, meas)
            info = None
            heading_error = 0
            debug = {
                "event": "pickup_pause",
                "curvature": 0,
                "goal_local": (0, 0),
                "target_heading_deg": rad2deg(pose_est.theta),
                "heading_error_deg": 0,
                "v_cmd": 0,
                "omega_cmd": 0,
            }
        else:
            info = tracker.step(pose_est)
            goal_pt = info["goal_global"]
            heading_goal = heading_to_point(pose_est, goal_pt)
            heading_error = wrap2pi(heading_goal - pose_est.theta)

            chunk, debug = compute_chunk_from_pose(
                pose=pose_est,
                info=info,
                heading_error_rad=heading_error,
                nominal_speed=nominal_speed,
                turn_threshold_deg=turn_threshold_deg,
                drive_chunk_mm=drive_chunk_mm,
            )

            new_pose_exec, meas = executor.execute(pose_exec, chunk)
            pose_exec = new_pose_exec
            pose_est = update_pose_estimate_from_measurement(pose_est, chunk, meas)

        if enable_workspace_update:
            try:
                ws.update_robot(pose_est.x, pose_est.y)
            except Exception:
                pass

        traj_est.append([pose_est.x, pose_est.y, pose_est.theta])
        traj_exec.append([pose_exec.x, pose_exec.y, pose_exec.theta])

        dist_to_target = distance_xy(pose_est, final_target)
        switch_event = None

        if phase == 1 and dist_to_target < block_tolerance and not picked_up:
            picked_up = True
            phase = 2
            pickup_pause_remaining = pickup_pause_iters
            tracker = PurePursuitTracker(path2, lookahead_phase2)
            switch_event = "reached_block"
            if verbose:
                print("Reached block region. Switching to phase 2 after pickup pause.")

        elif phase == 2 and dist_to_target < goal_tolerance:
            switch_event = "reached_goal"
            success = True
            if verbose:
                print("Reached goal region.")

        log = {
            "iter": it,
            "phase": phase_name,
            "x_est": pose_est.x,
            "y_est": pose_est.y,
            "theta_est_deg": rad2deg(pose_est.theta),
            "x_exec": pose_exec.x,
            "y_exec": pose_exec.y,
            "theta_exec_deg": rad2deg(pose_exec.theta),
            "chunk_cmd": chunk.cmd,
            "chunk_heading_deg": chunk.heading_deg,
            "chunk_distance_mm": chunk.distance_mm,
            "heading_error_deg": debug["heading_error_deg"],
            "target_heading_deg": debug["target_heading_deg"],
            "curvature": debug["curvature"],
            "v_cmd": debug["v_cmd"],
            "omega_cmd": debug["omega_cmd"],
            "event": debug["event"],
            "dist_to_target": dist_to_target,
            "switch_event": switch_event,
        }
        loop_logs.append(log)

        if verbose:
            print(
                f"ITER={it:03d}, {phase_name}, "
                f"POSE_EST=({pose_est.x:.2f}, {pose_est.y:.2f}, {rad2deg(pose_est.theta):.2f} deg), "
                f"CMD={chunk.cmd}, "
                f"HEAD_ERR={debug['heading_error_deg']:.2f} deg, "
                f"KAPPA={debug['curvature']:.5f}, "
                f"DIST_TO_TARGET={dist_to_target:.2f}"
            )

        if switch_event == "reached_goal":
            break

        if sleep_per_iter > 0:
            plt.pause(sleep_per_iter)
            time.sleep(min(sleep_per_iter, 0.5))

    traj_est = np.asarray(traj_est, dtype=float) if traj_est else np.zeros((0, 3), dtype=float)
    traj_exec = np.asarray(traj_exec, dtype=float) if traj_exec else np.zeros((0, 3), dtype=float)

    return {
        "ws": ws,
        "block": block,
        "goal": goal,
        "path1": path1,
        "path2": path2,
        "trajectory_est": traj_est,
        "trajectory_exec": traj_exec,
        "logs": loop_logs,
        "final_pose_est": pose_est,
        "final_pose_exec": pose_exec,
        "success": success,
    }



# Plotting

def plot_chunked_closed_loop_result(result, save_path: Optional[str] = None, show: bool = True):
    path1 = result["path1"]
    path2 = result["path2"]
    block = result["block"]
    goal = result["goal"]
    traj_est = result["trajectory_est"]
    traj_exec = result["trajectory_exec"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(path1[:, 0], path1[:, 1], "--", linewidth=2, label="Path to Block")
    ax.plot(path2[:, 0], path2[:, 1], "--", linewidth=2, label="Path to Goal")

    if len(traj_exec) > 0:
        ax.plot(traj_exec[:, 0], traj_exec[:, 1], linewidth=2, label="Executed Trajectory")
        ax.scatter(traj_exec[0, 0], traj_exec[0, 1], marker="D", s=70, label="Start")
        ax.scatter(traj_exec[-1, 0], traj_exec[-1, 1], marker="s", s=70, label="Final")

    if len(traj_est) > 0:
        ax.plot(traj_est[:, 0], traj_est[:, 1], ":", linewidth=2, label="Estimated Trajectory")

    ax.scatter(block[0], block[1], marker="o", s=120, label="Block")
    ax.scatter(goal[0], goal[1], marker="*", s=160, label="Goal")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_title("Chunked Closed Loop Result")

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



# CSV logging


def save_logs_csv(logs, csv_path: str):
    import csv

    if not logs:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)



# Main


if __name__ == "__main__":
    result = run_chunked_closed_loop(
        block_xy=(20, 80),
        goal_xy=(90, 90),
        lookahead_phase1=20,
        lookahead_phase2=60,
        nominal_speed=80,
        turn_threshold_deg=25,
        drive_chunk_mm=5,
        block_tolerance=5,
        goal_tolerance=5,
        pickup_pause_iters=5,
        max_iters=300,
        enable_workspace_update=True,
        sleep_per_iter=1,
        verbose=True,
    )

    print("\nChunked closed loop success:", result["success"])
    print("Final estimated pose:", result["final_pose_est"])
    print("Final executed pose:", result["final_pose_exec"])

    save_logs_csv(result["logs"], "chunked_closed_loop_log.csv")
    plot_chunked_closed_loop_result(result, save_path="chunked_closed_loop_result.png", show=True)
