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
    if not show_plot:
        plt.ioff()

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



# Command representation


@dataclass
class WheelCommand:
    left_vel: float          # mm/s
    right_vel: float         # mm/s
    dt: float = 0.05         # s
    note: str = ""



# Differential-drive conversion


def vw_to_wheel_velocities(v: float, omega: float, track_width_mm: float) -> Tuple[float, float]:
    """
    Convert body linear/angular velocity to left/right wheel linear velocity.

    v      : robot center linear velocity [mm/s]
    omega  : robot angular velocity [rad/s]
    L      : track width [mm]

    v_l = v - (L/2)*omega
    v_r = v + (L/2)*omega
    """
    left_vel = v - 0.5 * track_width_mm * omega
    right_vel = v + 0.5 * track_width_mm * omega
    return left_vel, right_vel



# Controller: Pure Pursuit -> wheel velocities


def compute_wheel_command_from_pose(
    pose: Pose, info: Dict,
    heading_error_rad: float,
    track_width_mm: float,
    nominal_speed: float = 80,         # mm/s
    turn_in_place_gain: float = 2,     # rad/s per rad error
    turn_threshold_deg: float = 8,
    max_wheel_speed: float = 150,      # mm/s
    dt: float = 0.05,
) -> Tuple[WheelCommand, Dict]:
    """
    Convert current pure-pursuit geometry into one short differential-drive command.
    """
    goal_pt = info["goal_global"]
    x_r, y_r = info["goal_local"]
    curvature = float(info["curvature"])

    heading_err_deg = rad2deg(heading_error_rad)
    target_heading_deg = rad2deg(heading_to_point(pose, goal_pt)) % 360

    # Base pure pursuit command
    v = nominal_speed
    omega = v * curvature

    '''
    # If heading error is large, prioritize turning
    if abs(heading_err_deg) > turn_threshold_deg:
        v = 0
        omega = turn_in_place_gain * heading_error_rad
        event = "turn_in_place"
    else:
        if abs(curvature) > 0.03:
            v *= 0.7
            omega = v * curvature
        event = "track_path"
    '''
    curvature_turn_threshold = 0.15  # change the THRESHOLD whatever we want

    if abs(heading_err_deg) > turn_threshold_deg or abs(curvature) > curvature_turn_threshold:
        v = 0
        omega = turn_in_place_gain * heading_error_rad
        event = "turn_in_place"
    else:
        v = nominal_speed
        omega = v * curvature
        event = "track_path"

    left_vel, right_vel = vw_to_wheel_velocities(v, omega, track_width_mm)

    left_vel = clamp(left_vel, -max_wheel_speed, max_wheel_speed)
    right_vel = clamp(right_vel, -max_wheel_speed, max_wheel_speed)

    cmd = WheelCommand(
        left_vel=float(left_vel),
        right_vel=float(right_vel),
        dt=float(dt),
        note="pure pursuit differential drive"
    )

    debug = {
        "event": event,
        "curvature": curvature,
        "goal_local": (x_r, y_r),
        "target_heading_deg": target_heading_deg,
        "heading_error_deg": heading_err_deg,
        "v_cmd": v,
        "omega_cmd": omega,
        "left_vel": left_vel,
        "right_vel": right_vel,
    }
    return cmd, debug



# Executor: simulate differential drive

class PCDifferentialDriveExecutor:
    """
    PC simulation executor for differential drive.
    Uses simple forward kinematics with a bit of noise.
    """

    def __init__(
        self,
        track_width_mm: float = 300,
        vel_noise_std_ratio: float = 0.02,
        omega_noise_std_deg: float = 1,
    ):
        self.track_width_mm = float(track_width_mm)
        self.vel_noise_std_ratio = float(vel_noise_std_ratio)
        self.omega_noise_std_deg = float(omega_noise_std_deg)

    def execute(self, pose: Pose, cmd: WheelCommand) -> Tuple[Pose, Dict]:
        vl = float(cmd.left_vel)
        vr = float(cmd.right_vel)
        dt = float(cmd.dt)

        # Differential-drive forward kinematics
        v = 0.5 * (vr + vl)
        omega = (vr - vl) / self.track_width_mm

        # Add some noise
        v_noisy = v * (1 + self.vel_noise_std_ratio * np.random.randn())
        omega_noisy = omega + np.deg2rad(np.random.randn() * self.omega_noise_std_deg)

        # Integrate motion
        new_theta = wrap2pi(pose.theta + omega_noisy * dt)
        new_x = pose.x + v_noisy * math.cos(pose.theta) * dt
        new_y = pose.y + v_noisy * math.sin(pose.theta) * dt

        new_pose = Pose(new_x, new_y, new_theta)

        meas = {
            "theta_meas": new_theta,
            "distance_executed": v_noisy * dt,
            "duration_s": dt,
            "v_exec": v_noisy,
            "omega_exec": omega_noisy,
            "left_vel": vl,
            "right_vel": vr,
        }
        return new_pose, meas



# Pose estimation update


def update_pose_estimate_from_measurement(prev_pose: Pose, cmd: WheelCommand, meas: Dict) -> Pose:
    """
    Minimal pose estimator:
      - heading corrected from measured heading
      - distance from executed motion
    """
    theta = wrap2pi(float(meas["theta_meas"]))
    d = float(meas["distance_executed"])

    x = prev_pose.x + d * math.cos(theta)
    y = prev_pose.y + d * math.sin(theta)

    return Pose(x, y, theta)



# Closed-loop main


def run_differential_drive_closed_loop(
    block_xy=(20, 80),
    goal_xy=(90, 90),
    lookahead_phase1: float = 20,
    lookahead_phase2: float = 60,
    nominal_speed: float = 80,
    turn_threshold_deg: float = 8,
    block_tolerance: float = 5,
    goal_tolerance: float = 5,
    pickup_pause_iters: int = 10,
    max_iters: int = 400,
    track_width_mm: float = 300, # mm
    dt: float = 0.05,
    enable_workspace_update: bool = True,
    sleep_per_iter: float = 0.05,
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
    executor = PCDifferentialDriveExecutor(track_width_mm=track_width_mm)

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

            wheel_cmd = WheelCommand(
                left_vel=0,
                right_vel=0,
                dt=dt,
                note="pickup pause"
            )

            new_pose_exec, meas = executor.execute(pose_exec, wheel_cmd)
            pose_exec = new_pose_exec
            pose_est = update_pose_estimate_from_measurement(pose_est, wheel_cmd, meas)

            debug = {
                "event": "pickup_pause",
                "curvature": 0,
                "goal_local": (0, 0),
                "target_heading_deg": rad2deg(pose_est.theta),
                "heading_error_deg": 0,
                "v_cmd": 0,
                "omega_cmd": 0,
                "left_vel": 0,
                "right_vel": 0,
            }

        else:
            info = tracker.step(pose_est)
            goal_pt = info["goal_global"]
            heading_goal = heading_to_point(pose_est, goal_pt)
            heading_error = wrap2pi(heading_goal - pose_est.theta)

            wheel_cmd, debug = compute_wheel_command_from_pose(
                pose=pose_est,
                info=info,
                heading_error_rad=heading_error,
                track_width_mm=track_width_mm,
                nominal_speed=nominal_speed,
                turn_threshold_deg=turn_threshold_deg,
                dt=dt,
            )

            new_pose_exec, meas = executor.execute(pose_exec, wheel_cmd)
            pose_exec = new_pose_exec
            pose_est = update_pose_estimate_from_measurement(pose_est, wheel_cmd, meas)

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
            "left_vel": debug["left_vel"],
            "right_vel": debug["right_vel"],
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
                f"VL={debug['left_vel']:.2f}, VR={debug['right_vel']:.2f}, "
                f"HEAD_ERR={debug['heading_error_deg']:.2f} deg, "
                f"CURVATURE={debug['curvature']:.5f}, "
                f"DIST_TO_TARGET={dist_to_target:.2f}"
            )

        if switch_event == "reached_goal":
            break

        if sleep_per_iter > 0:
            plt.pause(sleep_per_iter)
            time.sleep(min(sleep_per_iter, 0.05))

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


def plot_differential_drive_result(result, save_path: Optional[str] = None, show: bool = True):
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
    ax.set_title("Differential Drive Closed Loop Result")

    #if save_path:
    #    fig.savefig(save_path, dpi=200, bbox_inches="tight")

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
    result = run_differential_drive_closed_loop(
        block_xy=(20, 80),
        goal_xy=(90, 90),
        lookahead_phase1=20,
        lookahead_phase2=60,
        nominal_speed=80,
        turn_threshold_deg=8,
        block_tolerance=5,
        goal_tolerance=5,
        pickup_pause_iters=10,
        max_iters=400,
        track_width_mm=300,   # <-- replace with your real robot track width
        dt=0.05,
        enable_workspace_update=True,
        sleep_per_iter=0.05,
        verbose=True,
    )

    print("\nDifferential drive closed loop success:", result["success"])
    print("Final estimated pose:", result["final_pose_est"])
    print("Final executed pose:", result["final_pose_exec"])

    #save_logs_csv(result["logs"], "differential_drive_closed_loop_log.csv")
    plot_differential_drive_result(
        result,
        save_path="differential_drive_closed_loop_result.png",
        show=True
    )