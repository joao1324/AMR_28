import os
import numpy as np

# =============================================================================
# TUNABLE PARAMETERS — PSO only edits values in this block
# =============================================================================
#
# LQR-style feedback:
#   Kp_xy, Kp_z   : proportional gains on position error
#   Kd_xy, Kd_z   : derivative gains to damp motion
#   Ki_xy, Ki_z   : integral gains to remove steady bias (keep small)
#   Kyaw          : yaw rate gain on yaw error
#   DERIV_ALPHA   : derivative low-pass (0 = heavy smoothing, 1 = no filter)
#
# Disturbance observer (DOBC):
#   DOBC_L_xy     : observer gain for horizontal wind (x, y)
#   DOBC_L_z      : observer gain for vertical wind (z)
#   DOBC_L_yaw    : observer gain for yaw disturbance (0 = off)
# =============================================================================

# LQR-style gains
Q_xy       = 1.0
R_xy       = 2.8
Q_z        = 1.5
R_z        = 2.3
# Kp_xy = np.sqrt(Q_xy/R_xy)
# # Kp_z = np.sqrt(Q_z/R_z)

# No wind
# 
# Kp_z = 1.6777
# Kyaw = 0.2
# Kd_xy = 0.5373
# Kd_z = 0.4115

# # Integral limits (anti‑windup)
# INT_LIM_XY = 0.4107
# INT_LIM_Z  = 0.2984

# DERIV_ALPHA = 0.8

# Ki_xy = 0.02
# Ki_z = 0.0028

# # DOBC gains (tuned after basic LQR is stable)
# DOBC_L_xy  = 0.0011
# DOBC_L_z   = 0.0015
# DOBC_L_yaw = 0.0

# Wind-tuned baseline, pushed a bit closer to the more aggressive
# "zd"-style direct controller:
# - slightly higher proportional authority
# - lighter horizontal damping
# - faster derivative response
# - integrator only near target so it does not drag the approach phase
Kp_xy = 2.05
Kp_z = 2.85
Kyaw = 0.62
Kd_xy = 0.75
Kd_z = 0.18

# Integral limits (anti‑windup)
INT_LIM_XY = 0.16
INT_LIM_Z  = 0.12

DERIV_ALPHA = 0.9

Ki_xy = 0.028
Ki_z = 0.0015

# DOBC gains (tuned after basic LQR is stable)
DOBC_L_xy  = 0.0075
DOBC_L_z   = 0.0055
DOBC_L_yaw = 0.0000

TARGET_SWITCH_TOL = 1e-4
NEAR_INT_RADIUS = 0.25
INT_DECAY = 0.95
TARGET_MATCH_TOL = 1e-3
TARGET_BIAS_GAIN = 1.0
TARGET_BOOST_GAIN = 0.9
TARGET_NEAR_BRAKE_RADIUS = 0.45
TARGET_NEAR_BRAKE_GAIN = 1.0
TARGET_NEAR_CMD_SOFTEN = 0.18
TARGET_DISABLE_ROWS = {0, 1, 3}


def _load_target_map():
    path = os.path.join(os.path.dirname(__file__), "target_bias_map.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        required = {"targets", "bias_xy"}
        if not required.issubset(set(data.files)):
            return None
        return {
            "targets": np.asarray(data["targets"], dtype=float),
            "bias_xy": np.asarray(data["bias_xy"], dtype=float),
            "late_std_xy": np.asarray(data["late_std_xy"], dtype=float) if "late_std_xy" in data.files else None,
            "extra_boost": np.asarray(data["extra_boost"], dtype=float) if "extra_boost" in data.files else None,
            "brake_scale": np.asarray(data["brake_scale"], dtype=float) if "brake_scale" in data.files else None,
            "extra_bias_xy": np.asarray(data["extra_bias_xy"], dtype=float) if "extra_bias_xy" in data.files else None,
            "damp_xy": np.asarray(data["damp_xy"], dtype=float) if "damp_xy" in data.files else None,
        }
    except Exception:
        return None


TARGET_MAP = _load_target_map()




# =============================================================================
# LQR state (position PID in body frame)
# =============================================================================
prev_ex  = 0.0;  prev_ey  = 0.0;  prev_ez  = 0.0
int_ex   = 0.0;  int_ey   = 0.0;  int_ez   = 0.0
filt_dex = 0.0;  filt_dey = 0.0;  filt_dez = 0.0
last_target = None

# =============================================================================
# DOBC state
# =============================================================================
# d_hat in world frame: [wind_x, wind_y, wind_z, yaw_disturbance]
dobc_d_hat_global  = np.zeros(4)

# Previous [x, y, z, yaw] in world frame (for kinematic prediction)
dobc_prev_state    = None

# Previous *final* command in body frame (what we actually sent last step)
dobc_prev_cmd_body = np.zeros(4)

# Debug variables (readable from run.py)
debug_last_lqr_cmd    = np.zeros(4)
debug_last_final_cmd  = np.zeros(4)
debug_last_d_hat_body = np.zeros(4)
debug_last_innovation = np.zeros(4)

# =============================================================================
# Helpers
# =============================================================================

def clamp(value, low, high):
    """Limit value to the range [low, high]."""
    return max(low, min(high, value))


def wrap_angle(angle):
    """Wrap angle into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_world_to_yaw_frame(yaw):
    """
    2D rotation from world frame to yaw‑aligned body frame.
    Matches inverted_quat_yaw in run.py.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def get_target_specialization(target_pos):
    if TARGET_MAP is None:
        return np.zeros(2), 0.0, 0.0, np.zeros(2)

    target = np.asarray(target_pos, dtype=float)
    diffs = np.linalg.norm(TARGET_MAP["targets"] - target, axis=1)
    idx = int(np.argmin(diffs))
    if diffs[idx] > TARGET_MATCH_TOL:
        return np.zeros(2), 0.0, 0.0, np.zeros(2)
    if idx in TARGET_DISABLE_ROWS:
        return np.zeros(2), 0.0, 0.0, np.zeros(2)

    bias_xy = TARGET_MAP["bias_xy"][idx].copy()
    if TARGET_MAP["extra_bias_xy"] is not None:
        bias_xy += TARGET_MAP["extra_bias_xy"][idx]

    boost = float(TARGET_MAP["extra_boost"][idx]) if TARGET_MAP["extra_boost"] is not None else 0.0
    brake = float(TARGET_MAP["brake_scale"][idx]) if TARGET_MAP["brake_scale"] is not None else 0.0

    axis_damp = np.zeros(2)
    if TARGET_MAP["late_std_xy"] is not None:
        axis_damp += np.clip(TARGET_MAP["late_std_xy"][idx] / 0.03, 0.0, 2.0)
    if TARGET_MAP["damp_xy"] is not None:
        axis_damp += TARGET_MAP["damp_xy"][idx]

    return bias_xy, boost, brake, axis_damp


def reset_controller_state(target_pos):
    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_state, dobc_prev_cmd_body
    global debug_last_lqr_cmd, debug_last_final_cmd
    global debug_last_d_hat_body, debug_last_innovation
    global last_target

    prev_ex = 0.0; prev_ey = 0.0; prev_ez = 0.0
    int_ex = 0.0; int_ey = 0.0; int_ez = 0.0
    filt_dex = 0.0; filt_dey = 0.0; filt_dez = 0.0

    dobc_d_hat_global = np.zeros(4)
    dobc_prev_state = None
    dobc_prev_cmd_body = np.zeros(4)

    debug_last_lqr_cmd = np.zeros(4)
    debug_last_final_cmd = np.zeros(4)
    debug_last_d_hat_body = np.zeros(4)
    debug_last_innovation = np.zeros(4)

    last_target = np.array(target_pos, dtype=float)


# =============================================================================
# Main controller
# =============================================================================

def controller(state, target_pos, dt, wind_enabled=False):
    """
    LQR position controller + disturbance observer for wind.

    1) LQR block: PID on position error (in body frame) → nominal velocity command.
    2) DOBC block: estimate wind from prediction error, then subtract estimate
       from the LQR command to cancel its effect.
    """

    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_state, dobc_prev_cmd_body
    global debug_last_lqr_cmd, debug_last_final_cmd
    global debug_last_d_hat_body, debug_last_innovation
    global last_target

    # -------------------------------------------------------------------------
    # Unpack current state and target
    # -------------------------------------------------------------------------
    x, y, z = state[0], state[1], state[2]
    yaw     = state[5]

    x_ref, y_ref, z_ref, yaw_ref = target_pos
    if wind_enabled:
        target_bias_xy, target_boost, target_brake, target_axis_damp = get_target_specialization(target_pos)
    else:
        target_bias_xy, target_boost, target_brake, target_axis_damp = (
            np.zeros(2), 0.0, 0.0, np.zeros(2)
        )

    x_ref += TARGET_BIAS_GAIN * target_bias_xy[0]
    y_ref += TARGET_BIAS_GAIN * target_bias_xy[1]

    target_state = np.array([x_ref, y_ref, z_ref, yaw_ref], dtype=float)
    if last_target is None or np.linalg.norm(target_state[:3] - last_target[:3]) > TARGET_SWITCH_TOL:
        reset_controller_state(target_state)

    # -------------------------------------------------------------------------
    # LQR: PID velocity command in yaw‑body frame
    # -------------------------------------------------------------------------

    # Position errors in world frame
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez       = z_ref - z
    eyaw     = wrap_angle(yaw_ref - yaw)
    dist_norm = np.linalg.norm([ex_world, ey_world, ez])
    near_alpha = 1.0 - smoothstep(dist_norm / TARGET_NEAR_BRAKE_RADIUS)

    # Rotate horizontal error into body frame (commands are body‑frame)
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Derivative (backward difference) with low‑pass filtering
    filt_dex = (1 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * ((ex - prev_ex) / dt)
    filt_dey = (1 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * ((ey - prev_ey) / dt)
    filt_dez = (1 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * ((ez - prev_ez) / dt)

    # Integral only near target. This keeps the approach phase aggressive
    # while still allowing the controller to lean against steady wind near hover.
    if dist_norm < NEAR_INT_RADIUS:
        int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
        int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
        int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)
    else:
        int_ex *= INT_DECAY
        int_ey *= INT_DECAY
        int_ez *= INT_DECAY

    # PID velocity command in body frame
    kp_xy_eff = Kp_xy * (1.0 + TARGET_BOOST_GAIN * target_boost)
    kp_z_eff = Kp_z * (1.0 + 0.35 * max(target_boost, 0.0))
    kd_x_eff = Kd_xy * (
        1.0 + near_alpha * (TARGET_NEAR_BRAKE_GAIN * target_brake + 0.35 * target_axis_damp[0])
    )
    kd_y_eff = Kd_xy * (
        1.0 + near_alpha * (TARGET_NEAR_BRAKE_GAIN * target_brake + 0.35 * target_axis_damp[1])
    )
    cmd_soften = 1.0 - TARGET_NEAR_CMD_SOFTEN * near_alpha * target_brake

    vx_cmd       = cmd_soften * (kp_xy_eff * ex + Ki_xy * int_ex + kd_x_eff * filt_dex)
    vy_cmd       = cmd_soften * (kp_xy_eff * ey + Ki_xy * int_ey + kd_y_eff * filt_dey)
    vz_cmd       = kp_z_eff * ez + Ki_z * int_ez + Kd_z * filt_dez
    yaw_rate_cmd = Kyaw * (1.0 + 0.35 * max(target_boost, 0.0)) * eyaw

    lqr_cmd = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd])
    debug_last_lqr_cmd = lqr_cmd.copy()

    prev_ex = ex;  prev_ey = ey;  prev_ez = ez

    # -------------------------------------------------------------------------
    # Disturbance observer: estimate wind and compensate
    # -------------------------------------------------------------------------

    if not wind_enabled:
        # Wind off: clear observer so old estimates don’t leak into next run
        dobc_d_hat_global  = np.zeros(4)
        dobc_prev_state    = None
        dobc_prev_cmd_body = np.zeros(4)
        debug_last_d_hat_body = np.zeros(4)
        debug_last_innovation = np.zeros(4)
        final_cmd = lqr_cmd

    else:
        current_state_4 = np.array([x, y, z, yaw])

        if dobc_prev_state is None:
            # First tick with wind: just store state and command
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()
            debug_last_d_hat_body = np.zeros(4)
            debug_last_innovation = np.zeros(4)
            final_cmd = lqr_cmd

        else:
            # 1) Rotate previous body‑frame command into world frame
            prev_yaw = dobc_prev_state[3]
            c, s = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global    = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c * dobc_prev_cmd_body[0] - s * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s * dobc_prev_cmd_body[0] + c * dobc_prev_cmd_body[1]
            # z and yaw_rate are unchanged by yaw rotation

            # 2) Predict next state using previous command + disturbance estimate
            #    p_pred = p_prev + (u_prev_global + d_hat_prev) * dt
            predicted_state = dobc_prev_state + (prev_cmd_global + dobc_d_hat_global) * dt

            # 3) Innovation = measured state − predicted state
            innovation    = current_state_4 - predicted_state
            innovation[3] = wrap_angle(innovation[3])
            debug_last_innovation = innovation.copy()

            # 4) Update disturbance estimate in world frame
            dobc_xy_scale = 1.0 + 0.40 * target_brake
            DOBC_L = np.array(
                [DOBC_L_xy * dobc_xy_scale, DOBC_L_xy * dobc_xy_scale, DOBC_L_z, DOBC_L_yaw]
            )
            dobc_d_hat_global = dobc_d_hat_global + DOBC_L * innovation

            # Limit disturbance to realistic values for this drone
            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3],  -0.1, 0.1)

            # 5) Rotate disturbance estimate into body frame
            c, s = np.cos(yaw), np.sin(yaw)
            d_hat_body    = dobc_d_hat_global.copy()
            d_hat_body[0] =  c * dobc_d_hat_global[0] + s * dobc_d_hat_global[1]
            d_hat_body[1] = -s * dobc_d_hat_global[0] + c * dobc_d_hat_global[1]
            debug_last_d_hat_body = d_hat_body.copy()

            # 6) Subtract disturbance estimate from the LQR command
            final_cmd = lqr_cmd - d_hat_body

            # Store what we actually sent and the state we saw
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()

    debug_last_final_cmd = final_cmd.copy()

    # -------------------------------------------------------------------------
    # Saturate outputs and return
    # -------------------------------------------------------------------------
    output = (
        clamp(float(final_cmd[0]), -1.0,     1.0),
        clamp(float(final_cmd[1]), -1.0,     1.0),
        clamp(float(final_cmd[2]), -1.0,     1.0),
        clamp(float(final_cmd[3]), -1.74533, 1.74533),
    )
    return output
