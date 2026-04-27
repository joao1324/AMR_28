# wind_flag = True

import numpy as np
import os

# =============================================================================
# TUNABLE PARAMETERS — PSO can edit values in this block
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

Q_xy       = 1.0
R_xy       = 2.8
Q_z        = 1.5
R_z        = 2.3

# More conservative gains -> Kp_xy=0.8 and kp_z=1.2
# Kp_xy = 0.8000
# Kp_z = 1.2000
# Kyaw = 1.0000
# Kd_xy = 0.1500
# Kd_z = 0.2346
# INT_LIM_XY = 0.1522
# INT_LIM_Z = 0.8000
# DERIV_ALPHA = 0.5000
# Ki_xy = 0.0326
# Ki_z = 0.0447
# DOBC_L_xy = 0.0127
# DOBC_L_z = 0.0102
# DOBC_L_yaw = 0.0054

# More agressive gains -> Kp_xy=1 and kp_z=1.5
Kp_xy = 0.7580
Kp_z = 1.5000
Kyaw = 1.0000
Kd_xy = 0.1500
Kd_z = 0.2500
INT_LIM_XY = 0.1000
INT_LIM_Z = 0.8000
DERIV_ALPHA = 0.5000
Ki_xy = 0.0382
Ki_z = 0.0500
DOBC_L_xy = 0.0146
DOBC_L_z = 0.0108
DOBC_L_yaw = 0.0057

#Gain from last lab
# Kp_xy = 1.8551
# Kp_z  = 2.5
# Ki_xy = 0.02
# Ki_z  = 0.001
# Kd_xy = 1.0     # disabled for first flights — re-enable once stable
# Kd_z  = 0.0675
# Kyaw  = 0.4779
# DERIV_ALPHA = 0.8
## DOBC — always on for real drone (no wind_enabled flag)
# DOBC_ENABLED = True
# DOBC_L_xy    = 0.0062    # keep very conservative on real hardware
# DOBC_L_z     = 0.0047
# DOBC_L_yaw   = 0.0000      # disabled until xy is proven stable
# INT_LIM_XY = 0.1
# INT_LIM_Z  = 0.1

# =============================================================================
# LQR state (position PID in body frame)
# =============================================================================
prev_ex  = 0.0
prev_ey  = 0.0
prev_ez  = 0.0
int_ex   = 0.0
int_ey   = 0.0
int_ez   = 0.0
filt_dex = 0.0
filt_dey = 0.0
filt_dez = 0.0

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
# Logging state (simulation time-based)
# =============================================================================
log_header_written = False
log_time_s         = 0.0   # accumulate dt each call

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
    2D rotation from world frame to yaw-aligned body frame.
    Matches inverted_quat_yaw in run.py.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])

# =============================================================================
# Main controller
# =============================================================================

def controller(state, target_pos, dt, wind_enabled=False):
    """
    LQR position controller + disturbance observer for wind (simulation).

    Args:
        state      : [x, y, z, roll, pitch, yaw] (m, rad)
        target_pos : (x, y, z, yaw)              (m, rad)
        dt         : controller timestep (s)
        wind_enabled : if True, run disturbance observer

    Returns:
        (vx_set, vy_set, vz_set, yaw_rate_set) in m/s and rad/s
    """

    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_state, dobc_prev_cmd_body
    global debug_last_lqr_cmd, debug_last_final_cmd
    global debug_last_d_hat_body, debug_last_innovation
    global log_header_written, log_time_s

    # -------------------------------------------------------------------------
    # Unpack current state and target
    # -------------------------------------------------------------------------
    x, y, z = state[0], state[1], state[2]
    yaw     = state[5]

    x_ref, y_ref, z_ref, yaw_ref = target_pos

    # -------------------------------------------------------------------------
    # LQR: PID velocity command in yaw-body frame
    # -------------------------------------------------------------------------

    # Position errors in world frame
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez       = z_ref - z
    eyaw     = wrap_angle(yaw_ref - yaw)

    # Rotate horizontal error into body frame (commands are body-frame)
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Derivative (backward difference) with low-pass filtering
    filt_dex = (1.0 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * ((ex - prev_ex) / dt)
    filt_dey = (1.0 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * ((ey - prev_ey) / dt)
    filt_dez = (1.0 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * ((ez - prev_ez) / dt)

    # Integral with clamping
    int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)

    # PID velocity command in body frame
    vx_cmd       = Kp_xy * ex + Ki_xy * int_ex + Kd_xy * filt_dex
    vy_cmd       = Kp_xy * ey + Ki_xy * int_ey + Kd_xy * filt_dey
    vz_cmd       = Kp_z  * ez + Ki_z  * int_ez + Kd_z  * filt_dez
    yaw_rate_cmd = Kyaw  * eyaw

    lqr_cmd = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=float)
    debug_last_lqr_cmd = lqr_cmd.copy()

    prev_ex = ex
    prev_ey = ey
    prev_ez = ez

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
        current_state_4 = np.array([x, y, z, yaw], dtype=float)

        if dobc_prev_state is None:
            # First tick with wind: just store state and command
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()
            debug_last_d_hat_body = np.zeros(4)
            debug_last_innovation = np.zeros(4)
            final_cmd = lqr_cmd

        else:
            # 1) Rotate previous body-frame command into world frame
            prev_yaw = dobc_prev_state[3]
            c_p, s_p = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global    = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c_p * dobc_prev_cmd_body[0] - s_p * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s_p * dobc_prev_cmd_body[0] + c_p * dobc_prev_cmd_body[1]
            # z and yaw_rate components unchanged

            # 2) Predict next state using previous command + disturbance estimate
            #    p_pred = p_prev + (u_prev_global + d_hat_prev) * dt
            predicted_state = dobc_prev_state + (prev_cmd_global + dobc_d_hat_global) * dt

            # 3) Innovation = measured state − predicted state
            innovation    = current_state_4 - predicted_state
            innovation[3] = wrap_angle(innovation[3])
            debug_last_innovation = innovation.copy()

            # 4) Update disturbance estimate in world frame
            DOBC_L = np.array([DOBC_L_xy, DOBC_L_xy, DOBC_L_z, DOBC_L_yaw], dtype=float)
            dobc_d_hat_global = dobc_d_hat_global + DOBC_L * innovation

            # Limit disturbance to realistic values for this drone
            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3],  -0.1, 0.1)

            # 5) Rotate disturbance estimate into body frame
            c_n, s_n = np.cos(yaw), np.sin(yaw)
            d_hat_body    = dobc_d_hat_global.copy()
            d_hat_body[0] =  c_n * dobc_d_hat_global[0] + s_n * dobc_d_hat_global[1]
            d_hat_body[1] = -s_n * dobc_d_hat_global[0] + c_n * dobc_d_hat_global[1]
            debug_last_d_hat_body = d_hat_body.copy()

            # 6) Subtract disturbance estimate from the LQR command
            final_cmd = lqr_cmd - d_hat_body

            # Store what we actually sent and the state we saw
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()

    debug_last_final_cmd = final_cmd.copy()

    # -------------------------------------------------------------------------
    # CSV logging (simulation time-based)
    # -------------------------------------------------------------------------
    csv_filename = "flight_data_log_28.csv"
    log_time_s += dt  # accumulate simulated time

    try:
        if not log_header_written:
            file_exists = os.path.exists(csv_filename)
            with open(csv_filename, "a" if file_exists else "w") as f:
                if not file_exists:
                    f.write(
                        "t_s,x,y,z,yaw,"
                        "target_x,target_y,target_z,target_yaw,"
                        "vx_cmd,vy_cmd,vz_cmd,yaw_rate_cmd,"
                        "d_hat_x,d_hat_y,d_hat_z\n"
                    )
            log_header_written = True

        with open(csv_filename, "a") as f:
            row = np.array([
                log_time_s,
                state[0], state[1], state[2], yaw,
                target_pos[0], target_pos[1], target_pos[2], target_pos[3],
                final_cmd[0], final_cmd[1], final_cmd[2], final_cmd[3],
                dobc_d_hat_global[0], dobc_d_hat_global[1], dobc_d_hat_global[2],
            ], dtype=float)
            np.savetxt(f, [row], delimiter=",", fmt="%.6f")
    except Exception:
        # Never let logging crash the controller in sim
        pass

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
