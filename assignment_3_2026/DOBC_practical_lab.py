import numpy as np
import os

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
# IMPORTANT — COMMAND SCALE:
#   Confirm with lab staff whether the interface expects m/s or cm/s.
#   - If m/s  → keep gains as-is, MAX_CMD = 1.0 (same as simulator)
#   - If cm/s → multiply all Kp/Ki/Kd gains by 100, MAX_CMD = 100.0
#
#   Current gains assume m/s output (consistent with simulator tuning).
#   The Tello SDK internally uses cm/s (-100 to 100), so if the lab wrapper
#   passes through raw SDK values, SCALE_TO_CMS = True below.
# =============================================================================

SCALE_TO_CMS = False    # set True if lab interface expects cm/s, not m/s
SCALE_FACTOR = 100.0 if SCALE_TO_CMS else 1.0

#Need to update these value
Kp_xy = 1.8551
Kp_z  = 2.5
Ki_xy = 0.02
Ki_z  = 0.001
Kd_xy = 1.0     # disabled for first flights — re-enable once stable
Kd_z  = 0.0675
Kyaw  = 0.4779
DERIV_ALPHA = 0.8

# DOBC — always on for real drone (no wind_enabled flag)
DOBC_ENABLED = True
DOBC_L_xy    = 0.0062    # keep very conservative on real hardware
DOBC_L_z     = 0.0047
DOBC_L_yaw   = 0.0000      # disabled until xy is proven stable

INT_LIM_XY = 0.1
INT_LIM_Z  = 0.1

# MAX_CMD: read from lab environment if set, otherwise default to 1.0 m/s.
# The lab wrapper may inject MAX_SPEED as a global — this reads it safely.
MAX_CMD = float(globals().get("MAX_SPEED", 1.0)) if not SCALE_TO_CMS \
          else float(globals().get("MAX_SPEED", 100.0))

# =============================================================================
# Persistent state
# =============================================================================
prev_ex  = 0.0;  prev_ey  = 0.0;  prev_ez  = 0.0
int_ex   = 0.0;  int_ey   = 0.0;  int_ez   = 0.0
filt_dex = 0.0;  filt_dey = 0.0;  filt_dez = 0.0

dobc_d_hat_global  = np.zeros(4)
dobc_prev_pos      = None
dobc_prev_cmd_body = np.zeros(4)

prev_timestamp_ms  = None
log_header_written = False      # guards CSV header — fixed: declared global in function
prev_target        = None       # used to detect target changes and reset integrals

# =============================================================================
# Helpers
# =============================================================================

def clamp(value, low, high):
    return max(low, min(high, value))

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_world_to_yaw_frame(yaw):
    """Rotate 2D world-frame vector into yaw-aligned body frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])

def reset_integral_state():
    """Reset integral and derivative state — call on target change."""
    global int_ex, int_ey, int_ez
    global prev_ex, prev_ey, prev_ez
    global filt_dex, filt_dey, filt_dez
    int_ex   = 0.0;  int_ey   = 0.0;  int_ez   = 0.0
    prev_ex  = 0.0;  prev_ey  = 0.0;  prev_ez  = 0.0
    filt_dex = 0.0;  filt_dey = 0.0;  filt_dez = 0.0

def reset_dobc_state():
    """Reset DOBC observer — call on target change to avoid stale estimates."""
    global dobc_d_hat_global, dobc_prev_pos, dobc_prev_cmd_body
    dobc_d_hat_global  = np.zeros(4)
    dobc_prev_pos      = None
    dobc_prev_cmd_body = np.zeros(4)

# =============================================================================
# Controller — signature matches real drone lab interface
# =============================================================================

def controller(state, target_pos, timestamp):
    """
    LQR position controller with DOBC disturbance compensation.
    Designed for DJI Tello via Vicon motion capture.

    Args:
        state      : [x, y, z, roll, pitch, yaw]  (m, rad) from Vicon
        target_pos : (x, y, z, yaw)               (m, rad)
        timestamp  : current time in milliseconds

    Returns:
        (vx, vy, vz, yaw_rate) — units depend on SCALE_TO_CMS flag above
    """

    global prev_ex, prev_ey, prev_ez, int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_pos, dobc_prev_cmd_body
    global prev_timestamp_ms, log_header_written, prev_target  # BUG FIX: log_header_written declared global

    # =========================================================================
    # Detect target change — reset integral and DOBC to avoid lurch
    # =========================================================================
    # Without this, integral error from the old waypoint carries into the
    # new one, causing an initial overshoot on real hardware.
    current_target_key = tuple(target_pos)
    if prev_target is not None and current_target_key != prev_target:
        reset_integral_state()
        reset_dobc_state()
    prev_target = current_target_key

    # =========================================================================
    # Timestep — computed from Vicon timestamps, fallback to 0.02 s
    # =========================================================================
    dt = 0.02   # safe default
    if prev_timestamp_ms is not None:
        raw_dt = (timestamp - prev_timestamp_ms) / 1000.0
        if 0.001 < raw_dt < 0.1:    # reject outliers (dropped frames, startup spikes)
            dt = raw_dt
    prev_timestamp_ms = timestamp

    # =========================================================================
    # Extract state
    # =========================================================================
    x, y, z = state[0], state[1], state[2]
    yaw      = state[5]
    x_ref, y_ref, z_ref, yaw_ref = target_pos

    # =========================================================================
    # LQR — PID velocity command in yaw-body frame
    # =========================================================================

    # World-frame position errors
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez       = z_ref - z
    eyaw     = wrap_angle(yaw_ref - yaw)

    # Rotate horizontal error into yaw-body frame
    # (Tello velocity commands are in body frame — errors must match)
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Filtered derivative
    filt_dex = (1.0 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * ((ex - prev_ex) / dt)
    filt_dey = (1.0 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * ((ey - prev_ey) / dt)
    filt_dez = (1.0 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * ((ez - prev_ez) / dt)

    prev_ex = ex;  prev_ey = ey;  prev_ez = ez

    # Integral with anti-windup clamp
    int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)

    # PID command — Kd terms are 0.0 for initial flights (noise safety)
    vx_cmd       = Kp_xy * ex + Ki_xy * int_ex + Kd_xy * filt_dex
    vy_cmd       = Kp_xy * ey + Ki_xy * int_ey + Kd_xy * filt_dey
    vz_cmd       = Kp_z  * ez + Ki_z  * int_ez + Kd_z  * filt_dez
    yaw_rate_cmd = Kyaw * eyaw

    lqr_cmd   = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=float)
    final_cmd = lqr_cmd.copy()
    d_hat_body = np.zeros(4)

    # =========================================================================
    # DOBC — disturbance estimation and compensation
    # =========================================================================
    # Always active on real drone — real-world drag, battery sag, and airflow
    # all produce persistent disturbances that the LQR alone won't cancel.
    # =========================================================================

    if DOBC_ENABLED:
        current_pos_4 = np.array([x, y, z, yaw], dtype=float)

        if dobc_prev_pos is None:
            # First call: initialise memory only
            dobc_prev_pos      = current_pos_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()

        else:
            # Step 1: rotate previous body command into global frame
            prev_yaw = dobc_prev_pos[3]
            c_p, s_p = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global    = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c_p * dobc_prev_cmd_body[0] - s_p * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s_p * dobc_prev_cmd_body[0] + c_p * dobc_prev_cmd_body[1]

            # Step 2: kinematic prediction
            # p_pred[k] = p[k-1] + (v_cmd_global[k-1] + d_hat[k-1]) * dt
            # Note: commands here are in m/s (pre-scaling), so prediction is in metres
            predicted_pos = dobc_prev_pos + (prev_cmd_global + dobc_d_hat_global) * dt

            # Step 3: innovation (actual - predicted) in global frame
            innovation    = current_pos_4 - predicted_pos
            innovation[3] = wrap_angle(innovation[3])

            # Step 4: update disturbance estimate
            L = np.array([DOBC_L_xy, DOBC_L_xy, DOBC_L_z, DOBC_L_yaw])
            dobc_d_hat_global = dobc_d_hat_global + L * innovation
            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3],  -0.1, 0.1)

            # Step 5: rotate disturbance estimate into yaw-body frame
            c_n, s_n = np.cos(yaw), np.sin(yaw)
            d_hat_body    = dobc_d_hat_global.copy()
            d_hat_body[0] =  c_n * dobc_d_hat_global[0] + s_n * dobc_d_hat_global[1]
            d_hat_body[1] = -s_n * dobc_d_hat_global[0] + c_n * dobc_d_hat_global[1]

            # Step 6: compensate LQR command (pre-scaling — kept in m/s)
            final_cmd = lqr_cmd - d_hat_body

            # Store final (compensated) command — not lqr_cmd
            dobc_prev_pos      = current_pos_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()

    else:
        dobc_d_hat_global[:] = 0.0

    # =========================================================================
    # Scale to interface units if needed (m/s → cm/s)
    # =========================================================================
    # DOBC prediction is always kept in m/s internally for physical consistency.
    # Only the output is scaled.
    output_cmd = final_cmd * SCALE_FACTOR

    # =========================================================================
    # CSV logging
    # =========================================================================
    csv_filename = 'flight_data_log.csv'
    try:
        # BUG FIX: log_header_written declared global above so assignment persists
        if not log_header_written:
            write_mode = 'a' if os.path.exists(csv_filename) else 'w'
            with open(csv_filename, write_mode) as f:
                if write_mode == 'w':
                    f.write("timestamp_ms,x,y,z,yaw,"
                            "target_x,target_y,target_z,target_yaw,"
                            "vx_out,vy_out,vz_out,yaw_rate_out,"
                            "d_hat_x,d_hat_y,d_hat_z\n")
            log_header_written = True

        with open(csv_filename, 'a') as f:
            row = np.hstack((
                timestamp,
                state[0:3], yaw,
                np.array(target_pos),
                output_cmd,
                dobc_d_hat_global[0:3]
            ))
            np.savetxt(f, [row], delimiter=',', fmt="%.6f")

    except Exception as e:
        pass    # never let logging crash the controller on real hardware

    # =========================================================================
    # Clamp and return
    # =========================================================================
    vx_out       = clamp(float(output_cmd[0]), -MAX_CMD, MAX_CMD)
    vy_out       = clamp(float(output_cmd[1]), -MAX_CMD, MAX_CMD)
    vz_out       = clamp(float(output_cmd[2]), -MAX_CMD, MAX_CMD)
    yaw_rate_out = clamp(float(output_cmd[3]), -MAX_CMD, MAX_CMD)

    return vx_out, vy_out, vz_out, yaw_rate_out
