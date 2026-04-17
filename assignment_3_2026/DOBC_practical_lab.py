import numpy as np
import os

# =============================================================================
# TUNABLE PARAMETERS — start conservative for real drone
# =============================================================================
Kp_xy = 0.6
Kp_z  = 0.8
Ki_xy = 0.02
Ki_z  = 0.04
Kd_xy = 0.0   
Kd_z  = 0.0   
Kyaw  = 0.8   
DERIV_ALPHA = 0.2

# Disturbance observer
DOBC_ENABLED = True
DOBC_L_xy  = 0.001
DOBC_L_z   = 0.001
DOBC_L_yaw = 0.0

INT_LIM_XY = 0.5
INT_LIM_Z  = 0.5
MAX_CMD = float(globals().get("MAX_SPEED", 100.0))

# =============================================================================
# Persistent state
# =============================================================================
prev_ex  = 0.0; prev_ey  = 0.0; prev_ez  = 0.0
int_ex   = 0.0; int_ey   = 0.0; int_ez   = 0.0
filt_dex = 0.0; filt_dey = 0.0; filt_dez = 0.0

dobc_d_hat_global = np.zeros(4)
dobc_prev_pos     = None      
dobc_prev_cmd_body = np.zeros(4)  

prev_timestamp_ms = None
log_header_written = False # For clean CSV plotting

# =============================================================================
# Helpers
# =============================================================================
def clamp(value, low, high):
    return max(low, min(high, value))

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_world_to_yaw_frame(yaw):
    """Rotate a 2D vector from world frame into yaw-aligned body frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])

# =============================================================================
# Controller
# =============================================================================
def controller(state, target_pos, timestamp):
    global prev_ex, prev_ey, prev_ez, int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_pos, dobc_prev_cmd_body
    global prev_timestamp_ms, log_header_written

    # -------------------------------------------------------------------------
    # Time step calculation
    # -------------------------------------------------------------------------
    dt = None
    if prev_timestamp_ms is not None:
        raw_dt = (timestamp - prev_timestamp_ms) / 1000.0
        if 0.001 < raw_dt < 0.1:
            dt = raw_dt
    if dt is None:
        dt = 0.02
    prev_timestamp_ms = timestamp

    # -------------------------------------------------------------------------
    # Extract state and target
    # -------------------------------------------------------------------------
    x, y, z = state[0], state[1], state[2]
    yaw     = state[5]
    x_ref, y_ref, z_ref, yaw_ref = target_pos

    # -------------------------------------------------------------------------
    # LQR - Coordinate Transformation & PID
    # -------------------------------------------------------------------------
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez       = z_ref - z
    eyaw     = wrap_angle(yaw_ref - yaw)

    # CRITICAL FIX: Rotate world errors into yaw-body frame
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Filtered derivatives 
    dex = (ex - prev_ex) / dt
    dey = (ey - prev_ey) / dt
    dez = (ez - prev_ez) / dt

    filt_dex = (1.0 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * dex
    filt_dey = (1.0 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * dey
    filt_dez = (1.0 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * dez

    prev_ex = ex; prev_ey = ey; prev_ez = ez

    # Integrals
    int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)

    # Base LQR Command in Body Frame
    vx_cmd = Kp_xy * ex + Ki_xy * int_ex + Kd_xy * filt_dex
    vy_cmd = Kp_xy * ey + Ki_xy * int_ey + Kd_xy * filt_dey
    vz_cmd = Kp_z  * ez + Ki_z  * int_ez + Kd_z  * filt_dez
    yaw_rate_cmd = Kyaw * eyaw

    lqr_cmd = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=float)
    final_cmd = lqr_cmd.copy()
    d_hat_body = np.zeros(4)

    # -------------------------------------------------------------------------
    # Disturbance Observer (DOBC)
    # -------------------------------------------------------------------------
    if DOBC_ENABLED:
        current_pos_4 = np.array([x, y, z, yaw], dtype=float)

        if dobc_prev_pos is None:
            dobc_prev_pos = current_pos_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()
        else:
            # Step 1: Rotate previous body command to global frame for prediction
            prev_yaw = dobc_prev_pos[3]
            c_prev, s_prev = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c_prev * dobc_prev_cmd_body[0] - s_prev * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s_prev * dobc_prev_cmd_body[0] + c_prev * dobc_prev_cmd_body[1]

            # Step 2: Predict global position
            predicted_pos = dobc_prev_pos + (prev_cmd_global + dobc_d_hat_global) * dt

            # Step 3: Innovation (Global frame error)
            innovation = current_pos_4 - predicted_pos
            innovation[3] = wrap_angle(innovation[3])

            # Step 4: Update disturbance estimate
            L = np.array([DOBC_L_xy, DOBC_L_xy, DOBC_L_z, DOBC_L_yaw], dtype=float)
            dobc_d_hat_global = dobc_d_hat_global + L * innovation

            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3],  -0.1, 0.1)

            # Step 5: Rotate global disturbance to body frame
            c_now, s_now = np.cos(yaw), np.sin(yaw)
            d_hat_body = dobc_d_hat_global.copy()
            d_hat_body[0] =  c_now * dobc_d_hat_global[0] + s_now * dobc_d_hat_global[1]
            d_hat_body[1] = -s_now * dobc_d_hat_global[0] + c_now * dobc_d_hat_global[1]

            # Step 6: Apply compensation
            final_cmd = lqr_cmd - d_hat_body

            # Store for next tick
            dobc_prev_pos = current_pos_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()
    else:
        dobc_d_hat_global[:] = 0.0

    # -------------------------------------------------------------------------
    # Comprehensive Data Logging for Presentation Graphs
    # -------------------------------------------------------------------------
    csv_filename = 'flight_data_log.csv'
    
    # Write headers if it's the first time running
    if not log_header_written and not os.path.exists(csv_filename):
        with open(csv_filename, 'w') as f:
            f.write("timestamp,x,y,z,yaw,target_x,target_y,target_z,target_yaw,vx_cmd,vy_cmd,vz_cmd,yaw_rate_cmd,d_hat_x,d_hat_y,d_hat_z\n")
        log_header_written = True

    # Log the full suite of data
    with open(csv_filename, 'a') as f:
        log_data = np.hstack((
            timestamp, 
            state[0:3], yaw, 
            target_pos, 
            final_cmd, 
            dobc_d_hat_global[0:3]
        ))
        np.savetxt(f, [log_data], delimiter=',', fmt="%.6f")

    # -------------------------------------------------------------------------
    # Clamp and Return
    # -------------------------------------------------------------------------
    vx_out       = clamp(float(final_cmd[0]), -MAX_CMD, MAX_CMD)
    vy_out       = clamp(float(final_cmd[1]), -MAX_CMD, MAX_CMD)
    vz_out       = clamp(float(final_cmd[2]), -MAX_CMD, MAX_CMD)
    yaw_rate_out = clamp(float(final_cmd[3]), -MAX_CMD, MAX_CMD)

    return vx_out, vy_out, vz_out, yaw_rate_out
