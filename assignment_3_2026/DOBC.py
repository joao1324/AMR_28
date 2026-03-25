import numpy as np

# LQR gains
INT_LIM_XY  = 1.5
INT_LIM_Z   = 1.0

# Derivative filter (0–1). Higher = faster but noisier.
DERIV_ALPHA = 0.3

Q_xy = 1.0
R_xy = 2.8
Q_z  = 1.5
R_z  = 2.3

# From scalar LQR: K = sqrt(Q/R)
Kp_xy = np.sqrt(Q_xy / R_xy)
Kp_z  = np.sqrt(Q_z  / R_z)

# Small I for steady‑state error, D for damping
Ki_xy = 0.05
Ki_z  = 0.08

Kd_xy = 0.08
Kd_z  = 0.12

Kyaw  = 0.8


# LQR state
prev_ex  = 0.0
prev_ey  = 0.0
prev_ez  = 0.0
int_ex   = 0.0
int_ey   = 0.0
int_ez   = 0.0
filt_dex = 0.0
filt_dey = 0.0
filt_dez = 0.0


# DOBC state
# Disturbance estimate [dx, dy, dz, d_yaw] in world frame
dobc_d_hat_global  = np.zeros(4)
dobc_prev_state    = None          # [x, y, z, yaw] at previous step
dobc_prev_cmd_body = np.zeros(4)   # previous body‑frame command

# DOBC gains [x, y, z, yaw]. Small on purpose - gentle observer, for now
DOBC_L = np.array([0.01, 0.01, 0.008, 0.0])    # yaw DOB off for now


# Helpers functiions
def clamp(value, low, high):
    return max(low, min(high, value))

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def rotation_world_to_yaw_frame(yaw):
    # 2D rotation using yaw only
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])


# Main controller
def controller(state, target_pos, dt, wind_enabled=False):
    """
    Outer‑loop position controller + simple disturbance observer.

    state:      [x, y, z, roll, pitch, yaw]
    target_pos: (x, y, z, yaw)
    dt:         controller timestep (s)
    """

    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_state, dobc_prev_cmd_body

    #unack state/target
    x, y, z = state[0], state[1], state[2]
    yaw     = state[5]

    x_ref, y_ref, z_ref, yaw_ref = target_pos

    
    # LQR velocity
    # Position error in world frame
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez = z_ref - z
    eyaw = wrap_angle(yaw_ref - yaw)

    # Rotate xy error into yaw‑aligned body frame
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Filtered derivatives
    filt_dex = (1 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * ((ex - prev_ex) / dt)
    filt_dey = (1 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * ((ey - prev_ey) / dt)
    filt_dez = (1 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * ((ez - prev_ez) / dt)

    # Integrals with clamping
    int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)

    # PID‑style velocity in body frame
    vx_cmd       = Kp_xy * ex + Ki_xy * int_ex + Kd_xy * filt_dex
    vy_cmd       = Kp_xy * ey + Ki_xy * int_ey + Kd_xy * filt_dey
    vz_cmd       = Kp_z  * ez + Ki_z  * int_ez + Kd_z  * filt_dez
    yaw_rate_cmd = Kyaw * eyaw

    lqr_cmd = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd])

    prev_ex, prev_ey, prev_ez = ex, ey, ez

  
    # Disturbance observer
    if not wind_enabled:
        # No wind - keep DOB idle and reset its memory
        dobc_d_hat_global  = np.zeros(4)
        dobc_prev_state    = None
        dobc_prev_cmd_body = np.zeros(4)
        final_cmd = lqr_cmd

    else:
        current_state_4 = np.array([x, y, z, yaw])

        if dobc_prev_state is None:
            # First step with wind on: just initialise
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()
            final_cmd          = lqr_cmd

        else:
            # 1) previous body cmd - world frame
            prev_yaw = dobc_prev_state[3]
            c, s = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c * dobc_prev_cmd_body[0] - s * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s * dobc_prev_cmd_body[0] + c * dobc_prev_cmd_body[1]

            # 2) predict where we should be now
            predicted_state = dobc_prev_state + (prev_cmd_global + dobc_d_hat_global) * dt

            # 3) innovation = actual − predicted
            innovation = current_state_4 - predicted_state
            innovation[3] = wrap_angle(innovation[3])

            # 4) update disturbance estimate (world frame)
            dobc_d_hat_global = dobc_d_hat_global + DOBC_L * innovation

            # 5) limit disturbance magnitude
            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3], -0.1, 0.1)

            # 6) rotate estimate back to body frame
            c, s = np.cos(yaw), np.sin(yaw)
            d_hat_body    = dobc_d_hat_global.copy()
            d_hat_body[0] =  c * dobc_d_hat_global[0] + s * dobc_d_hat_global[1]
            d_hat_body[1] = -s * dobc_d_hat_global[0] + c * dobc_d_hat_global[1]

            # 7) subtract estimated disturbance from nominal command
            final_cmd = lqr_cmd - d_hat_body

            # store for next step
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()

    # Clamp and return
    return (
        clamp(float(final_cmd[0]), -1.0,     1.0),
        clamp(float(final_cmd[1]), -1.0,     1.0),
        clamp(float(final_cmd[2]), -1.0,     1.0),
        clamp(float(final_cmd[3]), -1.74533, 1.74533),
    )
