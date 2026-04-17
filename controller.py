# wind_flag = False
# Implement a controller

import numpy as np

# Global variables defined:
prev_ex = 0.0
prev_ey = 0.0
prev_ez = 0.0
int_ex = 0.0
int_ey = 0.0
int_ez = 0.0
INT_LIM_XY = 1.5
INT_LIM_Z = 1.0
DERIV_ALPHA = 0.2

Q_xy = 1.0
R_xy = 2.8
Q_z = 1.5
R_z = 2.3
# LQR Gains (analytic solution Kp = sqrt(Q/R) for a first-order system):
Kp_xy = np.sqrt(Q_xy/R_xy)
Kp_z = np.sqrt(Q_z/R_z)
Kyaw = 0.8

Ki_xy = 0.08
Ki_z = 0.10
Kd_xy = 0.15
Kd_z = 0.2

filt_dex = 0.0
filt_dey = 0.0
filt_dez = 0.0

# Ensure output stays within safe bounds
def clamp(value, low, high):
        return max(low, min(high, value))

# Keeps angle between -pi and +pi 
def wrap_angle(angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

# Matrix to convert vector from world frame to yaw-aligned drone frame
def rotation_world_to_yaw_frame(yaw):
       c = np.cos(yaw)
       s = np.sin(yaw)

       return np.array([
              [c, s],
              [-s, c]
       ])

def controller(state, target_pos, dt, wind_enabled=False):
    # state format: [position_x (m), position_y (m), position_z (m), roll (radians), pitch (radians), yaw (radians)]
    # target_pos format: (x (m), y (m), z (m), yaw (radians))
    # dt: time step (s)
    # wind_enabled: boolean flag to indicate if wind disturbance should be considered in the control algorithm
    # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (radians/s))

    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez

    # Extract values from state array:
    x = state[0]
    y = state[1]
    z = state[2]
    roll = state[3]
    pitch = state[4]
    yaw = state[5]

    # Unpack target state:
    x_ref = target_pos[0]
    y_ref = target_pos[1]
    z_ref = target_pos[2]
    yaw_ref = target_pos[3]

    # Compute errors, which we try and reduce to zero:
    # World frame:
    ex_world = x_ref - x # How far we are in x
    ey_world = y_ref - y # How far we are in y
    ez = z_ref - z # How far we are in z (height error)
    # Yaw error:
    eyaw = wrap_angle(yaw_ref - yaw) # Angular error which is wrapped
    # Rotate horizontal error into yaw-aligned frame:
    R_w2b = rotation_world_to_yaw_frame(yaw)
    error_world = np.array([ex_world, ey_world])
    error_body = R_w2b @ error_world

    ex = error_body[0]
    ey = error_body[1]

    # Estimate error derivatives using backward difference:
    raw_dex = (ex - prev_ex) / dt
    raw_dey = (ey - prev_ey) / dt
    raw_dez = (ez - prev_ez) / dt

    # Low pass filter the estimates to reduce noise
    filt_dex = (1 - DERIV_ALPHA)*filt_dex + DERIV_ALPHA*raw_dex
    filt_dey = (1 - DERIV_ALPHA)*filt_dey + DERIV_ALPHA*raw_dey
    filt_dez = (1 - DERIV_ALPHA)*filt_dez + DERIV_ALPHA*raw_dez
    dex = filt_dex
    dey = filt_dey
    dez = filt_dez

    # Update error integrals with anti-windup clamping:
    int_ex = clamp(int_ex + ex*dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey*dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez*dt, -INT_LIM_Z, INT_LIM_Z)

    # Form augmented state vectors [error, error_dot, error_integral] for each axis:
    x_state = np.array([ex, dex, int_ex])
    y_state = np.array([ey, dey, int_ey])
    z_state = np.array([ez, dez, int_ez])

    # LQR gain vectors K = [Kp, Kd, Ki], control law is u = K @ state:
    K_xy = np.array([Kp_xy, Kd_xy, Ki_xy])
    K_z  = np.array([Kp_z,  Kd_z,  Ki_z])

    # State-feedback control
    vx_cmd = K_xy @ x_state
    vy_cmd = K_xy @ y_state
    vz_cmd = K_z  @ z_state
    yaw_rate_cmd = Kyaw*eyaw

    # Clamp outputs to allowed range where yaw rate is in radians per second (100 deg/s = 1.74533 rad/s)
    vx_cmd = clamp(vx_cmd, -1.0, 1.0)
    vy_cmd = clamp(vy_cmd, -1.0, 1.0)
    vz_cmd = clamp(vz_cmd, -1.0, 1.0)
    yaw_rate_cmd = clamp(yaw_rate_cmd, -1.74533, 1.74533)

    prev_ex = ex
    prev_ey = ey
    prev_ez = ez

    output = (vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)    

    return output
