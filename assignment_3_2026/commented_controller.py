# wind_flag = True

"""
Controller: Linear Quadratic Regulator (LQR) with Disturbance Observer-Based Control (DOBC)
Auto-tuned using Particle Swarm Optimisation (PSO)

Methods implemented:
  1. Linear Quadratic Regulator (LQR)
     Proportional gains are derived analytically from the discrete single-state
     Riccati solution: K = sqrt(Q/R), where Q penalises position error and R
     penalises control effort. Higher Q/R → more aggressive correction.
     The z-axis uses a higher Q/R ratio than xy because altitude stability is
     more critical on a quadrotor - losing altitude is dangerous, lateral drift
     is recoverable.

  2. Disturbance Observer-Based Control (DOBC)
     Estimates external disturbances (e.g. wind) acting on the drone each timestep
     by comparing the kinematically predicted position against the actual measured
     position. The difference (innovation) is filtered through an observer gain L
     to update the disturbance estimate:
         d_hat[k] = d_hat[k-1] + L * (p_actual[k] - p_predicted[k])
     The estimate is rotated from global frame into the yaw-body frame and
     subtracted from the LQR command before it reaches the drone, pre-cancelling
     the disturbance effect.

  3. PSO Auto-Tuning
     All gains were tuned using Particle Swarm Optimisation. Each particle
     represents a candidate parameter set and is scored against a fitness function
     designed to mirror the marking criteria (pos_mean, pos_std, yaw_mean, yaw_std).
     Two separate PSO runs were conducted:
       - Simulation gains: wider bounds, higher Kp/Kd, optimised for simulator
       - Real drone gains (commented out): tighter bounds informed by lab results;
         Kd capped at 0.15 to prevent derivative amplifying hardware noise and
         WiFi latency (~100ms round trip) into oscillation.

Coordinate frames:
  - State positions are received in the GLOBAL (world) frame from Vicon/simulator
  - Wind disturbances act in the GLOBAL frame
  - Velocity commands are sent in the YAW-BODY frame (yaw-rotated from global,
    no pitch/roll - consistent with run.py's inverted_quat_yaw convention)
  - Horizontal errors are rotated from global into yaw-body frame before control
  - DOBC estimates disturbance in global frame, then rotates into yaw-body frame
    before compensation
"""

import numpy as np

# =============================================================================
# SIMULATION GAINS - active, used for marking
# =============================================================================
# LQR cost matrices: Q penalises position error, R penalises control effort.
# Gains are derived from the single-state Riccati solution: Kp = sqrt(Q/R)
# Z-axis has higher Q/R ratio than xy: altitude error is more critical on a
# quadrotor and warrants more aggressive correction.
# These gains were found by PSO (bounds: Kp_xy ≤ 1.0, Kp_z ≤ 1.5) and then
# back-calculated to their equivalent Q/R matrices for a proper LQR formulation.

Q_xy = 10.32   # xy position error cost
R_xy = 3.0     # xy control effort cost - moderate, balances speed vs smoothness
Q_z  = 15.0    # z position error cost - higher than xy, altitude is critical
R_z  = 2.4     # z control effort cost - lower than xy, willing to use more thrust

Kp_xy = np.sqrt(Q_xy / R_xy)   # = 1.8547 - proportional gain for x/y position
Kp_z  = np.sqrt(Q_z  / R_z)    # = 2.5    - proportional gain for altitude

Ki_xy = 0.02     # integral gain for x/y - removes steady-state offset under bias
Ki_z  = 0.001    # integral gain for z   - small, avoids windup on altitude
Kd_xy = 1.0      # derivative gain for x/y - damps overshoot during transit
Kd_z  = 0.0675   # derivative gain for z   - light damping on altitude
Kyaw  = 0.4779   # proportional yaw rate gain

DERIV_ALPHA = 0.8  # derivative low-pass filter coefficient (0=heavy filter, 1=none)
                   # 0.8 chosen for simulation: low noise allows less filtering

INT_LIM_XY = 0.1  # anti-windup clamp for x/y integral (m·s)
INT_LIM_Z  = 0.1  # anti-windup clamp for z integral (m·s)

# DOBC observer gains - tune after LQR is stable in still air
# Higher L → faster disturbance tracking but more noise sensitivity
DOBC_L_xy  = 0.0062  # observer gain for horizontal wind (x, y)
DOBC_L_z   = 0.0047  # observer gain for vertical wind (z)
DOBC_L_yaw = 0.0000  # observer gain for yaw disturbance (disabled)

DOBC_ENABLED = True  # enable disturbance observer (set wind_flag = True above)

# =============================================================================
# REAL DRONE GAINS - commented out, used in lab experiments
# =============================================================================
# These are more conservative gains derived from a second PSO run with tighter
# bounds (Kp_xy ≤ 0.8, Kp_z ≤ 1.2, Kd_xy ≤ 0.15).
# Motivation: the simulation gains caused a limit cycle at MAX_SPEED=60 in Lab 1.
# Root cause: Kd=1.0 amplified Vicon sensor noise and ~100ms WiFi latency into
# sustained oscillation. Tighter Kd bounds prevented this in Lab 2.
# Lower Kp also produces smaller raw commands before the lab's MAX_SPEED scaling
# is applied - genuinely less aggressive at the actuator level.

# Q_xy = 3.2    # conservative xy error cost
# R_xy = 5.0    # high control effort cost - prioritises smooth commands
# Q_z  = 7.2    # conservative z error cost
# R_z  = 5.0    # same R as xy - equally cautious on altitude

# Kp_xy = np.sqrt(Q_xy / R_xy)   # = 0.8
# Kp_z  = np.sqrt(Q_z  / R_z)    # = 1.2

# Kyaw       = 1.0000
# Kd_xy      = 0.1500   # capped at 0.15 to prevent noise amplification on hardware
# Kd_z       = 0.2346
# INT_LIM_XY = 0.1522
# INT_LIM_Z  = 0.8000
# DERIV_ALPHA = 0.5000  # more filtering than simulation (0.8) - hardware is noisier
# Ki_xy      = 0.0326
# Ki_z       = 0.0447

# DOBC_L_xy  = 0.0127
# DOBC_L_z   = 0.0102
# DOBC_L_yaw = 0.0054

# =============================================================================
# LQR persistent state
# =============================================================================
prev_ex  = 0.0;  prev_ey  = 0.0;  prev_ez  = 0.0
int_ex   = 0.0;  int_ey   = 0.0;  int_ez   = 0.0
filt_dex = 0.0;  filt_dey = 0.0;  filt_dez = 0.0

# =============================================================================
# DOBC persistent state
# =============================================================================
dobc_d_hat_global  = np.zeros(4)  # disturbance estimate [dx, dy, dz, d_yaw] - global frame
dobc_prev_state    = None          # previous [x, y, z, yaw] - for kinematic prediction
dobc_prev_cmd_body = np.zeros(4)  # previous final command sent - yaw-body frame

# Debug variables - readable externally for logging and analysis
debug_last_lqr_cmd    = np.zeros(4)
debug_last_final_cmd  = np.zeros(4)
debug_last_d_hat_body = np.zeros(4)
debug_last_innovation = np.zeros(4)


# =============================================================================
# Helper functions
# =============================================================================

def clamp(value, low, high):
    """Clamp value to [low, high] to keep commands within actuator limits."""
    return max(low, min(high, value))


def wrap_angle(angle):
    """Wrap angle to [-pi, pi] to avoid discontinuity in yaw error."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation_world_to_yaw_frame(yaw):
    """
    Rotate a 2D vector from global frame into yaw-aligned body frame.
    Only yaw rotation is applied - no pitch or roll - consistent with
    run.py's lin_vel computation using inverted_quat_yaw.

    R = | cos(yaw)   sin(yaw) |
        | -sin(yaw)  cos(yaw) |
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[ c, s],
                     [-s, c]])


# =============================================================================
# Controller
# =============================================================================

def controller(state, target_pos, dt, wind_enabled=False):
    """
    LQR position controller with DOBC wind disturbance compensation.

    Architecture:
        1. LQR computes a nominal velocity command from position error (PID form)
        2. DOBC estimates wind disturbance from position prediction vs measurement
        3. Disturbance estimate is subtracted from LQR command before output

    Args:
        state        : [x, y, z, roll, pitch, yaw] (m, rad) - global frame
        target_pos   : (x, y, z, yaw)              (m, rad) - global frame
        dt           : controller timestep          (s, nominally 0.02 at 50 Hz)
        wind_enabled : if True, activate DOBC wind compensation

    Returns:
        (vx, vy, vz, yaw_rate) - yaw-body frame, m/s and rad/s
    """

    global prev_ex, prev_ey, prev_ez
    global int_ex, int_ey, int_ez
    global filt_dex, filt_dey, filt_dez
    global dobc_d_hat_global, dobc_prev_state, dobc_prev_cmd_body
    global debug_last_lqr_cmd, debug_last_final_cmd
    global debug_last_d_hat_body, debug_last_innovation

    # -------------------------------------------------------------------------
    # Unpack state and target
    # -------------------------------------------------------------------------
    x, y, z = state[0], state[1], state[2]
    yaw      = state[5]
    x_ref, y_ref, z_ref, yaw_ref = target_pos

    # -------------------------------------------------------------------------
    # LQR - PID velocity command in yaw-body frame
    # -------------------------------------------------------------------------

    # Compute position errors in global frame
    ex_world = x_ref - x
    ey_world = y_ref - y
    ez       = z_ref - z
    eyaw     = wrap_angle(yaw_ref - yaw)

    # Rotate horizontal error into yaw-body frame so commands align with drone axes
    ex, ey = rotation_world_to_yaw_frame(yaw) @ np.array([ex_world, ey_world])

    # Derivative estimate via backward difference with low-pass filter.
    # Filter reduces noise amplification - DERIV_ALPHA trades off responsiveness
    # against noise: simulation uses 0.8 (less filtering), hardware uses 0.5 (more).
    filt_dex = (1.0 - DERIV_ALPHA) * filt_dex + DERIV_ALPHA * ((ex - prev_ex) / dt)
    filt_dey = (1.0 - DERIV_ALPHA) * filt_dey + DERIV_ALPHA * ((ey - prev_ey) / dt)
    filt_dez = (1.0 - DERIV_ALPHA) * filt_dez + DERIV_ALPHA * ((ez - prev_ez) / dt)

    # Integral with anti-windup clamping - removes steady-state bias (e.g. drag offset)
    int_ex = clamp(int_ex + ex * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ey = clamp(int_ey + ey * dt, -INT_LIM_XY, INT_LIM_XY)
    int_ez = clamp(int_ez + ez * dt, -INT_LIM_Z,  INT_LIM_Z)

    # LQR-PID velocity command: v = Kp*e + Ki*∫e + Kd*ė
    # Kp derived from Riccati solution (K = sqrt(Q/R)), Ki and Kd PSO-tuned
    vx_cmd       = Kp_xy * ex + Ki_xy * int_ex + Kd_xy * filt_dex
    vy_cmd       = Kp_xy * ey + Ki_xy * int_ey + Kd_xy * filt_dey
    vz_cmd       = Kp_z  * ez + Ki_z  * int_ez + Kd_z  * filt_dez
    yaw_rate_cmd = Kyaw * eyaw

    lqr_cmd = np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd], dtype=float)
    debug_last_lqr_cmd = lqr_cmd.copy()

    # Store previous errors for next derivative estimate
    prev_ex = ex;  prev_ey = ey;  prev_ez = ez

    # -------------------------------------------------------------------------
    # DOBC - disturbance estimation and compensation
    # -------------------------------------------------------------------------
    # When wind is disabled, the observer is reset so stale estimates from a
    # previous wind-on period do not corrupt the next run.

    if not wind_enabled:
        dobc_d_hat_global  = np.zeros(4)
        dobc_prev_state    = None
        dobc_prev_cmd_body = np.zeros(4)
        debug_last_d_hat_body = np.zeros(4)
        debug_last_innovation = np.zeros(4)
        final_cmd = lqr_cmd

    else:
        current_state_4 = np.array([x, y, z, yaw], dtype=float)

        if dobc_prev_state is None:
            # First wind-enabled tick: initialise memory, no compensation yet
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = lqr_cmd.copy()
            debug_last_d_hat_body = np.zeros(4)
            debug_last_innovation = np.zeros(4)
            final_cmd = lqr_cmd

        else:
            # Step 1: rotate previous yaw-body command into global frame
            # Prediction must live in global frame to match GPS/Vicon measurement.
            # Yaw-only rotation: global = R_yaw * body
            prev_yaw = dobc_prev_state[3]
            c_p, s_p = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global    = dobc_prev_cmd_body.copy()
            prev_cmd_global[0] = c_p * dobc_prev_cmd_body[0] - s_p * dobc_prev_cmd_body[1]
            prev_cmd_global[1] = s_p * dobc_prev_cmd_body[0] + c_p * dobc_prev_cmd_body[1]
            # z and yaw_rate are frame-invariant under yaw-only rotation

            # Step 2: kinematic prediction of current position
            # p_pred[k] = p[k-1] + (v_cmd_global[k-1] + d_hat[k-1]) * dt
            predicted_state = dobc_prev_state + (prev_cmd_global + dobc_d_hat_global) * dt

            # Step 3: innovation = actual position - predicted position
            # A non-zero innovation implies an unmodelled force (e.g. wind)
            innovation    = current_state_4 - predicted_state
            innovation[3] = wrap_angle(innovation[3])  # wrap yaw to [-pi, pi]
            debug_last_innovation = innovation.copy()

            # Step 4: update disturbance estimate (global frame)
            # Observer update law: d_hat[k] = d_hat[k-1] + L * innovation
            # L controls convergence speed vs noise sensitivity
            DOBC_L = np.array([DOBC_L_xy, DOBC_L_xy, DOBC_L_z, DOBC_L_yaw], dtype=float)
            dobc_d_hat_global = dobc_d_hat_global + DOBC_L * innovation

            # Clamp to physically plausible wind effect on this drone
            # (max wind: ~0.02 N on 0.088 kg → small velocity disturbance)
            dobc_d_hat_global[:3] = np.clip(dobc_d_hat_global[:3], -0.2, 0.2)
            dobc_d_hat_global[3]  = np.clip(dobc_d_hat_global[3],  -0.1, 0.1)

            # Step 5: rotate disturbance estimate into yaw-body frame
            # Compensation must match the frame of the LQR command
            # Inverse yaw rotation: body = R_yaw^T * global
            c_n, s_n = np.cos(yaw), np.sin(yaw)
            d_hat_body    = dobc_d_hat_global.copy()
            d_hat_body[0] =  c_n * dobc_d_hat_global[0] + s_n * dobc_d_hat_global[1]
            d_hat_body[1] = -s_n * dobc_d_hat_global[0] + c_n * dobc_d_hat_global[1]
            debug_last_d_hat_body = d_hat_body.copy()

            # Step 6: subtract disturbance from LQR command
            # This pre-cancels wind before it builds into positional error
            final_cmd = lqr_cmd - d_hat_body

            # Store final (compensated) command - not lqr_cmd - because the
            # prediction next tick must match what was actually sent to the drone
            dobc_prev_state    = current_state_4.copy()
            dobc_prev_cmd_body = final_cmd.copy()

    debug_last_final_cmd = final_cmd.copy()

    # Clamp outputs to simulator limits before returning
    return (
        clamp(float(final_cmd[0]), -1.0,     1.0),
        clamp(float(final_cmd[1]), -1.0,     1.0),
        clamp(float(final_cmd[2]), -1.0,     1.0),
        clamp(float(final_cmd[3]), -1.74533, 1.74533),
    )
