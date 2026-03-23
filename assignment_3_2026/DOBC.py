# wind_flag = True

import numpy as np


def controller(state, target_pos, dt, wind_enabled=False):
    """
    UAV position controller with a disturbance observer (DOBC) to compensate wind.

    Args:
        state       : [pos_x, pos_y, pos_z, roll, pitch, yaw]
        target_pos  : (x, y, z, yaw)
        dt          : controller timestep (s)
        wind_enabled: True when wind is active in the simulator

    Returns:
        (vx, vy, vz, yaw_rate) in the yaw-body frame
    """

    # Initialise persistent variables on first call
    if not hasattr(controller, 'initialised'):
        # Estimated disturbance in global frame [dx, dy, dz, d_yaw]
        controller.d_hat_global  = np.zeros(4)
        # Previous [x, y, z, yaw] for prediction
        controller.prev_state    = None
        # Previous command in yaw-body frame
        controller.prev_cmd_body = np.zeros(4)
        controller.initialised   = True

    # DOBC gains for [x, y, z, yaw]
    # Bigger L → faster tracking, but more noise sensitive
    L = np.array([0.4, 0.4, 0.3, 0.2])

    # Current state in global frame
    pos = np.array(state[0:3])
    yaw = state[5]
    current_state_4 = np.array([pos[0], pos[1], pos[2], yaw])

    # 1 - nominal controller (e.g. LQR) in yaw-body frame
    # Replace this placeholder with your actual LQR call:
    #   lqr_cmd = lqr_compute(state, target_pos)
    lqr_cmd = np.array([0.0, 0.0, 0.0, 0.0])

    # 2 - DOBC wind compensation (only when wind is enabled)
    if not wind_enabled:
        # No wind: reset observer and just use nominal command
        controller.d_hat_global  = np.zeros(4)
        controller.prev_state    = None
        controller.prev_cmd_body = np.zeros(4)
        final_cmd = lqr_cmd

    else:
        # First tick with wind: just store state and command
        if controller.prev_state is None:
            controller.prev_state    = current_state_4.copy()
            controller.prev_cmd_body = lqr_cmd.copy()
            final_cmd = lqr_cmd

        else:
            # Rotate previous body-frame command into global frame
            prev_yaw = controller.prev_state[3]
            c, s = np.cos(prev_yaw), np.sin(prev_yaw)
            prev_cmd_global = controller.prev_cmd_body.copy()
            prev_cmd_global[0] = c * controller.prev_cmd_body[0] - s * controller.prev_cmd_body[1]
            prev_cmd_global[1] = s * controller.prev_cmd_body[0] + c * controller.prev_cmd_body[1]
            # z and yaw_rate unaffected by yaw rotation

            # Predict current state with simple kinematics:
            # x_pred = x_prev + (u_prev_global + d_hat_prev) * dt
            predicted_state = (controller.prev_state
                               + (prev_cmd_global + controller.d_hat_global) * dt)

            # Innovation: measured minus predicted
            innovation = current_state_4 - predicted_state

            # Wrap yaw error to [-pi, pi]
            innovation[3] = (innovation[3] + np.pi) % (2 * np.pi) - np.pi

            # Disturbance observer update: d_hat_k = d_hat_{k-1} + L * innovation
            controller.d_hat_global = controller.d_hat_global + L * innovation

            # Limit disturbance estimate to reasonable values
            controller.d_hat_global[:3] = np.clip(controller.d_hat_global[:3], -2.0, 2.0)
            controller.d_hat_global[3]  = np.clip(controller.d_hat_global[3],  -1.0, 1.0)

            # Rotate disturbance estimate back to yaw-body frame
            c, s = np.cos(yaw), np.sin(yaw)
            d_hat_body = controller.d_hat_global.copy()
            d_hat_body[0] =  c * controller.d_hat_global[0] + s * controller.d_hat_global[1]
            d_hat_body[1] = -s * controller.d_hat_global[0] + c * controller.d_hat_global[1]

            # Compensate nominal command with estimated disturbance
            final_cmd = lqr_cmd - d_hat_body

            # Store for next step
            controller.prev_state    = current_state_4.copy()
            controller.prev_cmd_body = final_cmd.copy()

    # 3 - return final command
    output = (
        float(final_cmd[0]),
        float(final_cmd[1]),
        float(final_cmd[2]),
        float(final_cmd[3]),
    )
    return output
