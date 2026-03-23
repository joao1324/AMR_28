"""
Simple guide for AMR28_pso_tuner.py

What this script does
- This script tunes the parameters in othercontroller.py with PSO.
- It keeps the same overall tuning logic as pso_tuner.py, but is adapted for
  the current othercontroller implementation.

How to run
- Quick smoke test:
  ./AMR_assignment_3_env/bin/python AMR28_pso_tuner.py --particles 2 --iters 1 --trials 1
- Normal tuning:
  ./AMR_assignment_3_env/bin/python AMR28_pso_tuner.py --particles 4 --iters 2 --trials 2
- Stronger search:
  ./AMR_assignment_3_env/bin/python AMR28_pso_tuner.py --particles 6 --iters 4 --trials 3
- With wind:
  ./AMR_assignment_3_env/bin/python AMR28_pso_tuner.py --particles 6 --iters 4 --trials 3 --wind
- Wind-only local refinement:
  ./AMR_assignment_3_env/bin/python AMR28_pso_tuner.py --particles 6 --iters 4 --trials 3 --wind-only

Arguments
- --particles: number of particles in PSO. More particles means broader search,
  but slower runtime.
- --iters: number of PSO iterations. More iterations means deeper search, but
  slower runtime.
- --trials: number of target trials used in each evaluation. Larger values make
  results more reliable, but slower.
- --wind: evaluate both no-wind and wind cases.
- --wind-only: only tune for wind performance near the current baseline.

How to read the output
- Baseline score: score of the current parameter set in othercontroller.py.
- Best score: best score found by PSO. Lower is better.
- Best params: the tuned parameter vector. For the current module-style
  othercontroller.py, the order is:
  [Kp_xy, Kp_z, Kyaw, Kd_xy, Kd_z, INT_LIM_XY, INT_LIM_Z, DERIV_ALPHA]

How to use the tuned result
- Copy the values from Best params back into othercontroller.py.
- Example:
  Best params:
  [0.5656 0.9760 0.9066 0.2070 0.2863 0.4972 1.1081 0.3700]
  means:
  Kp_xy = 0.5656
  Kp_z = 0.9760
  Kyaw = 0.9066
  Kd_xy = 0.2070
  Kd_z = 0.2863
  INT_LIM_XY = 0.4972
  INT_LIM_Z = 1.1081
  DERIV_ALPHA = 0.3700

Practical advice
- Use a small run first to confirm the script works.
- Compare scores only when using the same command settings.
- If runtime is too long, reduce --particles, --iters, or --trials.
- If results are noisy, increase --trials.
"""

import argparse
import math
import random

import numpy as np
import pybullet as p

import othercontroller
from run_headless import Simulator
from src.wind import Wind


PHYSICS_DT = 1.0 / 1000.0
POS_CTRL_DT = 1.0 / 50.0
TOTAL_TIME = 20.0
SETTLE_TIME = 10.0


CLASS_PARAM_SPECS = [
    {"name": "kp_hover", "size": 3, "bounds": [(0.8, 2.2), (0.8, 2.2), (1.0, 2.8)]},
    {"name": "kd_hover", "size": 3, "bounds": [(1.0, 2.5), (1.0, 2.5), (1.3, 3.0)]},
    {"name": "dob_l_pos", "size": 3, "bounds": [(1.2, 3.5), (1.2, 3.5), (1.5, 4.5)]},
    {"name": "dob_l_dist", "size": 3, "bounds": [(0.30, 1.60), (0.30, 1.60), (0.40, 2.00)]},
    {"name": "hover_vel_limit", "size": 3, "bounds": [(0.10, 0.28), (0.10, 0.28), (0.08, 0.22)]},
    {"name": "kp_yaw_hover", "size": 1, "bounds": [(2.0, 5.0)]},
    {"name": "kd_yaw", "size": 1, "bounds": [(0.03, 0.30)]},
    {"name": "ki_yaw", "size": 1, "bounds": [(0.05, 0.35)]},
]

MODULE_PARAM_SPECS = [
    {"name": "Kp_xy", "size": 1, "bounds": [(0.20, 2.50)]},
    {"name": "Kp_z", "size": 1, "bounds": [(0.20, 2.50)]},
    {"name": "Kyaw", "size": 1, "bounds": [(0.20, 2.50)]},
    {"name": "Kd_xy", "size": 1, "bounds": [(0.00, 1.00)]},
    {"name": "Kd_z", "size": 1, "bounds": [(0.00, 1.00)]},
    {"name": "INT_LIM_XY", "size": 1, "bounds": [(0.10, 3.00)]},
    {"name": "INT_LIM_Z", "size": 1, "bounds": [(0.10, 3.00)]},
    {"name": "DERIV_ALPHA", "size": 1, "bounds": [(0.02, 0.80)]},
]


def get_param_specs():
    if hasattr(othercontroller, "AdvancedController"):
        return CLASS_PARAM_SPECS
    return MODULE_PARAM_SPECS


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def sample_target(rng):
    return (
        rng.uniform(-4.0, 4.0),
        rng.uniform(-4.0, 4.0),
        rng.uniform(0.2, 4.5),
        rng.uniform(-math.pi, math.pi),
    )


def get_controller_instance():
    if not hasattr(othercontroller, "AdvancedController"):
        return None
    return othercontroller.AdvancedController()


def get_controller_callable():
    if hasattr(othercontroller, "controller"):
        return othercontroller.controller
    raise AttributeError("othercontroller.py must define controller(state, target_pos, dt, wind_enabled=False)")


def flatten_controller_params(ctrl):
    param_specs = get_param_specs()
    values = []
    for spec in param_specs:
        if ctrl is None:
            if not hasattr(othercontroller, spec["name"]):
                raise AttributeError(
                    f"othercontroller.py missing global parameter '{spec['name']}'"
                )
            value = getattr(othercontroller, spec["name"])
        else:
            if not hasattr(ctrl, spec["name"]):
                raise AttributeError(
                    f"othercontroller.AdvancedController missing attribute '{spec['name']}'"
                )
            value = getattr(ctrl, spec["name"])
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != spec["size"]:
            raise ValueError(
                f"Attribute '{spec['name']}' expected size {spec['size']}, got {arr.size}"
            )
        values.append(arr)
    return np.concatenate(values)


def configure_controller(ctrl, params):
    param_specs = get_param_specs()
    offset = 0
    for spec in param_specs:
        chunk = np.asarray(params[offset : offset + spec["size"]], dtype=float)
        offset += spec["size"]
        value = float(chunk[0]) if spec["size"] == 1 else chunk.copy()
        if ctrl is None:
            setattr(othercontroller, spec["name"], value)
            continue
        if spec["size"] == 1:
            setattr(ctrl, spec["name"], float(chunk[0]))
        else:
            setattr(ctrl, spec["name"], chunk.copy())


def build_bounds():
    return np.array(
        [bound for spec in get_param_specs() for bound in spec["bounds"]],
        dtype=float,
    )


def reset_module_controller_state():
    reset_values = {
        "prev_ex": 0.0,
        "prev_ey": 0.0,
        "prev_ez": 0.0,
        "int_ex": 0.0,
        "int_ey": 0.0,
        "int_ez": 0.0,
        "filt_dex": 0.0,
        "filt_dey": 0.0,
        "filt_dez": 0.0,
    }
    for name, value in reset_values.items():
        if hasattr(othercontroller, name):
            setattr(othercontroller, name, value)


def run_trial(params, target, wind_enabled, seed):
    random.seed(seed)
    np.random.seed(seed)

    othercontroller.ctrl = get_controller_instance()
    configure_controller(othercontroller.ctrl, params)
    if othercontroller.ctrl is None:
        reset_module_controller_state()
    controller_fn = get_controller_callable()

    sim = Simulator()
    sim.wind_enabled = wind_enabled
    sim.targets = [target]
    sim.current_target = 0

    if wind_enabled:
        random.seed(seed)
        sim.wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1)

    steps_between_pos_control = int(round(POS_CTRL_DT / PHYSICS_DT))
    total_steps = int(round(TOTAL_TIME / PHYSICS_DT))
    settle_step = int(round(SETTLE_TIME / PHYSICS_DT))

    loop_counter = 0
    prev_rpm = np.zeros(4)
    desired_vel = np.zeros(3)
    yaw_rate_setpoint = 0.0

    pos_errors = []
    yaw_errors = []
    first_reach_time = None

    for step in range(total_steps):
        pos, quat = p.getBasePositionAndOrientation(sim.drone_id)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(sim.drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)

        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        _, inv_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
        _, inv_quat = p.invertTransform([0, 0, 0], quat)
        lin_vel = np.array(p.rotateVector(inv_quat_yaw, lin_vel_world))
        ang_vel = np.array(p.rotateVector(inv_quat, ang_vel_world))

        loop_counter += 1
        if loop_counter >= steps_between_pos_control:
            loop_counter = 0
            state = np.concatenate((pos, [roll, pitch, yaw]))
            out = sim.check_action(
                controller_fn(state, target, POS_CTRL_DT, sim.wind_enabled)
            )
            desired_vel = np.array(out[:3], dtype=float)
            yaw_rate_setpoint = float(out[3])

        pos_error = np.linalg.norm(np.array(pos) - np.array(target[:3]))
        yaw_error = abs(wrap_to_pi(yaw - target[3]))

        if first_reach_time is None and pos_error < 0.05 and yaw_error < 0.05:
            first_reach_time = step * PHYSICS_DT

        if step >= settle_step:
            pos_errors.append(pos_error)
            yaw_errors.append(yaw_error)

        rpm = sim.tello_controller.compute_control(
            desired_vel, lin_vel, quat, ang_vel, yaw_rate_setpoint, PHYSICS_DT
        )
        rpm = sim.motor_model(rpm, prev_rpm, PHYSICS_DT)
        prev_rpm = rpm
        force, torque = sim.compute_dynamics(rpm, lin_vel_world, quat)

        p.applyExternalForce(sim.drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(sim.drone_id, -1, torque, p.LINK_FRAME)

        if sim.wind_enabled:
            wind = sim.wind_sim.get_wind(PHYSICS_DT)
            p.applyExternalForce(sim.drone_id, -1, wind, pos, p.WORLD_FRAME)

        sim.spin_motors(rpm, PHYSICS_DT)
        p.stepSimulation()

    p.disconnect()

    return {
        "reach_time": 999.0 if first_reach_time is None else float(first_reach_time),
        "pos_mean": float(np.mean(pos_errors)),
        "pos_std": float(np.std(pos_errors)),
        "yaw_mean": float(np.mean(yaw_errors)),
        "yaw_std": float(np.std(yaw_errors)),
    }


def fitness_from_metrics(metrics):
    return (
        7.0 * metrics["pos_mean"]
        + 5.0 * metrics["pos_std"]
        + 0.9 * metrics["yaw_mean"]
        + 0.5 * metrics["yaw_std"]
        + 0.08 * max(metrics["reach_time"] - 5.0, 0.0)
    )


def evaluate_params(params, trials, seed, include_wind):
    rng = random.Random(seed)
    targets = [sample_target(rng) for _ in range(trials)]

    scores = []
    for idx, target in enumerate(targets):
        metrics = run_trial(params, target, wind_enabled=False, seed=seed + idx)
        scores.append(fitness_from_metrics(metrics))

        if include_wind:
            wind_metrics = run_trial(
                params, target, wind_enabled=True, seed=10_000 + seed + idx
            )
            scores.append(1.5 * fitness_from_metrics(wind_metrics))

    return float(np.mean(scores))


def evaluate_params_wind_only(params, trials, seed):
    rng = random.Random(seed)
    targets = [sample_target(rng) for _ in range(trials)]

    scores = []
    for idx, target in enumerate(targets):
        wind_metrics = run_trial(
            params, target, wind_enabled=True, seed=20_000 + seed + idx
        )
        scores.append(
            9.0 * wind_metrics["pos_mean"]
            + 7.0 * wind_metrics["pos_std"]
            + 0.08 * max(wind_metrics["reach_time"] - 5.0, 0.0)
            + 0.6 * wind_metrics["yaw_mean"]
            + 0.3 * wind_metrics["yaw_std"]
        )

    return float(np.mean(scores))


class PSO:
    def __init__(self, num_particles, iters, trials, seed, include_wind):
        self.num_particles = num_particles
        self.iters = iters
        self.trials = trials
        self.seed = seed
        self.include_wind = include_wind

        self.bounds = build_bounds()
        self.dim = len(self.bounds)

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        span = high - low

        rng = np.random.default_rng(seed)
        self.positions = rng.uniform(low, high, size=(self.num_particles, self.dim))
        self.velocities = rng.uniform(
            -0.20 * span, 0.20 * span, size=(self.num_particles, self.dim)
        )

        self.pbest_pos = self.positions.copy()
        self.pbest_scores = np.full(self.num_particles, np.inf)
        self.gbest_pos = None
        self.gbest_score = np.inf

    def run(self):
        print("Starting PSO tuning")
        for iteration in range(self.iters):
            for particle in range(self.num_particles):
                params = self.positions[particle]
                try:
                    score = evaluate_params(
                        params,
                        trials=self.trials,
                        seed=self.seed + 100 * iteration + particle,
                        include_wind=self.include_wind,
                    )
                except Exception as exc:
                    print(f"particle={particle:02d} failed: {exc}")
                    score = 999.0

                if score < self.pbest_scores[particle]:
                    self.pbest_scores[particle] = score
                    self.pbest_pos[particle] = params.copy()

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos = params.copy()
                    print(
                        f"iter={iteration:02d} particle={particle:02d} "
                        f"new_best={score:.6f}"
                    )

            w = 0.55
            c1 = 1.4
            c2 = 1.6
            r1 = np.random.rand(self.num_particles, self.dim)
            r2 = np.random.rand(self.num_particles, self.dim)

            self.velocities = (
                w * self.velocities
                + c1 * r1 * (self.pbest_pos - self.positions)
                + c2 * r2 * (self.gbest_pos - self.positions)
            )
            self.positions += self.velocities
            self.positions = np.clip(
                self.positions, self.bounds[:, 0], self.bounds[:, 1]
            )

        print("\nBest score:", self.gbest_score)
        print("Best params:")
        np.set_printoptions(precision=4, suppress=True)
        print(self.gbest_pos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-only", action="store_true")
    args = parser.parse_args()

    baseline_ctrl = get_controller_instance()
    baseline = flatten_controller_params(baseline_ctrl)

    if args.wind_only:
        baseline_score = evaluate_params_wind_only(
            baseline, trials=args.trials, seed=args.seed
        )
    else:
        baseline_score = evaluate_params(
            baseline, trials=args.trials, seed=args.seed, include_wind=args.wind
        )
    print(f"Baseline score: {baseline_score:.6f}")

    tuner = PSO(
        num_particles=args.particles,
        iters=args.iters,
        trials=args.trials,
        seed=args.seed,
        include_wind=args.wind,
    )
    tuner.positions[0] = baseline.copy()

    if args.wind_only:
        print("Starting wind-only local refinement")
        tuner.bounds = np.array(
            [(value * 0.9, value * 1.1) for value in baseline],
            dtype=float,
        )
        tuner.dim = len(tuner.bounds)
        low = tuner.bounds[:, 0]
        high = tuner.bounds[:, 1]
        span = high - low
        rng = np.random.default_rng(args.seed)
        tuner.positions = rng.uniform(low, high, size=(tuner.num_particles, tuner.dim))
        tuner.positions[0] = baseline.copy()
        tuner.velocities = rng.uniform(
            -0.15 * span, 0.15 * span, size=(tuner.num_particles, tuner.dim)
        )
        tuner.pbest_pos = tuner.positions.copy()
        tuner.pbest_scores = np.full(tuner.num_particles, np.inf)
        tuner.gbest_pos = None
        tuner.gbest_score = np.inf

        print("Starting PSO tuning")
        for iteration in range(tuner.iters):
            for particle in range(tuner.num_particles):
                params = tuner.positions[particle]
                try:
                    score = evaluate_params_wind_only(
                        params,
                        trials=tuner.trials,
                        seed=tuner.seed + 100 * iteration + particle,
                    )
                except Exception as exc:
                    print(f"particle={particle:02d} failed: {exc}")
                    score = 999.0

                if score < tuner.pbest_scores[particle]:
                    tuner.pbest_scores[particle] = score
                    tuner.pbest_pos[particle] = params.copy()

                if score < tuner.gbest_score:
                    tuner.gbest_score = score
                    tuner.gbest_pos = params.copy()
                    print(
                        f"iter={iteration:02d} particle={particle:02d} "
                        f"new_best={score:.6f}"
                    )

            w = 0.55
            c1 = 1.4
            c2 = 1.6
            r1 = np.random.rand(tuner.num_particles, tuner.dim)
            r2 = np.random.rand(tuner.num_particles, tuner.dim)

            tuner.velocities = (
                w * tuner.velocities
                + c1 * r1 * (tuner.pbest_pos - tuner.positions)
                + c2 * r2 * (tuner.gbest_pos - tuner.positions)
            )
            tuner.positions += tuner.velocities
            tuner.positions = np.clip(
                tuner.positions, tuner.bounds[:, 0], tuner.bounds[:, 1]
            )

        print("\nBest score:", tuner.gbest_score)
        print("Best params:")
        np.set_printoptions(precision=4, suppress=True)
        print(tuner.gbest_pos)
    else:
        tuner.run()


if __name__ == "__main__":
    main()
