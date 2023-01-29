import os
import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.stflow import LogFileWriter

from coordination_controller.agent import Agent
from coordination_controller.arm_coord_env import CoordinationEnv

from coordination_controller.plotting import plot_box

# Set Sacred Experiment
ex = Experiment(
    name="Upper Level Training",
    interactive=False,  # Turn this on to run on interactive environment (e.g. jupyter nb)
    save_git_info=False,
)
ex.observers.append(FileStorageObserver('logs/config_history'))
ex.add_config("config/upper_controller_main.yaml")


@ex.config
def config_finalize(num_arms, use_obstacle, multi_agent):
    if num_arms == 2:
        arena_type = "diagonal"
    if use_obstacle:
        target_type = "obstacle"
    if not multi_agent:
        action_size = 2 ** num_arms


@ex.command(unobserved=True)
def evaluation(load_model, n_sample, use_obstacle, num_targets, box_plot, _config):
    target_range = []
    obstacle_range = []

    env = CoordinationEnv.make_env(**_config)

    # Save target information for duplicated runs
    for _ in range(n_sample):
        env.generate_n_targets()
        target_range.append(env.target_range)
        if use_obstacle:
            obstacle_range.append(env.obstacle_pts_list)

    agent = Agent(**_config)

    # Target save
    np.save(
        os.path.join(load_model, f"target{num_targets}_test_across_model.npy"), target_range
    )
    # target_range = np.load(os.path.join(load_model, f"target{num_targets}_test_across_model.npy"))

    env.ep = 0
    env.render_dir = os.path.join(load_model, "frames_ppo/")
    os.makedirs(env.render_dir, exist_ok=True)
    agent.test(
        env,
        random=False,
        greedy=False,
        target_range=target_range,
        obstacle_range=obstacle_range,
        **_config
    )

    # Random policy
    env.ep = 0
    env.render_dir = load_model + "/frames_random/"
    os.makedirs(env.render_dir, exist_ok=True)
    agent.test(env, n_sample=n_sample, load_model=load_model, random=True,
               target_range=target_range, greedy=False, obstacle_range=obstacle_range)

    # Greedy policy
    env.ep = 0
    env.render_dir = load_model + "/frames_greedy/"
    os.makedirs(env.render_dir, exist_ok=True)
    agent.test(env, n_sample=n_sample, load_model=load_model, random=False,
               target_range=target_range, greedy=True, obstacle_range=obstacle_range)

    # Q-simplified policy
    env.ep = 0
    env.render_dir = os.path.join(load_model, "frames_simplified/")
    os.makedirs(env.render_dir, exist_ok=True)
    agent.test(
        env,
        random=False,
        greedy=False,
        simplified_soln=True,
        **_config
    )

    if box_plot:
        plot_box(**_config)


@ex.automain
def train(_config, _run):
    """
    Upper level controller trainer.
    """

    # Device Setup
    gpu_devices = tf.config.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(
            device, _config["gpu_set_memory_growth"]
        )

    env = CoordinationEnv.make_env(**_config)

    # Run Agent
    agent = Agent(**_config)
    agent.run(env, **_config, manual_control=False, ex_log_file_writer=LogFileWriter(ex),
              experiment_id=_run._id)  # Set manual_control=True for debug
