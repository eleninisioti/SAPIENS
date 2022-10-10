""" Running this script will launch simulations for collecting all results in the paper.
"""
import sys
import os
import argparse

sys.path.append(os.getcwd())
import gym
import copy
from lib.wordcraft.utils.task_utils import recipe_book_info
from gym.wrappers import FlattenObservation
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
import lib.wordcraft
from lib.wordcraft.wordcraft.env_nogoal import WordCraftEnvNoGoal
import gym
import pickle5 as pickle
import datetime
import os
import tempfile
from lib.stable_baselines3.common.env_util import make_vec_env
from lib.wordcraft.utils.task_utils import recipe_book_info
from lib.stable_baselines3 import A2C
from evaluate import evaluate_project
from lib.stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def wrap(env):
    """
    Wraps the environment with the required wrappers
    """
    env = SquashWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    return env


def build_envs(config, num_envs=10):
    """
    Environment generation tool

    Params
    ----------

    recipe_path : string, default = ".",
        path to the recipe book used to generate the environment

    log_path: string, default = ".",
        path to save tensorboard statistics
    """

    env_name = "wordcraft-multistep-no-goal-v0"
    # use gym to generate the environment with the required parameters
    env = make_vec_env(
        env_id=env_name,
        n_envs=num_envs,
        # wrapper_class=wrap,
        env_kwargs={"env_config": env_config},
    )
    env.reset()

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shared experience DQN")

    # ----- task settings -----
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--nsteps",
        type=int,
        default=5,
        help="maximum innovation level of goal state",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="server",
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="single_path",
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.25,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.1,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--grad",
        type=float,
        default=0.5,
        help="maximum innovation level of goal state",
    )

    parser.add_argument(
        "--n_agents",
        type=int,
        default=10,
        help="maximum innovation level of goal state",
    )

    args = parser.parse_args()

    top_dir = "/a2c/g_" + str(args.gamma) + "_lr_" + str(args.lr) + "_steps_" + str(args.nsteps) + \
              "_cf_" + \
              str(args.vf_coef) + "_coef_" + str(args.ent_coef) + "_grad_" + str(args.grad) + "_n_" + str(args.n_agents)

    if args.mode == "server":
        top_dir = "/gpfsscratch/rech/imi/utw61ti/a2c_log" + top_dir
    else:
        top_dir = "projects" + top_dir
    task = args.task
    project_path = top_dir + "/" + task + "/trial_" + str(args.trial)
    env_config = {
        "log_path": project_path + "/tb_logs",
        "data_path": recipe_book_info[task]["path"],
        "seed": None,
        "recipe_book_path": None,
        "feature_type": "one_hot",
        "shuffle_features": False,
        "random_feature_size": 300,
        "max_depth": 8,
        "split": "by_recipe",
        "train_ratio": 1.0,
        "num_distractors": 0,
        "uniform_distractors": False,
        "max_mix_steps": 8,
        "subgoal_rewards": True,
        "proportional": True}

    policy_kwargs = dict(net_arch=[64] * 3)

    env = build_envs(env_config, args.n_agents)
    eval_env = build_envs(env_config, 1)

    if "single_path" in task:
        start = 10
        stop = int(1e6)
        step = int((stop - start) / 50)
        total_timesteps_values = list(range(start, stop, step))
        total_timesteps = 1e6

    elif "merging" in task:
        total_timesteps = 6e7
    else:
        start = int(42e6)
        stop = int(1e8)
        step = int((stop - start) / 5)
        total_timesteps = 2e8

    c_callback = CheckpointCallback(
        save_freq=2000, save_path=project_path + "/models",
        name_prefix=f"agent_0"
    )
    eval_callback = EvalCallback(
        eval_env,
        log_path=project_path,
        best_model_save_path=project_path + "/models",
        eval_freq=1000,
        deterministic=True,
        render=False,
    )
    model = A2C("MlpPolicy", gamma=args.gamma,
                learning_rate=args.lr,
                n_steps=args.nsteps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                max_grad_norm=args.grad,
                env=env, verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=[c_callback, eval_callback])
    agent = 0
