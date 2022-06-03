""" This script can be used to evaluate already trained models.

Only pickle files are produced. To get plots and metrics, run analyse.py
"""
# ----- generic imports -----
import os
import sys

import shutil
import numpy as np
import pandas as pd
import pickle
import os
import yaml
import copy
import gym

# ----- project-specific imports -----
from lib.stable_baselines3 import DQN
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
from lib.wordcraft.utils.task_utils import recipe_book_info
from gym.wrappers import FlattenObservation
from scripts.plot_project import plot
from scripts.compute_metrics_project import compute_metrics


def process_mnemonic(model, last_length, process_occurs):
    """ Process mnemonic metrics of model

    Params
    ------
    model: instance of ES_DQN
        the trained model

    last_length: int
        last processed timestep

    process_occurs: bool
        if True, proccess occurences for computing inter-group alignment
    """

    diversity = np.mean(model.diversities[:last_length])
    group_diversity =  np.mean(model.group_diversities[:last_length])
    intragroup_alignment = np.mean(model.intragroup_alignments[:last_length])
    last_length = len(diversity)

    if process_occurs:
        group_occurs = model.group_occurs[-1]

        buffer_keys = list(group_occurs.keys())
        buffer_values = list(group_occurs.values())

    else:
        buffer_keys = []
        buffer_values = []



    return diversity, group_diversity, intragroup_alignment, last_length, buffer_keys, buffer_values


def eval_model(model, env):
    """
    A model is evaluated for a single episode


    Parameters
    ----------
    model: ES_DQN
        trained model

    env: WordcraftEnv
        a gym environment
    """
    path = []  # contains the path chosen during the episode
    action_logger = []  # contains the actions chosen during the episode
    new_word_logger = []  # contains the unique words created during the episode
    obs = env.reset()
    i = rew = done = 0
    while not done:
        action = model.predict(obs, deterministic=True)

        action_logger.append(action[0])
        obs, r, done, info = env.step(action[0])
        rew += r
        if not len(env.selection):  # composition step
            _table = np.array(env.table_index)[np.array(env.table_index) != -1]
            idx = _table[len(_table) - 1]
            e = env.recipe_book.entities[idx]
            if e not in new_word_logger and "base" not in e and "dist" not in e:
                new_word_logger.append(e)

            path.append(e)

        i += 1

    return action_logger, new_word_logger, rew, path


def build_envs(recipe_path="", log_path=""):
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
    env = gym.make(
        env_name,
        max_depth=8,
        max_mix_steps=8,
        num_distractors=0,
        subgoal_rewards=True,
        data_path=recipe_path,
        feature_type="one_hot"
    )

    # apply wrappers
    wrapper = SquashWrapper
    env = wrapper(env, proportional=True)
    env = FlattenObservation(env)

    env.reset()

    eval_env = copy.deepcopy(env)
    eval_env.reset()

    return env, eval_env



def evaluate(project):
    """
    Evaluates all models under a project and returns dataframes for how different metrics evolve  training time.

    Parameters
    ---------
    project: str
        directory of project

    n_trials: int
        number of independent evaluation trials


    recipe_name: str
        alis of recipe book

    episode_len: int
        length of episode

    n_distractors: int
        number of distractors
    """
    project = "projects/" + project
    config = yaml.safe_load(open(project + "/config.yaml", "r"))
    recipe_path = config["recipe_path"]
    recipe_name = [key for key, value in recipe_book_info.items() if value["path"]==recipe_path][0]
    max_rew = recipe_book_info[recipe_name]["best_reward"]
    n_agents = config["n_agents"]
    n_steps = list(range(0, config["total_episodes"], 10000))
    n_trials = config["n_trials"]

    env, _ = build_envs(recipe_path=recipe_book_info[recipe_name]["path"])


    # ---- evaluate -----
    total_rewards = []
    total_agents = []
    total_trajectories = []
    total_steps = []
    total_trials = []
    total_levels = []

    total_diversities = []
    total_group_diversities = []
    total_intragroup_alignment = []
    total_buffer_keys = []
    total_buffer_values = []

    last_length = 0

    for i, step in enumerate(n_steps[1:]):
        for trial in range(n_trials):

            for agent in range(n_agents):

                path = project + "/trial_" + str(trial) + "/models/agent_" + str(agent) + "_" + str(step) + "_steps"

                if os.path.exists(path + ".zip"):
                    try:
                        model = DQN.load(path)

                    except FileNotFoundError:
                        break

                    env.reset()
                    actions, unique_words, rewards, trajectory = eval_model(model, env)
                    if rewards:
                        print("check")
                    all_levels = [0] + [int(el[(el.rindex("_") + 1):]) for el in unique_words]
                    level = max(all_levels)
                    total_rewards.append(rewards)
                    total_agents.append(agent)
                    total_trajectories.append(trajectory)
                    total_steps.append(step)
                    total_trials.append(trial)
                    total_levels.append(level)

                if config["measure_mnemonic"]:
                    diversity, group_diversity, intragroup_alignment, last_length, buffer_keys, buffer_values = process_mnemonic(model, last_length)

                    total_diversities.append(diversity)
                    total_group_diversities.append(group_diversity)
                    total_intragroup_alignment.append(intragroup_alignment)
                    total_buffer_keys.append(buffer_keys)
                    total_buffer_values.append(buffer_values)
    occurs = {}

    if config["measure_mnemonic"]:
        eval_info = pd.DataFrame({"train_step": total_steps,
                                   "norm_reward": np.array(total_rewards) / max_rew,
                                   "agent": total_agents,
                                   "trial": total_trials,
                                   "level": total_levels,
                                   "trajectory": total_trajectories,
                                   "diversity": total_diversities,
                                   "group_diversity": total_group_diversities,
                                   "intragroup_alignment": total_intragroup_alignment})

        if config["measure_intergroup_alignment"]:

            occurs = pd.DataFrame({"buffer_keys": total_buffer_keys,
                                        "buffer_values": total_buffer_values,
                                        "train_step": total_steps,
                                        "trial": total_trials})

    else:

        eval_info = pd.DataFrame({"train_step": total_steps,
                                   "norm_reward": np.array(total_rewards) / max_rew,
                                   "agent": total_agents,
                                   "trial": total_trials,
                                   "level": total_levels,
                                   "trajectory": total_trajectories})


    # ----- save evaluation data -----
    eval_save_dir = project + "/data"

    if os.path.exists(eval_save_dir):
        shutil.rmtree(eval_save_dir)

    if not os.path.exists(eval_save_dir):
        os.makedirs(eval_save_dir, exist_ok=True)

    with open(eval_save_dir + "/eval_info.pkl", "wb") as f:
        pickle.dump(eval_info, f)

    with open(eval_save_dir + "/occurs.pkl", "wb") as f:
        pickle.dump(occurs, f)

    # ----- produce evaluation plots -----
    volatilities, conformities = compute_metrics(project)
    plot(eval_info, volatilities, conformities, config["measure_mnemonic"],  project)


