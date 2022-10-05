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
from server_scripts.plot import plot_project
from server_scripts.compute_metrics import compute_metrics_project
from server_scripts.script_utils import build_envs, find_ntrials
from lib.wordcraft.utils.task_utils import recipe_book_info


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
    group_diversity = np.mean(model.group_diversities[:last_length])
    intragroup_alignment = np.mean(model.intragroup_alignments[:last_length])
    last_length = len(model.diversities)

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
    env_temp = env.envs[0]
    while not done:
        action = model.predict(obs, deterministic=True)

        action_logger.append(action[0])
        obs, r, done, info = env.step(action[0])
        rew += r

        if not len(env_temp.selection):  # composition step
            _table = np.array(env_temp.table_index)[np.array(env_temp.table_index) != -1]
            idx = _table[len(_table) - 1]
            e = env_temp.recipe_book.entities[idx]
            if e not in new_word_logger and "base" not in e and "dist" not in e:
                new_word_logger.append(e)

            path.append(e)

        i += 1

    if len(new_word_logger):
        del new_word_logger[-1]
    if len(new_word_logger):
        del action_logger[-1]
        rew -=r
    path = [str(el) + "," for el in path]
    path = ','.join(path)
    return action_logger, new_word_logger, rew, path



def evaluate_project(project, playground="wordcraft"):
    """
    Evaluates all models under a project and returns dataframes for how different metrics evolve  training time.

    Parameters
    ---------
    project: str
        directory of project
    """

    project = project
    config = yaml.safe_load(open(project + "/config.yaml", "r"))
    recipe_book = config["task"]
    #recipe_name = [key for key, value in recipe_book_info.items() if value["path"] == recipe_path][0]

    env_config = {
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
    env_config["log_path"] = project
    env_config["data_path"] = recipe_book_info[recipe_book]["path"]


    max_rew = recipe_book_info[recipe_book]["best_reward"]
    n_steps = list(range(0, config["total_episodes"] * 16, 10000))
    env = build_envs(env_config)

    n_agents = config["n_agents"]
    n_trials = find_ntrials(project)

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
    occurs_steps = []
    occurs_trials = []

    last_length = -1

    for i, step in enumerate(n_steps[1:]):
        for trial in range(n_trials):

            for agent in range(n_agents):
                if step==-1:
                    path = project + "/trial_" + str(trial) + "/models/agent_" + str(agent) + "_" + str(step) + "_steps"
                else:

                    path = project + "/trial_" + str(trial) + "/models/agent_" + str(agent) + "_" + str(step) + "_steps"

                if os.path.exists(path + ".zip"):
                    try:
                        model = DQN.load(path)

                    except FileNotFoundError:
                        break

                    env.reset()
                    actions, unique_words, rewards, trajectory = eval_model(model, env)

                    all_levels = [0] + [int(el[(el.rindex("_") + 1):]) for el in unique_words]
                    level = max(all_levels)
                    total_rewards.append(float(rewards[0]))
                    total_agents.append(agent)
                    total_trajectories.append(trajectory)
                    total_steps.append(step)
                    total_trials.append(trial)
                    total_levels.append(level)

                if config["measure_mnemonic"]:
                    diversity, group_diversity, intragroup_alignment, last_length, buffer_keys, buffer_values = \
                        process_mnemonic(model, last_length, process_occurs=config["measure_intergroup_alignment"])

                    total_buffer_keys.extend(buffer_keys)
                    total_buffer_values.extend(buffer_values)
                    occurs_steps.extend([step] * len(buffer_keys))
                    occurs_trials.extend([trial] * len(buffer_keys))
                else:
                    diversity = group_diversity = intragroup_alignment = 0

                total_diversities.append(diversity)
                total_group_diversities.append(group_diversity)
                total_intragroup_alignment.append(intragroup_alignment)

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
                                   "train_step": occurs_steps,
                                   "trial": occurs_trials})

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
    volatilities, conformities = compute_metrics_project(project)

    with open(eval_save_dir + "/behavioral_metrics.pkl", "wb") as f:
        pickle.dump({"volatility": volatilities, "conformity": conformities}, f)

    plot_project({"": eval_info}, {"": volatilities}, {"": conformities}, config["measure_mnemonic"], project)


def compare_projects(projects, parameter, save_dir, task):
    """  Compares multiple projects whose configuration differs in a desired parameter.

    For example, if parameter="shape", we compare different topologies. If paramter="n_agents", we compare different
    group sizes. As comparisons, we produce plots of all performance metrics.

    Params
    ------
    projects: list of str
        project directories

    parameter: str
        name of parameter for comparison

    """
    total_eval_info = {}
    total_volatilities = {}
    total_conformities = {}
    for project in projects:
        # load eval_info
        eval_save_dir = project + "/data"

        with open(eval_save_dir + "/eval_info.pkl", "rb") as f:
            eval_info = pickle.load(f)

        # find label of project
        config = yaml.safe_load(open(project + "/trial_0/config.yaml", "r"))
        label = config[parameter]

        total_eval_info[label] = eval_info

        with open(eval_save_dir + "/behavioral_metrics.pkl", "rb") as f:
            beh_metrics = pickle.load(f)
        total_volatilities[label] = beh_metrics["volatility"]

        total_conformities[label] = beh_metrics["conformity"]

    if "measure_mnemonic" not in config.keys():
        config["measure_mnemonic"] = False


    max_step = recipe_book_info[task]["max_steps"]

    plot_project(total_eval_info, total_volatilities, total_conformities, config["measure_mnemonic"], save_dir,
                 max_step)
