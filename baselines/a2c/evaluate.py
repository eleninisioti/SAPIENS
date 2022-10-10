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
from plot import plot_project
from lib.stable_baselines3.common.env_util import make_vec_env
from lib.stable_baselines3 import A2C

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
    if len(action_logger):
        del action_logger[-1]
        rew -=r



    return action_logger, new_word_logger, rew, path


def build_envs(env_config):
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
        n_envs=1,
        # wrapper_class=wrap,
        env_kwargs={"env_config":env_config},
    )
    env.reset()

    return env

def is_success(trajectory, recipe):
    success = 0
    if "single" in recipe:
        if "a_8" in trajectory:
            success = 1
    elif "merging" in recipe:
        if "c_10" in trajectory:
            print(trajectory)
            succcss = 1
            print("successfully reached")
        elif "(a_7" in trajectory or ("b_7" in trajectory):
            print("suboptimal in merging")
            print(trajectory)
        else:
            print("just bad traj", trajectory)
        
    else:
        if "j_14" in trajectory:
            success = 1
            print(trajectory)

        elif "_4" in trajectory and "j_4" not in trajectory:
            print("suboptimal in 10path")


    return success



def find_nsteps(top_dir, agent_idx):
    """ Find the training steps for a project.
     """

    # find all files in directory
    files = [f for f in os.listdir(top_dir) if f.endswith('.zip')]
    nsteps = []
    for name in files:
        if "agent_" + str(agent_idx) in name:
            try:
                nsteps.append((float(name[8:(-10)])))
            except:
                nsteps.append((float(name[9:(-10)])))

    if not len(nsteps):
        print("Error: project has no saved models. Skipping")
        return False
    final_step = max(nsteps)
    nsteps.sort()
    return nsteps

def evaluate_project(project, task, n_trials, max_rew, n_agents):
    """
    Evaluates all models under a project and returns dataframes for how different metrics evolve  training time.

    Parameters
    ---------
    project: str
        directory of project
    """


    recipe_name = task

    max_rew = max_rew

    n_agents = n_agents
    n_trials = n_trials
    env_config = {
        "log_path":project + "/tb_logs",
        "data_path" :  recipe_book_info[task]["path"],
        "seed": None,
        "recipe_book_path":None,
        "feature_type":"one_hot",
        "shuffle_features":False,
        "random_feature_size":300,
        "max_depth":9,
        "split":"by_recipe",
        "train_ratio":1.0,
        "num_distractors":0,
        "uniform_distractors":False,
        "max_mix_steps":9,
        "subgoal_rewards":True,
        "proportional":True}

    env = build_envs(env_config)

    # ---- evaluate -----

    for trial in range(n_trials):
        task_transform = {"single_path": "1path",
                          "merging_paths": "cross_easier",
                          "bestoften_paths": "10easier"}
        config = {"shape": "A2C",
                  "recipe_path":task_transform[task],
                          "n_agents": 10}
        with open(project + "/trial_" + str(trial)  + "/config.yaml", "w") as f:
            yaml.dump(config, f)

    for trial in range(n_trials):

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
        successes = []

        last_length = -1

        n_steps = find_nsteps(project + "/trial_" + str(trial) + "/models", 0)

        for i, step in enumerate(n_steps):

            for agent in range(n_agents):


                path = project  + "/trial_" + str(trial) + "/models/agent_" + str(agent) + "_" +  str(int(step)) + \
                       "_steps"

                if os.path.exists(path + ".zip"):
                    try:
                        model = A2C.load(path)

                    except FileNotFoundError:
                        break

                    env.reset()
                    actions, unique_words, rewards, trajectory = eval_model(model, env)
                    print('step is ', str(step), "for trial ", str(trial))
                    success = is_success(trajectory, recipe_name)

                    all_levels = [0] + [int(el[(el.rindex("_") + 1):]) for el in unique_words]
                    level = max(all_levels)
                    total_rewards.append(float(rewards))
                    total_agents.append(agent)
                    total_trajectories.append(trajectory)
                    total_steps.append(int(step/n_agents))
                    total_trials.append(trial)
                    total_levels.append(level)
                    successes.append(success)



        occurs = {}

        #print(total_trajectories, total_rewards)
        #print(np.array(total_rewards) / max_rew)

        rewards = pd.DataFrame({"train_step": total_steps,
                                  "norm_reward": np.array(total_rewards) / max_rew,
                                  "agent": total_agents,
                                  "trial": total_trials,
                                  "level": total_levels,
                                  "trajectory": total_trajectories,
                                "success": successes})

        # ----- save evaluation data -----
        eval_save_dir = project + "/trial_" + str(trial)  + "/data/eval"

        if os.path.exists(eval_save_dir):
            shutil.rmtree(eval_save_dir)

        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir, exist_ok=True)

        with open(eval_save_dir + "/trajectories.pkl", "wb") as f:
            pickle.dump(rewards, f)

        with open(eval_save_dir + "/rewards.pkl", "wb") as f:
            pickle.dump(rewards, f)

        with open(eval_save_dir + "/occurs.pkl", "wb") as f:
            pickle.dump(occurs, f)



def compare_projects(projects, parameter, save_dir):
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
        config = yaml.safe_load(open(project + "/config.yaml", "r"))
        label = config[parameter]

        total_eval_info[label] = eval_info

        with open(eval_save_dir + "/behavioral_metrics.pkl", "rb") as f:
            beh_metrics = pickle.load(f)
        total_volatilities[label] = beh_metrics["volatilities"]

        total_conformities[label] = beh_metrics["conformities"]

    plot_project(total_eval_info, total_volatilities, total_conformities, config["measure_mnemonic"], save_dir)
