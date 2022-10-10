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
from scripts.plot import plot_project
from scripts.compute_metrics import compute_metrics_project
from scripts.script_utils import build_envs, find_ntrials
from lib.wordcraft.utils.task_utils import recipe_book_info


def measure_volatility(trajectories, n_trials):
    """ Measure volatility.

    Volatility is the cumulative number of switches in the policy followed by an agent during consecutive evaluation
    episodes.

    trajectories: Dataframe
        contains infromation collected during the evaluation of the project

    n_trials: int
        number of trials for this project


    Returns information in two formats: a list for saving as a yaml file and a dataframe for plotting
    """
    n_agents = max(trajectories["agent"]) + 1
    n_trials = n_trials
    df_volatility = {"train_step": [], "volatility": [], "agent": [], "trial": []}
    volatility = []
    for trial in range(n_trials):
        trial_switches = []

        trial_traj = trajectories.loc[trajectories["trial"] == trial]
        for agent in range(n_agents):

            agent_traj = trial_traj.loc[trial_traj["agent"] == agent]
            steps = list(agent_traj["train_step"])
            switches = [0]

            for idx, step in enumerate(steps[1:]):
                current_traj = agent_traj.loc[agent_traj["train_step"] == step]["trajectory"].tolist()[0].split(",")
                prev_traj = agent_traj.loc[agent_traj["train_step"] == steps[idx]]["trajectory"].tolist()[0].split(",")
                transition = pd.DataFrame({"after": current_traj, "before": prev_traj})
                diffs = list(np.where(transition["after"] != transition["before"], 1, 0))
                switches.append(switches[-1] + np.prod(diffs))

                df_volatility["train_step"].append(step)
                df_volatility["volatility"].append(switches[-1])
                df_volatility["agent"].append(agent)
                df_volatility["trial"].append(trial)

            trial_switches.append(switches[-1] / len(steps))

        volatility.append(np.mean(trial_switches))

    df_volatility = pd.DataFrame.from_dict(df_volatility)

    return volatility, df_volatility


def measure_conformity(trajectories, n_trials, n_agents):
    """ Measures conformity.

    Conformity is a behavioral metric that measures the percentage of agents following the same trajectory during the
    same evaluation trial.

    Params
    ------
    trajectories: Dataframe
        contains infromation collected during the evaluation of the project

    n_trials: int
        number of trials for this project

    n_agents: int
        number of agents

    Returns information in two formats: a list for saving as a yaml file and a dataframe for plotting

    """
    conformity = []
    df_conformity = {"train_step": [], "group_conformity": [], "trial": []}

    for trial in range(n_trials):
        agent_final_states = []
        for agent in range(n_agents):

            agent_traj = trajectories.loc[trajectories["agent"] == agent]
            steps = list(set(list(agent_traj["train_step"])))

            final_states = []
            for idx, step in enumerate(steps):
                traj = agent_traj.loc[agent_traj["train_step"] == step]
                traj = traj["trajectory"].values.tolist()[0]

                final_states.append(traj[-1])
            agent_final_states.append(final_states)

        trial_conformities = []
        for idx, step in enumerate(steps):
            current_step = set([agent_final_states[agent][idx] for agent in range(n_agents) if idx < len(
                agent_final_states[agent])])
            current_conformity = 1 - ((len(current_step) - 1) / n_agents)
            trial_conformities.append(current_conformity)
            df_conformity["train_step"].append(step)
            df_conformity["group_conformity"].append(current_conformity)
            df_conformity["trial"].append(trial)

        conformity.append(np.mean(trial_conformities))

    df_conformity = pd.DataFrame.from_dict(df_conformity)

    return conformity, df_conformity


def compute_performance_metrics(eval_info, n_trials, n_agents):
    """ Compute performance-based metrics.


    Params
    ------
    eval_info: Dataframe
        contains infromation collected during the evaluation of the project

    n_trials: int
        number of trials for this project

    n_agents: int
        number of agents
    """
    metrics = {"time_to_first_success": [], "time_to_all_successes": [], "spread_time": [],
               "group_success": [], "avg_reward_conv": [], "max_reward_conv": []}

    for trial in range(n_trials):

        # ----- at which timestep did at least one agent find the correct solution -----
        first_steps = []
        results = eval_info.loc[(eval_info["trial"] == trial)]

        for agent in range(n_agents):

            results_max = results.loc[(results["norm_reward"] == 1.0)]
            results_max = results_max.loc[(results["agent"] == agent)]
            steps = list(results_max["train_step"])
            if len(steps):
                first_steps.append(min(steps))

        if len(first_steps):
            first_step_one = min(first_steps)
            failed_trial = 0
        else:
            first_step_one = np.nan
            failed_trial = 1

        # detect when all agents found the optimal solution
        if len(first_steps) == n_agents:
            first_step_all = max(first_steps)
            time_to_spread = first_step_all - first_step_one
        else:
            first_step_all = np.nan
            time_to_spread = np.nan

        metrics["time_to_all_successes"].append(first_step_all)
        metrics["time_to_first_success"].append(first_step_one)
        metrics["group_success"].append(1 - failed_trial)
        metrics["spread_time"].append(time_to_spread)

        last_rewards = []
        for agent in range(n_agents):
            agent_results = results.loc[(results["agent"] == agent)]
            last_step = max(list(agent_results["train_step"]))
            last_reward = agent_results.loc[agent_results["train_step"] == last_step]
            last_reward = float(last_reward["norm_reward"])
            last_rewards.append(last_reward)

        metrics["avg_reward_conv"].append(np.mean(last_rewards))
        metrics["max_reward_conv"].append(np.max(last_rewards))

    return metrics


def compute_behavioral_metrics(eval_info, n_trials, n_agents):
    """ Compute behavioral metrics

    Params
    ------
    eval_info: Dataframe
        contains infromation collected during the evaluation of the project

    n_trials: int
        number of trials for this project

    n_agents: int
        number of agents
    """
    volatility, df_volatility = measure_volatility(eval_info, n_trials)
    conformity, df_conformity = measure_conformity(eval_info, n_trials, n_agents)

    metrics = {"volatility": volatility, "conformity": conformity}

    return metrics, df_volatility, df_conformity


def compute_metrics_project(project):
    """ Compute all metrics for project

    Params
    ------
    project: str
        directory of project (under SAPIENS)
    """
    config = yaml.safe_load(open(project + "/config.yaml", "r"))
    n_agents = config["n_agents"]
    n_trials = find_ntrials(project)

    with open(project + "/data/eval_info.pkl", "rb") as f:
        eval_info = pickle.load(f)

        performance_metrics = compute_performance_metrics(eval_info, n_trials, n_agents)
        behavioral_metrics, df_volatility, df_conformity = compute_behavioral_metrics(eval_info, n_trials, n_agents)

        metrics = {**performance_metrics, **behavioral_metrics}

        # pkl file contains values in all trials
        save_file = project + "/data/pop_metrics.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(metrics, f)

        # yaml file contains average over trials
        metrics_mean = {}
        metrics_var = {}
        for key, value in metrics.items():
            metrics_mean[key + "_mean"] = float(np.nanmean(value))
            metrics_var[key + "_var"] = float(np.nanvar(value))

        metrics_stat = {**metrics_mean, **metrics_var}

        save_file = project + "/data/pop_metrics.yaml"
        with open(save_file, "w") as f:
            yaml.dump(metrics_stat, f)

    return df_volatility, df_conformity


def measure_intergroup_alignment(projects):
    """ Measure intergroup alignment.

    Params
    ------
    projects: list of str
        directories of projects for comparing alignment

    """
    total_occurs = {}
    for project in projects:
        # find label of project
        config = yaml.safe_load(open(project + "/config.yaml", 'r'))
        label = config["shape"]

        with open(project + "/data/occurs.pkl", "rb") as f:
            occurs = pickle.load(f)
        total_occurs[label] = occurs

    n_steps = list(range(0, config["total_episodes"] * 16, 10000))
    n_trials = config["n_trials"]
    total_df = []
    for step in n_steps[1:]:
        step_diffs = {}
        for trial in range(n_trials):
            done_pairs = []

            for idx1, data1 in total_occurs.items():
                for idx2, data2 in total_occurs.items():
                    if idx1 != idx2 and tuple([idx1, idx2]) not in done_pairs and  tuple([idx2, idx1]) not in \
                            done_pairs:

                        current_data1 = data1.loc[data1["trial"] == trial]
                        current_data1 = current_data1.loc[current_data1["train_step"] == step]
                        current_data1 = current_data1.groupby('buffer_keys')['buffer_values'].apply(list).to_dict()
                        #to_dict()

                        current_data2 = data2.loc[data2["trial"] == trial]
                        current_data2 = current_data2.loc[current_data2["train_step"] == step]
                        current_data2 = current_data2.groupby('buffer_keys')['buffer_values'].apply(list).to_dict()

                        # compare the two dicts
                        list1 = []
                        for key, value in current_data1.items():
                            for _ in range(value[0]):
                                list1.append(key)
                        list1.sort()
                        list2 = []
                        for key, value in current_data2.items():
                            for _ in range(value[0]):
                                list2.append(key)
                        list2.sort()

                        diffs = 0
                        for idx, el in enumerate(list1):
                            if (len(list2) > idx) and el != list2[idx]:
                                diffs += 1

                        diff = diffs / max([len(list1), len(list2), 1])
                        df = pd.DataFrame(columns=["pair", "trial", "train_step", "diff"])
                        df.loc[0] = [tuple([idx1, idx2]), trial, step, 1 - diff]

                        if len(total_df):

                            total_df = total_df.append(df, ignore_index=True)
                        else:
                            total_df = df

                        done_pairs.append(tuple([idx1, idx2]))

    return total_df
