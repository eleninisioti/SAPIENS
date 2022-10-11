""" Running this script will launch simulations for collecting all results in the paper.
"""
import sys

import os
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
project="/gpfsdswork/projects/rech/imi/utw61ti/SAPIENS/scripts"
project="/gpfsdswork/projects/rech/imi/utw61ti/SAPIENS"

sys.path.append(project)

print(sys.path)

import gym
import copy
from lib.wordcraft.utils.task_utils import recipe_book_info
from gym.wrappers import FlattenObservation
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
import lib.wordcraft
from lib.wordcraft.wordcraft.env_nogoal import WordCraftEnvNoGoal
from sapiens.sapiens import Sapiens
from server_scripts.evaluate import evaluate_project
from lib.stable_baselines3.common.env_util import make_vec_env
import datetime

def build_envs(env_config, n_envs=1):
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
        n_envs=n_envs,
        # wrapper_class=wrap,
        env_kwargs={"env_config": env_config},
    )
    env.reset()

    return env


def mnemonic_nosharing_merging(trial):
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the study_mnemonic flag

    """
    now = datetime.datetime.now()
    project = top_dir + str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "/mnemonic_"

    shape = "no-sharing"
    n_agents = 10
    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2
    tasks = {"merging_paths": 1e7}

    for task, total_episodes in tasks.items():

            project_path = project + "/task_" + task + "/shape_" + shape
            train_envs = []
            eval_envs = []
            for i in range(n_agents):

                env_config["log_path"] = project_path
                env_config["data_path"] = recipe_book_info[task]["path"]
                train_env = build_envs(env_config)
                eval_env = build_envs(env_config)

                train_envs.append(train_env)
                eval_envs.append(eval_env)

            group = Sapiens(gamma=gamma,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            num_neurons=num_neurons,
                            num_layers=num_layers,
                            n_agents=n_agents,
                            shape=shape,
                            train_envs=train_envs,
                            eval_envs=eval_envs,
                            project_path=project_path,
                            total_episodes=total_episodes,
                            measure_mnemonic=True,
                            trial=trial)
            # train
            #group.learn()

            # evaluate
            evaluate_project(project_path)


def mnemonic_dynamic_merging(trial):
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the study_mnemonic flag

    """
    now = datetime.datetime.now()
    project = top_dir + str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "/mnemonic_"

    shape = "dynamic-Boyd"
    n_agents = 10
    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2
    tasks = {"merging_paths": 1e7}

    for task, total_episodes in tasks.items():

        project_path = project + "/task_" + task + "/shape_" + shape
        train_envs = []
        eval_envs = []
        for i in range(n_agents):
            env_config["log_path"] = project_path
            env_config["data_path"] = recipe_book_info[task]["path"]
            train_env = build_envs(env_config)
            eval_env = build_envs(env_config)

            train_envs.append(train_env)
            eval_envs.append(eval_env)

        group = Sapiens(gamma=gamma,
                        buffer_size=buffer_size,
                        batch_size=batch_size,
                        num_neurons=num_neurons,
                        num_layers=num_layers,
                        n_agents=n_agents,
                        shape=shape,
                        train_envs=train_envs,
                        eval_envs=eval_envs,
                        project_path=project_path,
                        total_episodes=total_episodes,
                        measure_mnemonic=True,
                        trial=trial)
        # train
        group.learn()

        # evaluate
        evaluate_project(project_path)


def run_intergroup_alignment(trial):
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the measure_intergroup_alignment flag

    """

    now = datetime.datetime.now()
    project = top_dir + str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "/alignment"
    shapes = ["no-sharing", "fully-connected", "small-world", "ring", "dynamic-Boyd"]
    n_agents = 10

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2

    tasks = {"single_path": 50000}


    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = project + "/task_" + task + "/shape_" + shape
            train_envs = []
            eval_envs = []
            for i in range(n_agents):

                env_config["log_path"] = project_path
                env_config["data_path"] = recipe_book_info[task]["path"]
                train_env = build_envs(env_config)
                eval_env = build_envs(env_config)
                train_envs.append(train_env)
                eval_envs.append(eval_env)

            group = Sapiens(gamma=gamma,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            num_neurons=num_neurons,
                            num_layers=num_layers,
                            n_agents=n_agents,
                            shape=shape,
                            train_envs=train_envs,
                            eval_envs=eval_envs,
                            project_path=project_path,
                            total_episodes=total_episodes,
                            measure_intergroup_alignment=True,
                            measure_mnemonic=True,
                            trial=trial,
                            task=task)
            # train
            group.learn()

            # evaluate
            #evaluate_project(project_path)


if __name__ == "__main__":

    trial = int(sys.argv[1])

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

    top_dir = "/gpfsscratch/rech/imi/utw61ti/sapiens_log/projects/"
    run_intergroup_alignment(trial) #  intra-group aligment in 6.4.3, as well as all diversity plots


