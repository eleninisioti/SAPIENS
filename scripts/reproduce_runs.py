""" Running this script will launch simulations for collecting all results in the paper.
"""
import sys
sys.path.append(".")
import gym
import copy
from lib.wordcraft.utils.task_utils import recipe_book_info
from gym.wrappers import FlattenObservation
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
import lib.wordcraft
from lib.wordcraft.wordcraft.env_nogoal import WordCraftEnvNoGoal
from sapiens.sapiens import Sapiens
from scripts.evaluate import evaluate_project
from scripts.script_utils import build_envs


def run_all_main():
    """ Runs all experiments used in plots in the main paper.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """


    project_dir = top_dir + "/paper/main"
    shapes = [ "no-sharing", "ring" , "fully-connected" ]
    n_agents = 10

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2
    n_trials = 10

    #tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}
    #tasks = {"bestoften_paths": 500000}
    tasks = {"merging_paths": 500000}

    for trial in range(n_trials):

        for task, total_episodes in tasks.items():

            for shape in shapes:

                project_path = project_dir + "/task_" + task + "/shape_" + shape
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
                                trial=trial)
                # train
                group.learn()

                # evaluate
                evaluate_project(project_path)


def run_scaling():
    """ Runs experiments studying the effect of group size.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/paper/appendix/scaling"
    shapes = ["fully-connected", "small-world", "ring", "dynamic-Boyd"]

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2

    group_sizes = [6, 20, 50]

    tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}

    for task, total_episodes in tasks.items():

        for shape in shapes:

            for n_agents in group_sizes:

                project_path = top_dir + "/task_" + task + "/shape_" + shape + "/size_" + str(n_agents)
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
                            n_trials=10)
                # train
                group.learn()

                # evaluate
                evaluate_project(project_path)


def run_varying_dynamic_topologies():
    """ Runs experiments studying the effect of group size.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/paper/appendix/varying_dynamic/dynamic_Boyd"
    shape = "dynamic-Boyd"

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2
    n_agents = 10

    # configure dynamic topology
    migrate_rates = [0.005, 0.01, 0.1, 0.5]
    visit_durations = [10, 100, 100]


    tasks = {"merging_paths": 500000, "bestoften_paths": 500000}

    for task, total_episodes in tasks.items():

        for migrate_rate in migrate_rates:
            for visit_duration in visit_durations:


                project_path = top_dir + "/task_" + task + "/migrate_" + str(migrate_rate) + "_duration_" + str(
                    visit_duration)

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
                                visit_duration=visit_duration,
                                migrate_rate=migrate_rate,
                            n_trials=10)
                # train
                group.learn()

                # evaluate
                evaluate_project(project_path)

    # ----- tuning dynamic-periodic
    top_dir = "/media/elena/LaCie/SAPIENS/projects/paper/appendix/varying_dynamic/dynamic_Boyd"
    shape = "dynamic-periodic"

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2
    n_agents = 10

    # configure dynamic topology
    phase_period_values = [[10,10],[10,100],[10,1000],[100,10],[100,100],[100,1000],[1000,10],[1000,100],[1000,1000]]

    tasks = {"merging_paths": 500000, "bestoften_paths": 500000}

    for task, total_episodes in tasks.items():

        for phase_periods in phase_period_values:

                project_path = top_dir + "/task_" + task + "/migrate_" + str(migrate_rate) + "_duration_" + str(
                    visit_duration)

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
                                phase_periods=phase_periods,
                                n_trials=10)
                # train
                group.learn()

                # evaluate
                evaluate_project(project_path)


def run_mnemonic():
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the study_mnemonic flag

    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/paper/mnemonic"
    shapes = ["no-sharing", "fully-connected", "small-world", "ring", "dynamic-Boyd"]
    n_agents = 10

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2


    tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}

    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = top_dir + "/task_" + task + "/shape_" + shape
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
                            n_trials=10)
            # train
            group.learn()

            # evaluate
            evaluate_project(project_path)

def run_intergroup_alignment():
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the measure_intergroup_alignment flag

    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/paper/mnemonic"
    shapes = ["no-sharing", "fully-connected", "small-world", "ring", "dynamic-Boyd"]
    n_agents = 10

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2

    tasks = {"single_path": 50000, "merging_paths": 50000, "bestoften_paths": 50000}

    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = top_dir + "/task_" + task + "/shape_" + shape
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
                            n_trials=10)
            # train
            group.learn()

            # evaluate
            evaluate_project(project_path)

if __name__ == "__main__":

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

    top_dir = "projects"

    # reproduce experiments analysed in the main paper
    run_all_main()




