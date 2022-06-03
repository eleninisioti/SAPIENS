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
from scripts.eval_project import evaluate


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


def run_all_main():
    """ Runs all experiments used in plots in the main paper.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """

    top_dir = "paper/main"
    shapes = ["no-sharing", "fully-connected", "small-world", "ring", "dynamic-Boyd"]
    n_agents = 10

    gamma = 0.9
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2

    #tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}
    tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}

    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = top_dir + "/task_" + task + "/shape_" + shape
            train_envs = []
            eval_envs = []
            for i in range(n_agents):
                train_env, eval_env = build_envs(
                    recipe_path=recipe_book_info[task]["path"],
                    log_path=project_path + "/tb_logs"
                )
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
                            total_episodes=total_episodes)
            # train
            group.learn()

            # evaluate
            evaluate(project_path)


def run_scaling():
    """ Runs experiments studying the effect of group size.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """

    top_dir = "paper/appendix/scaling"
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
                    train_env, eval_env = build_envs(
                        recipe_path=recipe_book_info[task]["path"],
                        log_path=project_path + "/tb_logs"
                    )
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
                                total_episodes=total_episodes)
                # train
                group.learn()

                # evaluate
                evaluate(project_path)


def run_varying_dynamic_topologies():
    """ Runs experiments studying the effect of group size.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).

    """

    top_dir = "paper/appendix/varying_dynamic/dynamic_Boyd"
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
                    train_env, eval_env = build_envs(
                        recipe_path=recipe_book_info[task]["path"],
                        log_path=project_path + "/tb_logs"
                    )
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
                                migrate_rate=migrate_rate)
                # train
                group.learn()

                # evaluate
                evaluate(project_path)

    # ----- tuning dynamic-periodic
    top_dir = "paper/appendix/varying_dynamic/dynamic_Boyd"
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
                    train_env, eval_env = build_envs(
                        recipe_path=recipe_book_info[task]["path"],
                        log_path=project_path + "/tb_logs"
                    )
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
                                phase_periods=phase_periods)
                # train
                group.learn()

                # evaluate
                evaluate(project_path)


def run_mnemonic():
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the study_mnemonic flag

    """

    top_dir = "paper/mnemonic"
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
                train_env, eval_env = build_envs(
                    recipe_path=recipe_book_info[task]["path"],
                    log_path=project_path + "/tb_logs"
                )
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
                            measure_mnemonic=True)
            # train
            group.learn()

            # evaluate
            evaluate(project_path)

def run_intergroup_alignment():
    """ Runs all experiments used in plots in the main paper.

        In particular, we run the five social network topologies on the three tasks for 10 agents. We keep track
        of diversity and intra-group alignment by activating the study_mnemonic flag

    """

    top_dir = "paper/mnemonic"
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
                train_env, eval_env = build_envs(
                    recipe_path=recipe_book_info[task]["path"],
                    log_path=project_path + "/tb_logs"
                )
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
                            measure_intergroup_alignment=True)
            # train
            group.learn()

            # evaluate
            evaluate(project_path)

if __name__ == "__main__":

    # reproduce experiments analysed in the main paper
    run_all_main()

    # ----- experiments in appendices below -----
    #run_scaling() # group size effect in 6.4.4

    #run_varying_dynamic_topologies() # varying dynamic topologies in 6.4.5

    #run_mnemonic() #  intra-group aligment in 6.4.3, as well as all diversity plots

    #run_intergroup_alignment() # inter-group alignment in 6.4.3


