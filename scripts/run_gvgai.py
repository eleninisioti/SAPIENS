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
import lib.maze.gym_maze
import gym
import gym_gvgai

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


def run_gvgai():
    """ Runs all experiments used in plots in the main paper.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).
    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/gvgai/test_single"
    shapes = [  "no-sharing"]
    n_agents = 1

    gamma = 0.99
    buffer_size = 5000
    batch_size = 1024
    num_neurons = 64
    num_layers = 2

    #tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}
    tasks = { "decepticoins": 5000000}


    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = top_dir + "/task_" + task + "/shape_" + shape
            train_envs = []
            eval_envs = []
            for i in range(n_agents):
                game = "gvgai-decepticoins"
                level = "lvl0-v0"
                train_env = gym_gvgai.make(game + '-' + level, episode_len=200)
                #train_env = FlattenObservation(train_env)

                train_env.reset()
                train_envs.append(train_env)

                game = "gvgai-decepticoins"
                level = "lvl0-v0"
                eval_env = gym_gvgai.make(game + '-' + level, episode_len=200)

                #eval_env = FlattenObservation(eval_env)
                eval_env.reset()
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
                            n_trials=2)
            # train
            group.learn()

            # evaluate
            #evaluate_project(project_path, playground="task")





if __name__ == "__main__":

    # reproduce experiments analysed in the main paper
    run_gvgai()



