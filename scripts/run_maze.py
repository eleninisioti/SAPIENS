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




def run_maze():
    """ Runs all experiments used in plots in the main paper.

    In particular, we run the five social network topologies on the three tasks for 10 agents. We do not keep track
    of mnemonic metrics (doing so requires a lot of additional memory).
    """

    top_dir = "/media/elena/LaCie/SAPIENS/projects/maze/test"
    shapes = [ "fully-connected", "no-sharing"]
    n_agents = 10

    gamma = 0.99
    buffer_size = 5000
    batch_size = 64
    num_neurons = 64
    num_layers = 2

    #tasks = {"single_path": 50000, "merging_paths": 500000, "bestoften_paths": 500000}
    tasks = { "maze-random-10x10-v0": 50, "maze-random-10x10-plus-v0": 50}


    for task, total_episodes in tasks.items():

        for shape in shapes:

            project_path = top_dir + "/task_" + task + "/shape_" + shape
            train_envs = []
            eval_envs = []
            for i in range(n_agents):

                train_env = gym.make(task, enable_render=False)
                eval_env = gym.make(task, enable_render=False)

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
                            n_trials=2)
            # train
            group.learn()

            # evaluate
            #evaluate_project(project_path, playground=task)





if __name__ == "__main__":

    # reproduce experiments analysed in the main paper
    run_maze()



