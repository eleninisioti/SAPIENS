import sys
sys.path.append(".")
import gym
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
from lib.wordcraft.wordcraft.env_nogoal import WordCraftEnvNoGoal
import copy
from lib.stable_baselines3.common.env_util import make_vec_env
from lib.wordcraft.utils.task_utils import recipe_book_info
from lib.stable_baselines3 import A2C


def wrap(env):
    """
    Wraps the environment with the required wrappers
    """
    env = SquashWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    return env

def build_envs(config):
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

if __name__ == "__main__":

    top_dir = "/media/elena/LaCie/SAPIENS/projects/apex/test"
    task = "single_path"
    project_path = top_dir + "/" + task
    env_config = { 
        "log_path":project_path + "/tb_logs",
        "data_path" :  recipe_book_info[task]["path"], 
        "seed": None,
        "recipe_book_path":None,
        "feature_type":"one_hot",
        "shuffle_features":False,
        "random_feature_size":300,
        "max_depth":8,
        "split":"by_recipe",
        "train_ratio":1.0,
        "num_distractors":0,
        "uniform_distractors":False,
        "max_mix_steps":8,
        "subgoal_rewards":True,
        "proportional":True}
    
    env = build_envs(env_config)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_cartpole")
