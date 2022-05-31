""" Main script for running an experiment
"""

# ---- generic imports -----
import gym
import sys
import os
import time
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


sys.path.insert(0, os.getcwd())
from gym.wrappers import Monitor, FlattenObservation
from lib.wordcraft.wrappers.chooser_wrapper import ChooserWrapper
from lib.wordcraft.wrappers.squash_wrapper import SquashWrapper
from lib.wordcraft.wrappers.depth_insentive_wrapper import NoGoalWrapper
from lib.wordcraft.wordcraft.env_nogoal import WordCraftEnvNoGoal
import yaml
import datetime
import torch

# ----- project-specific imports -----
from source.flags import get_cmd_args
from source.centralized_trainer import CentralizedTrainer
from source.help_funcs import plot_exectime, recipe_book_info


# useful for accessing dictionary elements with dot instead of ""
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def prepare_log_paths(params):
    """
    To avoid losing data, this function checks that a file exists for saving,
    And will create a new one if it is missing or if using it would cause overwriting a pre-existing file

    //!\\ This function is only a safeguard, logging and model-saving paths should be set to existing empty folders
    indeed, this function might not consistently number all experiments in the same manner

    """
    if not os.path.exists(params.log_path):
        os.makedirs(params.log_path, exist_ok=True)

    model_path = f"{params.log_path}/models/{params.exp_name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)


    log_path = f"{params.log_path}/logs/{params.exp_name}"


    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    return model_path, log_path

def build_envs(max_depth,
               max_mix_steps,
               num_distractors,
               subgoal_rewards=False,
               proportional=False,
               wrapper=None,
               env_name="wordcraft-multistep-no-goal-v0",
               recipe_path="",
               monitor_training=False,
               log_path=""
               ):
    """
    Environment generation tool

    Parameters
    ----------
    id: int
        unique id of agent
    max_depth: int
        maximum depth where the goal can be set in the goal-based version of the environment
    max_mix_steps: int
        maximum allowed number of mixing steps an agent can make. If it is lower than max_depth,
        max_depth is used instead to ensure the agent can reach its goal
    num_distractors: int
        number of distractors initially available to the agent
    subgoal_rewards: bool, default = False
        (obsolete) In the goal-based version of the environment, whether finding an intermediary entity that
        will lead to the goal is rewarded
    proportional: bool, default = False
        (obsolete) if the intermediate reward is proportional to the level of innovation in the goal_based
        version of the environment
    wrapper, default = None
        wrapper to be applied to the environment affecting observation and action
    env_name : string,default ="wordcraft-multistep-no-goal-v0",
        name of the gym environment
        to be changed if a variation of the environment is coded and used
    recipe_path : string, default = "../wordcraft/datasets/2br_30depth_recipes.json",
        path to the recipe book used to generate the environment
    monitor_training=False,
        activating the monitor will add a wrapper that will write additional training statistics
        to tensorboard
    """
    # use gym to generate the environment with the required parameters
    ENV = gym.make(
        env_name,
        max_depth=max_depth,
        max_mix_steps=max_mix_steps,
        num_distractors=num_distractors,
        subgoal_rewards=subgoal_rewards,
        data_path=recipe_path,
        feature_type="one_hot"
    )

    # apply wrappers
    if wrapper is not None:
        ENV = wrapper(ENV, proportional=proportional)
    ENV = FlattenObservation(ENV)

    if monitor_training:
        ENV = Monitor(ENV, directory=log_path, force=True)
    ENV.reset()

    # generate the corresponding evaluation environment, which is the same as the previous one
    # TODO : remove code dulication
    eval_env = gym.make(
        env_name,
        max_depth=max_depth,
        max_mix_steps=max_mix_steps,
        num_distractors=num_distractors,
        subgoal_rewards=subgoal_rewards,
        data_path=recipe_path,
        feature_type="one_hot",
    )
    # add the same wrappers for the testing environment as in the training environment
    if wrapper is not None:
        eval_env = wrapper(eval_env, proportional=proportional)
    eval_env = FlattenObservation(eval_env)
    if monitor_training:
        eval_env = Monitor(eval_env, directory=log_path)
    eval_env.reset()

    return ENV, eval_env


def run(params):
    """ Run an experiment for a single configuration, determined by params
    """
    # save config for reproducibility
    if not len(params.seed):
        seed = datetime.datetime.now().timestamp()
    else:
        seed = params.seed
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    params.seed = seed

    #params.seed = ""

    # ----- define reward structure -----
    if params.reward == 0:
        subgoal_rewards = False
        proportional = False
    elif params.reward == 1:
        subgoal_rewards = True
        proportional = False
    else:
        subgoal_rewards = True
        proportional = True

    # prepare logging and saving
    model_path, log_path = prepare_log_paths(params)

    # ----- setup training & evaluation environment for each agent -----
    wrappers = [
        None,  # !! TODO : RewardWrapper should be here if proportional reward is chosen
        SquashWrapper,
        ChooserWrapper,
        NoGoalWrapper,
    ]

    train_envs = []
    eval_envs = []
    for i in range(params.n_agents):
        if params.playground == "wordcraft":
            train_env, eval_env = build_envs(
                params.max_depth,
                params.max_mix_steps,
                params.num_distractors,
                subgoal_rewards,
                proportional,
                wrappers[params.wrapper_idx],
                monitor_training=True,
                recipe_path=recipe_book_info[params.recipe_path]["path"],
                log_path=log_path
            )
        elif params.playground == "pong":
            train_env = gym.make("Pong-v4")
            train_env = FlattenObservation(train_env)
            train_env = Monitor(train_env, directory=log_path, force=True)
            train_env.reset()

            eval_env = gym.make("Pong-v4")
            eval_env = FlattenObservation(eval_env)
            eval_env = Monitor(eval_env, directory=log_path, force=True)
            eval_env.reset()

        elif params.playground == "deceptive":
            game = "decepticoins"
            level = "lv0-v0"
            train_env = gym_gvgai.make(game + '-' + level)
            train_env = Monitor(train_env, directory=log_path, force=True)
            train_env.reset()

            game = "decepticoins"
            level = "lv0-v0"
            env = gym_gvgai.make(game + '-' + level)
            eval_env = FlattenObservation(eval_env)
            eval_env = Monitor(eval_env, directory=log_path, force=True)
            eval_env.reset()

        train_envs.append(train_env)
        eval_envs.append(eval_env)

    log_config = dotdict({"best_model_save_path": model_path, "logpath": log_path})
    trainer = CentralizedTrainer(train_envs=train_envs, eval_envs=eval_envs, train_config=params,
                                 log_config=log_config)
    temp = type(params)
    print(temp)
    if not isinstance(params, dict):
        params = vars(params)
        with open(model_path + "/config.yaml", "w") as f:
            params_dic = {}
            for value, key in params.items():
                params_dic[value] = key
            yaml.dump(params_dic, f)
    else:

        with open(model_path + "/config.yaml", "w") as f:
            params_dic = {}
            for value, key in params.items():
                params_dic[value] = key
            yaml.dump(params_dic, f)
    # train models
    start = time.time()
    trainer.learn()
    print("Execution finished. Total time =" + str(time.time() - start))

    plot_exectime(project=model_path, times=trainer.times)
    plot_exectime(project=model_path, times=trainer.rollout_times, type="rollout")
    plot_exectime(project=model_path, times=trainer.step_times,     type="step")




if __name__ == "__main__":
    params = get_cmd_args()
    run(params)
