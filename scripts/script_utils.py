""" Contains various utilities useful for scripts.
"""
from lib.stable_baselines3.common.env_util import make_vec_env
import os


metric_labels = {"norm_reward": "Reward, $R_t$",
                 "diversity": "Diversity, $D_t$",
                 "level": "Level, $L_t$",
                 "group_diversity": "Group divesrity, $D^{\\mathcal{G}}_t$",
                 "intragroup_alignment": "Intra-group alignment, $A^{\mathcal{G}}_t$",
                 "conformity": "Conformity, $C_t$",
                 "conformities": "Conformity, $C_t$",
                 "volatility": "Average Volatility, $\\bar{V}_t$",
                 "volatilities": "Average Volatility, $\\bar{V}_t$",
                 }

metric_labels_avg = {"norm_reward": "Average Reward, $R^+_t$",
                     "diversity": "Average Diversity, $\\bar{D}_t$",
                     "level": "Average Level, $\\bar{L}_t$",
                     "conformity": "Conformity, $C_t$",
                     "volatility": "Average Volatility, $\\bar{V}_t$",
                     "volatilities": "Average Volatility, $\\bar{V}_t$"
                     }

metric_labels_max = {"norm_reward": "Maximum Reward, $R^*_t$",
                     "diversity": "Maximum Diversity, $\\hat{D}_t$",
                     "level": "Maximum Level, $\\hat{L}_t$",
                     "volatility": "Maximum Volatility, $\\hat{V}_t$",
                     "conformity": "Conformity, $C_t$",

                     "volatilities": "Maximum Volatility, $\\hat{V}_t$"

                     }



def find_ntrials(top_dir):
    trial_dirs = [os.path.join(top_dir, o) for o in os.listdir(top_dir) if "trial" in o]
    return len(trial_dirs)


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
