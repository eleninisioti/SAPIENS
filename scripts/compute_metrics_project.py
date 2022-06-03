from lib.wordcraft.utils.task_utils import recipe_book_info
import yaml
import pickle
import pandas as pd
import numpy as np


def measure_volatility(trajectories, n_trials, n_agents):
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
                current_traj = agent_traj.loc[agent_traj["train_step"] == step]
                prev_traj = agent_traj.loc[agent_traj["train_step"] == steps[idx]]
                current_traj = list(current_traj["trajectory"])[0]
                prev_traj = list(prev_traj["trajectory"])[0]
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
    """ Measures conformity as the variance in the last visited state in the trajectories of agents.

            Returns:
                the mean value across timesteps
                the value at each timestep
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
            df_conformity["group_conformity"].append(conformity)
            df_conformity["trial"].append(trial)

        conformity.append(np.mean(trial_conformities))

    df_conformity = pd.DataFrame.from_dict(df_conformity)

    return conformity, df_conformity


def compute_performance_metrics(eval_info, n_trials, n_agents):
    metrics = {"time_to_first_success": [], "time_to_all_successes": [], "spread_time": [],
               "group_success": [], "avg_reward_conv": [], "max_reward_conv":[]}

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
            first_step_one = None
            failed_trial = 1

        # detect when all agents found the optimal solution
        if len(first_steps) == n_agents:
            first_step_all = max(first_steps)
            time_to_spread = first_step_all - first_step_one
        else:
            first_step_all = None
            time_to_spread = None

        metrics["time_to_all_successes"].append(first_step_all)
        metrics["time_to_first_success"].append(first_step_one)
        metrics["group_success"].append(1-failed_trial)
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
    volatility, df_volatility = measure_volatility(eval_info, n_trials, n_agents)
    conformity, df_conformity = measure_conformity(eval_info, n_trials, n_agents)

    metrics = {"volatility": volatility,  "conformity": conformity}

    return metrics, df_volatility, df_conformity

def compute_metrics(project):
    config = yaml.safe_load(open( project + "/config.yaml", "r"))
    n_trials = config["n_trials"]
    n_agents = config["n_agents"]

    with open(project + "/data/eval_info.pkl", "rb") as f:
        eval_info = pickle.load( f)

        performance_metrics = compute_performance_metrics(eval_info, n_trials, n_agents)
        behavioral_metrics, df_volatility, df_conformity = compute_behavioral_metrics(eval_info, n_trials, n_agents)


        metrics = {**performance_metrics, **behavioral_metrics}

        # pkl file contains values in all trials
        save_file = project+ "/data/pop_metrics.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(metrics, f)

        # yaml file contains average over trials
        metrics_mean = metrics.mean(axis=0)
        metrics_var = metrics.var(axis=0)

        metrics_stat = {**metrics_mean , **metrics_var}

        save_file = project+ "/data/pop_metrics.yaml"
        with open(save_file, "w") as f:
            yaml.dump(metrics_stat, f)

    return df_volatility, df_conformity


