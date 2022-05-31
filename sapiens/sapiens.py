import os
import time
import random

from lib.stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sapiens.es_dqn import DES_DQN
from lib.stable_baselines3.common.utils import set_random_seed
from sapiens.utils import group_mnemonic_metrics, indiv_mnemonic_metrics
from sapiens.topologies import init_topology, update_topology_periodic, update_topology_Boyd
from lib.wordcraft.utils.task_utils import recipe_book_info

class Sapiens:
    """
    Structuring multi-Agent toPology for Innovation through ExperieNce Sharing (SAPIENS)

    SAPIENS is a learning algorithm that uses a group of DQN agents whose share experiences from their replay buffers
     with their neighbors, based on a chosen topology.

    Params
    ------
    train_envs: list of gym environments
        for each agent, the training environment

    eval_envs: list of gym enviroments
        for each agent, the evaluation environment

    train_config: dict
        training configuration

    p_s: float
        probability of sharing experience at the end of an episode
    """

    def __init__(self, train_envs, eval_envs, n_agents, n_neighbors, n_subgroups, phase_periods, project_path,
                 shape, total_timesteps, buffer_size=5000, batch_size=128, num_neurons=64, num_layers=2,
                 visit_duration=10,
                 seed=0, gamma=0.9, L_s=1, learning_rate=0.0001, p_s=1, collect_metrics_period=500, policy="MlpPolicy"):
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.shape = shape
        self.n_agents = n_agents
        self.n_neighbors = n_neighbors
        self.project_path = project_path
        self.collect_metrics_period = collect_metrics_period
        self.seed = seed
        self.policy = policy
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.visit_duration = visit_duration
        self.L_s = L_s
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.total_timesteps = total_timesteps

        # config related to dynamic topologies
        self.phase_periods = [int(el) for el in phase_periods.split(",")]
        self.phases = ["no-sharing", "fully-connected"]
        self.p_s = p_s
        self.n_subgroups = n_subgroups
        self.phase_idx = 0
        self.timer_dynamic_boyd = 0

    def _setup_model(self):
        self.models = []
        self.times = []  # keep track of execution time
        self.rollout_times = []
        self.step_times = []
        self.visit = False
        self.sharing = True
        self.dynamic = False

        set_random_seed(self.seed)

        project_dirs = ["/models", "/tb_logs", "/plots", "/data"]
        for project_dir in project_dirs:
            new_dir = self.project_path + project_dir
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

    def create_callbacks(self, agent_idx):
        """ Defines callbacks for one of the group's agents.

        Parameters
        ----------
        agent_idx: int
            agent index
        """
        # 1) Every eval_freq timesteps, the model evaluation determines if the model has improved, and saves this model
        eval_callback = EvalCallback(
            self.eval_envs[agent_idx],
            log_path=self.project_path + "/tb_logs",
            best_model_save_path=self.project_path + "/models",
            eval_freq=100,
            deterministic=True,
            render=False,
        )
        # 2) a regular saving point (every save_freq timesteps) is also made to measure improvement throughout training
        c_callback = CheckpointCallback(
            save_freq=10000, save_path=self.project_path + "/models",
            name_prefix=f"agent_{agent_idx}"
        )
        callbacks = [eval_callback, c_callback]
        return callbacks

    def init_group(self):
        """ Initialize the ES-DQN agents of the group
        """
        if self.n_agents > 1:
            self.graph = init_topology(shape=self.shape, n_agents=self.n_agents,
                                       project_path=self.project_path,
                                       n_subgroups=self.n_subgroups, n_neighbors=self.n_neighbors)

        # NN architecture
        policy_kwargs = dict(net_arch=[self.num_neurons] * self.num_layers)

        # ----- for each agent: 1) check if we want to reload or train from scratch 3) initialize 4) update neighbors
        agents = []
        callbacks = []
        for i, env in enumerate(self.train_envs):
            # this value from the original DQN implementation
            learning_starts = (self.buffer_size // 100 * 5)

            # ----- exploration schedule -----
            exploration_fraction = 0
            exploration_initial_eps = 0.1
            exploration_final_eps = 0.05

            agent = DES_DQN(idx=i,
                            policy=self.policy,
                            env=self.train_envs[i],
                            verbose=True,
                            gamma=self.gamma,
                            buffer_size=self.buffer_size,
                            L_s=self.L_s,
                            learning_starts=learning_starts,
                            optimize_memory_usage=True,
                            tensorboard_log=self.project_path + "/tb_logs",
                            batch_size=self.batch_size,
                            policy_kwargs=policy_kwargs,
                            exploration_fraction=exploration_fraction,
                            exploration_initial_eps=exploration_initial_eps,
                            exploration_final_eps=exploration_final_eps,
                            train_freq=16,
                            learning_rate=self.learning_rate)

            self.init_training_step = 0

            # attach callbacks
            temp_callbacks = self.create_callbacks(i)
            _, callback = agent._setup_learn(self.total_timesteps, eval_env=self.eval_envs[i],
                                             callback=temp_callbacks)
            callbacks.append(callback)
            agent.num_timesteps = self.init_training_step

            if self.shape == "dynamic-Boyd":
                agent.visit_duration = self.visit_duration

            agents.append(agent)

        # now that all models have been created, attach them to their neighbors
        if self.n_agents > 1:
            for i, model in enumerate(agents):
                neighbor_idxs = list(self.graph.neighbors(i))
                model.neighbors = [agents[n] for n in neighbor_idxs]

        self.agents = agents
        self.callbacks = callbacks

    def learn(self):
        """ Train all agents in the group.

        At each episode, an agent collects experience, sends experience to all its neighbors with probability p_s and
        trains.
        """
        self.init_group()
        current_training_step = self.init_training_step
        total_timesteps = self.total_timesteps + self.init_training_step
        if current_training_step:
            # if model is loaded train from scratch
            learning_starts = current_training_step
        else:
            learning_starts = (self.buffer_size // 100 * 5)

        # ----- training phase -----
        stop_training = False

        for i, ENV in enumerate(self.train_envs):
            self.callbacks[i].on_training_start(locals(), globals())

        while (current_training_step < total_timesteps) and (not stop_training):

            start_time = time.time()
            rollouts = []
            for i, agent in enumerate(self.agents):

                # ------ log for mnemonic metrics -----
                if current_training_step % self.collect_metrics_period == 0 and self.train_config.study_diversity:
                    diversity, group_diversity, group_occurs = indiv_mnemonic_metrics(agent)
                    agent.diversities.append(diversity)

                    if i == 0:
                        group_diversity, occurs = group_mnemonic_metrics(self.agents)
                    agent.group_diversities.append(group_diversity)
                    agent.group_occurs.append(occurs)

                # collect experience
                rollout = agent.collect_rollouts(
                    agent.env,
                    train_freq=agent.train_freq,
                    action_noise=agent.action_noise,
                    callback=self.callbacks[i],
                    learning_starts=agent.learning_starts,
                    replay_buffer=agent.replay_buffer,
                    log_interval=100)
                rollouts.append(rollout)

                # share experience
                epsilon = random.uniform(0, 1)
                if self.sharing and current_training_step > learning_starts and epsilon < self.p_s:
                    agent.share()

                if rollout.continue_training is False:
                    stop_training = True
                    break

                # train
                if current_training_step > learning_starts:
                    gradient_steps = agent.gradient_steps if agent.gradient_steps > 0 else rollouts[
                        i].episode_timesteps
                    agent.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

            current_training_step += 1

            # ------- update dynamic topologies ------
            if self.shape == "dynamic-periodic":
                self.timer_dynamic += 1
                if self.timer_dynamic == self.phase_periods[self.phase_idx]:
                    self.timer_dynamic = 0
                    self.phase_idx += 1
                    self.phase_idx = (self.phase_idx % len(self.phases))
                    self.graph, self.agents = update_topology_periodic()

            if self.shape == "dynamic-Boyd":
                self.graph, self.visit, self.agents = update_topology_Boyd(agents=self.agents,
                                                                           graph=self.graph,
                                                                           migrate_rate=self.migrate_rate,
                                                                           visit=self.visit)
            # -----------------------------------------
        for c in self.callbacks:
            c.on_training_end()
