
import random
import copy

from lib.stable_baselines3.common.utils import get_linear_fn
from lib.stable_baselines3 import DQN
from lib.stable_baselines3.common.buffers import ReplayBuffer


class DES_DQN(DQN):
    """ Dynamic Experience-sharing DQN (ES-DQN) is an extension of DQN where an agent belongs to a group in which it can
    a) share experience tuples from its replay buffer with its neighbors b) visit another agent in the group.

    Default hyper-parameters are the same with the stable-baselines3 implementation of DQN (
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py)

    Params
    ------
    idx: int
        index of agent int the group

    L_s: int
        length of shared batch
    """

    def __init__(self,
                 policy,
                 env,
                 idx=0,
                 L_s: int = 128,
                 learning_rate=1e-4,
                 buffer_size: int = 1000000,
                 learning_starts: int = 50000,
                 batch_size=32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 train_freq=4,  # [4,"episode"]
                 gradient_steps: int = 1,
                 optimize_memory_usage: bool = False,
                 target_update_interval: int = 10000,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10,
                 tensorboard_log=None,
                 create_eval_env: bool = False,
                 policy_kwargs=None,
                 verbose: int = 0,
                 seed=None,
                 device="cpu",
                 _init_setup_model: bool = True,
                 ):
        self.idx = idx
        self.neighbors = []
        self.visiting = False
        self.under_visit = False
        self.sent_messages = []
        self.L_s = L_s

        self.diversities = []
        self.intragroup_alignments = []
        self.group_diversities = []
        self.group_occurs = []

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )

        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction)

        self.beta = 1

    def share(self):
        """ Share experience with neighbors.
        """
        for neighbor in self.neighbors:
            # sample experience
            send_exp = self.replay_buffer.sample(self.L_s)

            for i in range(self.L_s):
                neighbor.replay_buffer.add(
                    obs=send_exp.observations[i].detach().cpu().numpy(),
                    next_obs=send_exp.next_observations[i].detach().cpu().numpy(),
                    action=send_exp.actions[i].detach().cpu().numpy(),
                    reward=send_exp.rewards[i].detach().cpu().numpy(),
                    done=send_exp.dones[i].detach().cpu().numpy()
                )

    def visit(self, agents):
        """ Visit another sub-group chosen at random.

        Params
        -----
        agents: list of ES-DQN
            the group of agents
        """
        self.visiting = False
        self.old_neighbs = copy.copy(self.neighbors)
        self.neighbors = []

        random.shuffle(agents)
        for agent_idx, agent in enumerate(agents):
            if agent not in self.old_neighbs and agent.idx != self.idx  and not self.visiting:
                potential_group = copy.copy(agent.neighbors)
                potential_group.append(agent)

                # update immigrant with new neighbosrs
                for new_neighbor in potential_group:
                    self.neighbors.append(new_neighbor)

                for idx, agent in enumerate(agents):
                    if agent.idx == self.idx:
                        agents[idx] = self

                # update new neighbors with immigrant
                for neighbor in potential_group:
                    neighbor.neighbors.append(self)

                    for idx, agent in enumerate(agents):
                        if agent.idx == neighbor.idx:
                            agents[idx] = neighbor

                # inform previous neighbors imigrant has left
                for neighbor in self.old_neighbs:
                    for temp in neighbor.neighbors:
                        if temp.idx == self.idx:
                            neighbor.neighbors.remove(temp)

                    for idx, agent in enumerate(agents):
                        if agent.idx == neighbor.idx:
                            agents[idx] = neighbor

                self.visiting = True
                self.visit_count = 0
                # just debug
                nidxs= []
                for agent in agents:
                    if agent.idx == self.idx:
                        for nghb in agent.neighbors:
                            nidxs.append(nghb.idx)

                print("I am", self.idx, 'and my new neighbors are', nidxs)
        return agents

    def visit_return(self, agents):
        """ The agent returns from a visit.

        Params
        -----
        agents: list of ES-DQN
            the group of agents
        """
        if self.visiting and self.visit_count == self.visit_duration:
            end_visit = True
            self.visiting = False

            current_neighbs = copy.copy(self.neighbors)
            self.neighbors = []

            # inform past neighbors immigrant is coming back
            for neighbor in self.old_neighbs:
                neighbor.neighbors.append(self)
                self.neighbors.append(neighbor)

                for idx, agent in enumerate(agents):
                    if agent.idx == neighbor.idx:
                        agents[idx] = neighbor
            self.old_neighbs = []

            for idx, agent in enumerate(agents):
                if agent.idx == self.idx:
                    agents[idx] = self

            # inform neighbors immigrant has left
            for neighbor in current_neighbs:
                for temp in neighbor.neighbors:
                    if temp.idx == self.idx:
                        neighbor.neighbors.remove(temp)

                for idx, agent in enumerate(agents):
                    if agent.idx == neighbor.idx:
                        agents[idx] = neighbor

            # just debug
            nidxs = []
            for agent in agents:
                if agent.idx == self.idx:
                    for nghb in agent.neighbors:
                        nidxs.append(nghb.idx)

            print("I am", self.idx, 'and I am returning to', nidxs)

        elif self.visiting:
            end_visit = False
            self.visit_count += 1

        else:
            end_visit = False

        return agents, end_visit
