from gym import Wrapper
from gym import spaces
import numpy as np

# from reward_wrapper import RewardWrapper


class LiteWrapper(Wrapper):
    """
    works with lite environment, and one_hot encoding, to result in a smaller replay buffer, and allow deeper search
    """

    def __init__(self, env, proportional=False):
        super().__init__(env)
        self.env = env
        self.proportional = proportional
        self.innovation_modifier = 1
        self.hover_index = 0

        num_entities = len(self.env.recipe_book.entities)
        dspaces = {
            "goal_index": spaces.MultiDiscrete([num_entities]),
            "goal_features": spaces.Box(
                shape=self.env.goal_features.shape, low=-1.0, high=1.0
            ),
            "table_index": spaces.MultiDiscrete(
                self.env.max_table_size * [num_entities]
            ),
            "table_features": spaces.Box(
                shape=self.env.table_features.shape, low=-1.0, high=1.0
            ),
            "selection_index": spaces.MultiDiscrete(
                self.env.max_selection_size * [num_entities]
            ),
            "selection_features": spaces.Box(
                shape=self.env.selection_features.shape, low=-1.0, high=1.0
            ),
        }
        self.observation_space = spaces.Dict(dspaces)

    def reset(self):  # has to be modified to return the modified observation space
        first_state = self.env.reset()
        first_state["goal_features"] = self.idx2feature(first_state["goal_index"])
        first_state["table_features"] = self.idx2feature(first_state["table_index"])
        first_state["selection_features"] = self.idx2feature(
            first_state["selection_index"]
        )
        return first_state

    def step(self, action):
        obs, reward, done, info = self.unwrapped.step(action)
        obs["goal_features"] = self.idx2feature(obs["goal_index"])
        obs["table_features"] = self.idx2feature(obs["table_index"])
        obs["selection_features"] = self.idx2feature(obs["selection_index"])
        return obs, reward, done, info

    def idx2feature(self, table):
        features = []
        for index in table:
            feature = [0] * self.env.max_table_size
            feature[index] = 1
            features.append(feature)
        return features