import sys
import os
from gym import Wrapper
from gym import spaces
from gym.spaces import MultiDiscrete, Box, Dict, Discrete
import numpy as np
import re
from lib.wordcraft.wordcraft.recipe_book import Recipe


class NoGoalWrapper(Wrapper):
    def __init__(self, env, proportional=True, from_base=False):
        """
        proportional remains for compatibility with the others wrappers, but isn't required
        TODO : make the env always start from base
        """
        super().__init__(env)
        self.env = env
        self.discovered = []
        self.env.max_table_size += 2
        num_entities = len(self.env.recipe_book.entities)
        dspaces = {
            "table_index": MultiDiscrete(self.env.max_table_size * [num_entities]),
            "table_features": Box(shape=self.table_features.shape, low=-1.0, high=1.0),
            "selection_index": MultiDiscrete(self.max_selection_size * [num_entities]),
            "selection_features": Box(
                shape=self.selection_features.shape, low=-1.0, high=1.0
            ),
        }
        self.observation_space = Dict(dspaces)
        self.action_space = Discrete(
            self.env.max_table_size
        )  # Actions correspond to choosing an entity in a table position
        self.reset()
        self.proportional = proportional

    def reset(self):  # has to be modified to return the modified observation space
        first_state = self.env.reset()
        # Check which branch the goal is from, & add base entities from the other branch
        _task = self.env.recipe_book.sample_task(depth=self.env.sample_depth)

        # TODO : Attention, n'est pas mélangé correctement
        while _task.goal[0] == self.env.task.goal[0]:
            _task = self.env.recipe_book.sample_task(depth=self.env.sample_depth)
        for e in _task.base_entities:
            # !! adding these entities without changing table length might cause rare cases where table becomes full at runtime
            result_i = self.env.recipe_book.entity2index[e]
            table_size = len(self.env.table)
            self.env.table.append(e)
            self.env.table_index[table_size] = result_i
            self.env.table_features[table_size, :] = self.env.feature_map.feature(e)

        self.discovered = []

        del first_state["goal_index"]
        del first_state["goal_features"]
        print(self.env.max_table_size)
        print([a for a in first_state])
        print([np.array(first_state[a]).shape for a in first_state])
        return first_state

    def step(self, action):
        reward = self.systematic_proportional_reward(action)
        next_state, _, done, info = self.env.step(action)

        del next_state["goal_index"]
        del next_state["goal_features"]

        obs = next_state
        return obs, reward, done, info

    def systematic_proportional_reward(self, action):
        # Detect reward :
        if len(self.env.selection) == self.env.max_selection_size - 1:
            # check depth of created entity
            i = self.env.table_index[action]
            e = self.env.recipe_book.entities[i]
            recipe = Recipe(np.concatenate((self.env.selection, [e])))
            word = self.env.recipe_book.evaluate_recipe(recipe)
            if word is not None and word not in self.discovered:
                # TODO : apply modifier depending on depth
                self.discovered.append(word)
                reward = 1
                return reward
        return 0