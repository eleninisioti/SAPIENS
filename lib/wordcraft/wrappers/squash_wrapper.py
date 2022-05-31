from gym import Wrapper
from gym.spaces import MultiDiscrete, Dict, Discrete
import numpy as np
import re
import sys


# no obs or act change


class SquashWrapper(Wrapper):
    def __init__(self, env, proportional=False):
        super().__init__(env)
        self.env = env
        self.n_entities = len(self.env.recipe_book.entities)
        self.action_space = Discrete(
            self.n_entities
        )  # could be changed to only the elements than can be reached in the allowed steps (for now, too deep to be requires)
        dspaces = {
            "table_features": MultiDiscrete(self.n_entities * [self.n_entities]),
            "selection_features": MultiDiscrete(self.n_entities * [self.n_entities]),
        }
        self.observation_space = Dict(dspaces)
        self.proportional = proportional
        self.discovered = []

    def reset(self):  # has to be modified to return the modified observation space
        first_state = self.env.reset()
        # print([key for key in first_state])
        first_state["table_features"] = np.sum(first_state["table_features"], axis=0)
        first_state["selection_features"] = np.sum(
            first_state["selection_features"], axis=0
        )
        del first_state["table_index"]
        del first_state["selection_index"]
        # print(len(first_state))
        return first_state

    def step(self, action):
        ## Action wrapping
        # find in table_index which one it is = the arg
        #action = 1
        itemindex = np.where(
            self.unwrapped._get_observation()["table_index"] == action
        )[
            0
        ]  # The first amongst chosen actions is taken
        # Dealing with the case where the agent choses an entity that is not available at this step
        n_available_objects = len(
            [a for a in self.unwrapped._get_observation()["table_index"] if a >= 0]
        )
        if len(itemindex) == 0:
            # print("empty pick")
            itemindex = min(
                n_available_objects, self.env.action_space.n
            )  # make it chose the first unavailable slot
        # Dealing with the case where multiple instances of the chosen entity are available
        else:
            itemindex = np.min(itemindex)  # take the first instance of the object
        next_state, reward, done, info = self.env.step(itemindex)

        ## Observation wrapping (better results without)
        next_state["table_features"] = np.sum(next_state["table_features"], axis=0)
        next_state["selection_features"] = np.sum(
            next_state["selection_features"], axis=0
        )
        del next_state["table_index"]
        del next_state["selection_index"]

        return next_state, reward, done, info