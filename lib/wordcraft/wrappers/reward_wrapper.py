from gym import Wrapper
from gym import spaces
import numpy as np
import re
import sys

from lib.wordcraft.wordcraft.recipe_book import Recipe

# no obs or act change


class RewardWrapper(Wrapper):
    def __init__(self, env, proportional=False):
        super().__init__(env)
        self.env = env
        self.proportional = proportional
        self.discovered = []

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.proportional:
            reward = self.proportional_reward(reward, action)

        return obs, reward, done, info

    def proportional_reward(self, reward, action):
        # Detect reward :
        if reward == 1:
            i = self.env.table_index[action]
            e = self.env.recipe_book.entities[i]
            recipe = Recipe(np.concatenate((self.env.selection, [e])))
            word = self.env.recipe_book.evaluate_recipe(recipe)
            if word is not None and word not in self.discovered:
                self.discovered.append(word)
                reward = len(self.discovered)
                return reward
        return reward


class PunisherWrapper(Wrapper):
    def __init__(self, env, proportional=False):
        super().__init__(env)
        self.env = env
        self.n_entities = len(self.env.recipe_book.entities)
        self.action_space = spaces.Discrete(
            self.n_entities
        )  # could be changed to only the elements than can be reached in the allowed steps (for now, too deep to be requires)
        # self.observation_space = spaces.Box(low=0, high=np.inf,shape=(self.n_entities * 4,),dtype=np.int64)
        self.proportional = proportional
        self.discovered = []

    # def reset(self): # has to be modified to return the modified observation space
    #     first_state = self.env.reset()
    #     return self.count_entities(first_state)

    def step(self, action):
        ## Action wrapping
        # find in table_index which one it is = the arg
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
            itemindex = n_available_objects  # make it chose the first unavailable slot
        # Dealing with the case where multiple instances of the chosen entity are available
        else:
            itemindex = np.min(itemindex)  # take the first instance of the object
        next_state, reward, done, info = self.env.step(itemindex)

        ## Reward Wrapping (let's see if we can do without)
        # if action > (n_available_objects -1): # -1 because indexes in tables begin at 0
        #     reward =-1 # impossible actions are strongly penalised

        # v2 growing proportional reward (to be made better)
        if self.proportional:
            reward = self.proportional_reward(reward, action)

        ## Observation wrapping (better results without)
        # obs = self.count_entities(next_state)
        obs = next_state
        return obs, reward, done, info

    def count_entities(self, next_state):
        # Observation is now a count of each existing entity
        ent_count = np.array(
            [
                np.count_nonzero(next_state["table_index"] == idx) != 0
                for idx in range(self.n_entities)
            ]
        )
        obs = np.concatenate(
            (
                ent_count,
                np.concatenate(next_state["selection_features"]),
                next_state["goal_features"],
            )
        )
        return obs

    def proportional_reward(self, reward, action):
        print(
            "OBSOLETE REWARD FUNCTION reward_wrapper.PunisherWrapper.proportional_reward"
        )
        # Detect reward :
        if reward == 1:
            i = self.env.table_index[action]
            e = self.env.recipe_book.entities[i]
            recipe = Recipe(np.concatenate((self.env.selection, [e])))
            word = self.env.recipe_book.evaluate_recipe(recipe)
            if word is not None and word not in self.discovered:
                self.discovered.append(word)
                reward = len(
                    self.discovered
                )  # OBSOLETE : does not take into account branches
                return reward
        return reward