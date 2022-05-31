from gym import Wrapper
from gym import spaces
import numpy as np
import re
class EmbodiedWrapper(Wrapper):
    def __init__(self, env,proportional=False) :
        super().__init__(env)
        self.env = env
        self.n_entities = len(self.env.recipe_book.entities)
        self.action_space = spaces.Discrete(3) # missing one_hot
        self.proportional = proportional
        self.hover_index = 0


    def step(self, action):
        ## Action wrapping            
        if action == 0:
            itemindex =np.where(self.unwrapped._get_observation()['table_index']==self.hover_index)[0]
            n_available_objects = len([a for a in self.unwrapped._get_observation()['table_index'] if a >= 0])
            if len(itemindex) == 0:
                itemindex = n_available_objects # make it chose the first unavailable slot
            # Dealing with the case where multiple instances of the chosen entity are available
            else:
                itemindex = np.min(itemindex)
            obs, reward, done, info = self.env.step(itemindex)
        else:
            if action ==1 and self.hover_index < self.n_entities:
                self.hover_index += 1
            elif action ==2 and self.hover_index > 0:
                self.hover_index -= 1
            obs = self.env._get_observation() # use previous observation
            reward = 0
            done = False
            info = {}
        
        # v2 growing proportional reward (to be made better)
        if self.proportional:
            reward = self.proportional_reward(reward, action)
        return obs, reward, done, info

    def proportional_reward(self, reward, action):
        # Detect reward : 
        if reward == 1 :
            # check depth
            word = self.env.recipe_book.entities[action]
            # apply modifier depending on depth
            reward *= 1.1*[int(s) for s in re.findall(r'\d+',word)][0]
        return reward