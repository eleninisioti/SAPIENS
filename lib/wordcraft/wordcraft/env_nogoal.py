import os
from enum import IntEnum

import numpy as np
import gym
from gym.utils import seeding
# import re

# from lib.wordcraft.utils import seed as utils_seed
from lib.wordcraft.utils.word2feature import FeatureMap
from lib.wordcraft.wordcraft.recipe_book import Recipe, RecipeBook
# from gym.spaces import MultiDiscrete, Dict, Discrete

NO_RECIPE_PENALTY = 0
IRRELEVANT_RECIPE_PENALTY = 0
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0


class WordCraftEnvNoGoal(gym.Env):
    """
    Simple text-only RL environment for crafting multi-step recipes.

    At a high level, the state consists of a goal, the inventory, and the current selection.
    """

    def __init__(self,env_config):
        super().__init__()

        self.eval_mode = False

        if env_config["seed"] is None:
            env_config["seed"] = int.from_bytes(os.urandom(4), byteorder="little")
        self.set_seed(env_config["seed"])
        # utils_seed(seed)

        if env_config["recipe_book_path"] is not None:
            self.recipe_book = RecipeBook.load(env_config["recipe_book_path"])
            self.recipe_book.set_seed(env_config["seed"])
            env_config["max_depth"] = self.recipe_book.max_depth
        else:
            self.recipe_book = RecipeBook(
                data_path=env_config["data_path"],
                max_depth=env_config["max_depth"],
                split=env_config["split"],
                train_ratio=env_config["train_ratio"],
                seed=env_config["seed"],
            )

        self.feature_map = FeatureMap(
            words=self.recipe_book.entities,
            feature_type=env_config["feature_type"],
            random_feature_size=env_config["random_feature_size"],
            shuffle=env_config["shuffle_features"],
            seed=env_config["seed"],
        )

        self.max_selection_size = self.recipe_book.max_recipe_size
        self.max_mix_steps = max(env_config["max_mix_steps"] or env_config["max_depth"], env_config["max_depth"])
        self.max_steps = self.max_selection_size * self.max_mix_steps

        self.sample_depth = env_config["max_depth"]

        self.subgoal_rewards = env_config["subgoal_rewards"]
        self.max_depth = env_config["max_depth"]
        self.num_distractors = env_config["num_distractors"]
        self.uniform_distractors = env_config["uniform_distractors"]
        # self.distractors = []
        self.distractors = np.random.choice(
            self.recipe_book.distractors, self.num_distractors
        )
        self.orig_table = [
            entity
            for entity in self.recipe_book.entity2level
            if self.recipe_book.entity2level[entity] == 0
            and entity not in self.recipe_book.distractors
        ]
        self.max_table_size = (
            2 ** env_config["max_depth"] + env_config["num_distractors"] + self.max_mix_steps + 4
        )  # + 4 is to go from 2 base elements to 6 (3 per branch)

        self.task = None

        self.goal_features = np.zeros(self.feature_map.feature_dim)

        self._reset_table()
        self._reset_selection()
        self._reset_history()

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        num_entities = len(self.recipe_book.entities)
        # dspaces = {
        #     "table_index": gym.spaces.MultiDiscrete(
        #         self.max_table_size * [num_entities]
        #     ),
        #     "table_features": gym.spaces.Box(
        #         shape=self.table_features.shape, low=-1.0, high=1.0
        #     ),
        #     "selection_index": gym.spaces.MultiDiscrete(
        #         self.max_selection_size * [num_entities]
        #     ),
        #     "selection_features": gym.spaces.Box(
        #         shape=self.selection_features.shape, low=-1.0, high=1.0
        #     ),
        # }
        # self.observation_space = gym.spaces.Dict(dspaces)
        self.action_space = gym.spaces.Discrete(num_entities)  # Actions correspond to choosing an entity in a table position

        ### inserting squash wrapper here
        dspaces = {
            "table_features": gym.spaces.MultiDiscrete(num_entities * [num_entities]),
            "selection_features": gym.spaces.MultiDiscrete(num_entities * [num_entities]),
        }
        self.initial_observation_space = gym.spaces.Dict(dspaces)
        self.proportional = env_config["proportional"]
        self.discovered = []

        ### inserting flatten observations wrapper here
        self.observation_space = gym.spaces.flatten_space(self.initial_observation_space)        
        self.reset()


    def reset(self):
        self.discovered = []
        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        self._reset_selection()
        self._reset_table()
        self._reset_history()

        ### inserting squash wrapper here
        original_first_state = self._get_observation()
        first_state = self.smaller_obs(original_first_state)
        return gym.spaces.flatten(self.initial_observation_space, first_state)

    def eval(self, split="test"):
        self.eval_mode = True
        self.recipe_book.test_mode = split == "test"

    def train(self):
        self.eval_mode = False
        self.recipe_book.test_mode = False

    def set_seed(self, seed):
        self.np_random, self.seed = seeding.np_random(seed)

    def sample_depth(self, depth):
        self.sample_depth = depth

    def __max_table_size_for_depth(self, depth):
        return 2 ** depth - 1

    def _reset_table(self):

        self.distractors = np.random.choice(
            self.recipe_book.distractors, self.num_distractors
        )

        self.table = list(
            np.concatenate(
                (
                    self.orig_table,
                    self.distractors,
                )
            )
        )
        self.np_random.shuffle(self.table)
        self.table_index = -np.ones(self.max_table_size, dtype=int)
        self.table_features = np.zeros(
            (self.max_table_size, self.feature_map.feature_dim)
        )

        num_start_items = len(self.table)
        self.table_index[:num_start_items] = np.array(
            [self.recipe_book.entity2index[e] for e in self.table], dtype=int
        )
        self.table_features[:num_start_items, :] = np.array(
            [self.feature_map.feature(e) for e in self.table]
        )

    def _reset_selection(self):
        self.selection = []
        self.selection_index = -np.ones(self.max_selection_size, dtype=int)
        self.selection_features = np.zeros(
            (self.max_selection_size, self.feature_map.feature_dim)
        )

    def _reset_history(self):
        self.subgoal_history = set()

    def _get_observation(self):
        """
        Note, includes indices for each inventory and selection item,
        since torchbeast stores actions in a shared_memory tensor shared among actor processes
        """
        return {
            "table_index": self.table_index,
            "table_features": self.table_features,
            "selection_index": self.selection_index,
            "selection_features": self.selection_features,
        }
        

    def unsquashed_step(self, action):
        reward = 0
        if self.done:  # no-op if env is done
            return self._get_observation(), reward, self.done, {}

        # Handle invalid actions
        invalid_action = not (0 <= action < self.max_table_size)
        if invalid_action:
            self.episode_step += 1
            if self.episode_step >= self.max_steps:
                self.done = True

        i = self.table_index[action]
        e = self.recipe_book.entities[i]

        selection_size = len(self.selection)
        if selection_size < self.max_selection_size:
            # Update selection
            self.selection.append(e)
            self.selection_index[selection_size] = i
            self.selection_features[selection_size, :] = self.feature_map.feature(e)
            selection_size = len(self.selection)

        if selection_size == self.max_selection_size:
            self.episode_mix_steps += 1

            # Evaluate selection
            recipe = Recipe(self.selection)
            result = self.recipe_book.evaluate_recipe(recipe)

            if result != None and result not in self.discovered:
                reward = self.recipe_book.entity2level[result]
                self.discovered.append(result)

            self.episode_reward += reward

            if result:
                result_i = self.recipe_book.entity2index[result]
                table_size = len(self.table)
                self.table.append(result)
                self.table_index[table_size] = result_i
                self.table_features[table_size, :] = self.feature_map.feature(result)

            # Clear selection
            self._reset_selection()

        self.episode_step += 1
        if (
            self.episode_mix_steps >= self.max_mix_steps
            or self.episode_step >= self.max_steps
        ):
            self.done = True

        obs = self._get_observation()

        return obs, reward, self.done, {}

    def step(self, action):
        ### inserting squash wrapper here
        # we use table position as actions, and here translate it to actual recipe_book indexes
        # if multiple actions are selected, only the first is used
        itemindex = np.where(self.table_index == action)[0]

        # Dealing with the case where the agent choses an entity that is not available at this step
        # by making it chose the first unavailable slot
        n_available_objects = len([a for a in self.table_index if a >= 0])
        if len(itemindex) == 0:
            itemindex = min(n_available_objects, self.action_space.n)  
        
        # Dealing with the case where multiple instances of the chosen entity are available
        else:
            itemindex = np.min(itemindex)  # take the first instance of the object

        # Running the original environment step
        orig_next_obs, reward, done, info = self.unsquashed_step(itemindex)
        next_obs = self.smaller_obs(orig_next_obs)

        # flatten observation
        obs = gym.spaces.flatten(self.initial_observation_space, next_obs)

        return obs, reward, done, info
    def smaller_obs(self, obs):
        small_obs = {}
        ## Observation wrapping
        small_obs["table_features"] = np.sum(obs["table_features"], axis=0).astype(int)
        small_obs["selection_features"] = np.sum(
            obs["selection_features"], axis=0
        ).astype(int)
        return small_obs

    def _display_ascii(self, mode="human"):
        """
        Render the env state as ascii:

        Combine the ingredients to make *torch*

        -------------------------------------------------------
        1:fire, 2:wind, 3:sand, 4:star, 5:wood, 6:stick, 7:coal
        -------------------------------------------------------

        (on hand): stick

        Subgoal rewards: 0
        """
        goal_str = f"Combine the ingredients to make *{self.task.goal}*"
        if mode == "human":
            table_str = f"{', '.join([f'{i+1}:{e}' for i, e in enumerate(self.table)])}"
        else:
            table_str = f"{', '.join(self.table)}"
        selection_str = f"(on hand): {', '.join(self.selection)}"
        hr = "".join(["-"] * 50)

        # output = f'\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\nSubgoal rewards: {self.episode_reward}\n'
        output = f"\n{goal_str}\n\n{hr}\n{table_str}\n{hr}\n\n{selection_str}\n\n"

        print(output)

    def render(self, mode="human"):
        self._display_ascii(mode)


gym.envs.registration.register(
    id="wordcraft-multistep-no-goal-v0",
    entry_point=f"{__name__}:WordCraftEnvNoGoal",
)
