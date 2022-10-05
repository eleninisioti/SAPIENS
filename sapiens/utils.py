""" Contains miscalleneous utilities, for example creating directories.
"""
import os
import logging
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import numpy as np
import torch.nn as nn
import torch
import gym
import torch as th
from lib.stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
# ----- information about recipe books ----
class Net(nn.Module):
    def __init__(self, n_input_channels):
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(n_input_channels, 32, 8, 4)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)


        # First fully connected layer
        #self.fc1 = nn.Linear(3136, 512)
        self.fc1 = nn.Linear(3840, 512)
        # Second fully connected layer that outputs our 10 labels
        #self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)


        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.fc2(x)

        # Apply softmax to x
        #output = F.log_softmax(x, dim=1)
        return x

def custom_nn():
    class CustomCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            super(CustomCNN, self).__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            # Re-ordering will be done by pre-preprocessing or wrapper
            n_input_channels = observation_space.shape[0]
            self.my_nn = Net(n_input_channels)


        def forward(self, x):
            return self.my_nn(x / 255.0)

    policy_kwargs = dict(features_extractor_class=CustomCNN,
                     features_extractor_kwargs=dict(features_dim=512))
    return policy_kwargs

def indiv_mnemonic_metrics(agent):
    """ Computes mnemonic metrics for a single agent.

    The mnemonic metrics include diversity (number of unique tuples in replay buffer)
    """
    buffer = agent.replay_buffer
    buffer_observations = buffer.observations[:, 0, :].tolist()
    buffer_actions = buffer.actions[:, 0, 0].tolist()
    occurs = {}
    for idx, el in enumerate(buffer_observations):
        el.append(buffer_actions[idx])
        el_tuple = tuple(el)

        if el_tuple not in occurs.keys():
            num_occur = sum([1 for el1 in buffer_observations if el == el1])
            occurs[el_tuple] = num_occur

    diversity = len(occurs.keys())

    return diversity


def group_mnemonic_metrics(agents, measure_intergroup_alignment):
    """ Computes mnemonic metrics for a single agent.

    The group mnemonic metrics include diversity (number of unique tuples in the concatenated
    replay buffer) and intra-group alignment. We also save a dictionary with the number of occurences of each
    experience tuple in order to later compute inter-group alignment.
    """
    total_buffer_observations = []
    total_buffer_actions = []
    for agent in agents:
        buffer = agent.replay_buffer

        buffer_observations = buffer.observations[:, 0, :].tolist()
        buffer_actions = buffer.actions[:, 0, 0].tolist()

        total_buffer_actions.extend(buffer_actions)
        total_buffer_observations.extend(buffer_observations)

    occurs = {}
    for idx, el in enumerate(total_buffer_observations):
        el.append(total_buffer_actions[idx])
        el_tuple = tuple(el)

        if el_tuple not in occurs.keys():
            num_occur = sum([1 for el1 in total_buffer_observations if el == el1])
            occurs[el_tuple] = num_occur

    diversity = len(occurs.keys())

    total_values = []
    for agent in agents:
        buffer = agent.replay_buffer

        buffer_observations = buffer.observations[:, 0, :].tolist()
        buffer_actions = buffer.actions[:, 0, 0].tolist()

        model_values = []
        for idx, el in enumerate(buffer_observations):
            el.append(buffer_actions[idx])
            model_values.append(el)
        total_values.append(model_values)

    differences = {}
    for idx1, data1 in enumerate(total_values):
        for idx2, data2 in enumerate(total_values):
            if tuple([idx1, idx2]) not in differences.keys() and tuple([idx2, idx1]) not in differences.keys():
                differences[tuple([idx1, idx2])] = len([value for value in data1 if value not in data2])

    max_diffs = len(differences.values()) * len(model_values)
    alignment = 1 - (sum(differences.values()) / max_diffs)

    return diversity, alignment, occurs


def find_label(p, parameter):

    # find number of agents
    config_file = p + "/trial_0/config.yaml"
    with open(config_file, "rb") as c:
        config = yaml.safe_load(c)

    if parameter == "shape":
        if config["n_agents"] == 1:
            label = "single"
        else:
            label = shape_labels[config["shape"]]

    elif parameter == "tune_replay":
        if config["normalize"] and config["buffer_size"] == 500:
            label = "A"
        elif config["normalize"] and config["buffer_size"] == 5000:
            label = "B"
        elif not config["normalize"] and config["buffer_size"] == 50000:
            label = "C"
        elif not config["normalize"] and config["buffer_size"] == 5000:
            label = "D"

    elif parameter == "dynamic_structures":
        if config["shape"] == "dynamic_3phase":
            shapes  = ["I", "F", "R"]
        elif config["shape"] == "dynamic_3phase_v2":
            shapes  = ["I", "R", "F"]
        elif config["shape"] == "dynamic":
            shapes  = ["I", "F"]
        elif config["shape"] == "dynamic_ring":
            shapes = ["I",  "R"]
        label = ""
        for idx, el in enumerate(config["phase_periods"]):

            label += shapes[idx] + "=" + str(el) + ", "

        label = label[:-2]

    elif parameter == "shape_size":
        if config["n_agents"] == 1:
            label = "single"
        else:
            label = shape_labels[config["shape"]]
        size = config["n_agents"]
        label += "_N=" + str(size)

    elif parameter == "shape_explore":

        label = shape_labels[config["shape"]]
        explore = config["explore"]
        if explore == "none":
            label += ", $\epsilon=0$"
        elif explore == "high":
            label += ", $\epsilon=0.05$"
        elif explore == "low":
            label += ", $\epsilon=0.01$"

        if "intrinsic" in config.keys() and config["intrinsic"]:
            label += ", IM"

    elif parameter == "prioritized":
        if config[parameter]:
            label = "priorities"
        else:
            label = "no-priorities"
    else:

        size = config[parameter]
        label = "N=" + str(size)
    return label



