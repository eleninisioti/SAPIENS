""" This scripts contains various tools related to graphs that represent social network topologies.
"""

import networkx as nx
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


def init_topology(shape, n_agents, project_path, n_subgroups=1, n_neighbors=1):
    """ Initializes the social network topology.

    Params
    ------
    shape: str
        the social network topology

    n_agents: int
        number of agents

    project_path: str
        project directory for saving

    n_subgroups: int
        number of sub-groups. Only useful for the dynamic-Boyd topology

    n_neighbors: int
        number of neighbors. Only useful for the small-world topology

    """

    if shape == "ring":
        graph = nx.cycle_graph(n_agents)

    elif shape == "fully-connected":
        graph = nx.complete_graph(n_agents)

    elif shape == "dynamic-Boyd":
        # divides group into fully-connected sub-groups
        subgroup_size = int(n_agents / n_subgroups)

        for subgroup in range(n_subgroups):
            subgraph = nx.complete_graph(subgroup_size)
            mapping = {}
            for el in range(subgroup_size):
                mapping[el] = subgroup * subgroup_size + el
            subgraph = nx.relabel_nodes(subgraph, mapping)
            if not subgroup:
                graph = subgraph
            else:
                graph = nx.compose(graph, subgraph)

    elif shape == "dynamic-periodic":
        graph = nx.empty_graph(n_agents)  # no connections between agents

    elif shape == "no-sharing":
        graph = nx.empty_graph(n_agents)

    elif shape == "small-world":  # small-world
        graph = nx.watts_strogatz_graph(n=n_agents, k=n_neighbors, p=0.2)

    # ----- save topology ------
    print("saving under", project_path)
    nx.draw(graph, node_color='green', node_size=150)
    plt.savefig(project_path + "/plots/network.png")
    nx.write_gexf(graph, path=project_path + "/plots/graph.gexf")
    plt.clf()
    return graph


def update_topology_periodic(agents, phases, phase_idx, graph):
    """ Updates the dynamic-periodic topology.

    Params
    ------
    agents: list of ES-DQN
        the group of agents

    phases: list of str
        the phases of dynamic-periodic

    phase_idx: int
        index of current phase. Only useful for dynamic-periodic

    graph: networkx.Graph
        the graph modeling the social network topology
    """
    n_agents = len(agents)
    if phases[phase_idx] == "no-sharing":
        graph = nx.empty_graph(n_agents)
    elif phases[phase_idx] == "fully":
        graph = nx.complete_graph(n_agents)
    for i, agent in enumerate(agents):
        agent.neighbors = [agents[n] for n in list(graph.neighbors(i))]

    return graph, agents


def update_topology_Boyd(agents, graph, migrate_rate, visiting, project_path, episode):
    """ Updates the dynamic-Boyd topology

    Params
    ------

    agents: list of ES-DQN
        the group of agents

    graph: networkx.Graph
        the graph modeling the social network topology

    migrate_rate: float
        probability of visit. Only useful for dynamic-Boyd

    visit: bool
        indicates whether a visit is currently taking place
    """
    epsilon = random.uniform(0, 1)
    if epsilon < migrate_rate and not visiting:
        # a random agent visits a random sub-group
        keep_agents = copy.copy(agents)

        immigrant = random.choice(keep_agents)
        agents = immigrant.visit(keep_agents)

        while not len(agents):
            immigrant = random.choice(keep_agents)
            agents = immigrant.visit(keep_agents)

        visiting = True

    if visiting:
        # check if an agent needs to return from a visit
        for agent in agents:
            if agent.visiting:
                agents, end_visit = agent.visit_return(agents)

                if end_visit:
                    visiting = False

    # order agents (it matters because the callbacks are ordered)
    ordered_agents = []
    for counter in range(len(agents)):
        for agent in agents:
            if agent.idx == counter:
                ordered_agents.append(agent)

    return graph, visiting, ordered_agents
