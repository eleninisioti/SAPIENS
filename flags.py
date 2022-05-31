""" Contains arguments for configurating an experiment.
"""

import argparse

def get_cmd_args():
    """ Defines all arguments and their default values.
    """
    parser = argparse.ArgumentParser(description="Shared experience DQN")

    # ----- task settings -----
    parser.add_argument(
        "--max_depth",
        type=int,
        default=8,
        help="maximum innovation level of goal state",
    )
    parser.add_argument(
        "--max_mix_steps",
        type=int,
        default=8,
        help="maximum allowed mixing steps per innovation level (max_depth * max_mix_steps = episode_length)",
    )
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=2,
        help="number of useless entities originally on the table to 'distract' the agent",
    )
    parser.add_argument(
        "--wrapper_idx",
        choices=[0, 1, 2, 3, 4],
        type=int,
        default=0,
        help="0 applies no wrapping, 1 is squash wrapping, 2 is embodied, 3 is chooser, 4 is noGoal (automatic proportional reward)",
    )
    parser.add_argument(
        "--reward",
        default=0,
        type=float,
        choices=[0, 1, 2],
        help="(mostly obsolete) 0 means no sub_goal rewards, 1 means constant subgoal rewards, 2 means incremental subgoal rewards",
    )

    parser.add_argument(
        "--recipe_path",
        default="",
        type=str,
        help="path to recipe book",
    )

    # ----- DQN settings -----
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="discount factor",
    )
    parser.add_argument(
        "--prioritized",
        type=int,
        default=False,
        help="when true, the buffer sampling will be proportional to the absolute TD error (this also affects sampling during the sharing phase)",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=30000, help="Experiment timesteps"
    )
    parser.add_argument("--buffer_size", type=int, default=100000, help="Name of model")

    parser.add_argument(
        "--n_agents",
        type=int,
        default=2,
        help="number of interacting agents",
    )

    parser.add_argument(
        "--num_neurons",
        type=int,
        default=64,
        help="number of neurons in one layer",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="number of layers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="number of examples in batch",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="temp",
        help="Name of experiment",
    )

    parser.add_argument(
        "--explore",
        type=str,
        default=0,
        help="If 1, the default hyperparams for epsilon-greedy exploration are used. If 0, no exploration.",
    )

    # ----- communication settings -----
    parser.add_argument(
        "--shape",
        type=str,
        default=1,
        help="shape of the communication graph, where each node is an agent, and each edge allows communication :\n 0 --> circle,\n 1 --> fully connected,\n 2 --> small-world,\n 3 --> star, \n 4 --> dynamic, \n 5 --> custom",
    )
    parser.add_argument(
        "--shape_path",
        type=str,
        default="",
        help="Path to custom graph in gexf format, must only be used when shape is set to 5. Number of nodes must correspond to number of agents",
    )
    parser.add_argument(
        "--shared_batch",
        type=int,
        default=128,
        help="amount of transitions shared from an agent's buffer on a sharing step",
    )

    parser.add_argument(
        "--normalize",
        type=int,
        default=0,
        help="If 1, normalize the learning process so that all network structures have the same learning dynamics"
             "(ratio of gradient updates to sammples and buffer size)",
    )

    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=2,
        help="number of neighboring agents",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="number of neighboring agents",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="random seed",
    )

    parser.add_argument(
        "--fixed_seed",
        type=int,
        default=0,
        help="indicates whether the seed is fixed",
    )

    parser.add_argument(
        "--phase_periods",
        type=str,
        default="1000,10",
        help="Period of each phase (only appliccable in a dynamic setting)",
    )


    # ---- configuring projects' directories ----
    parser.add_argument(
        "--log_path",
        type=str,
        default=".",
        help="path where logfile is saved",
    )

    parser.add_argument(
        "--reload",
        type=int,
        default=0,
        help="Indicates whether to load a trained model",
    )


    parser.add_argument(
        "--playground",
        type=str,
        default="wordcraft",
        help="Choose between wordcraft and pong.",
    )

    parser.add_argument(
        "--study_diversity",
        type=int,
        default=0,
        help="1 or 0.",
    )

    parser.add_argument(
        "--visit_duration",
        type=int,
        default=10,
        help="1 or 0.",
    )

    parser.add_argument(
        "--migrate_rate",
        type=float,
        default=0.1,
        help="1 or 0.",
    )

    parser.add_argument(
        "--n_subgroups",
        type=int,
        default=2,
        help="1 or 0.",
    )

    parser.add_argument(
        "--intrinsic",
        type=int,
        default=0,
        help="1 or 0.",
    )

    return parser.parse_args()