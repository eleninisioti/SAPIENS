# SAPIENS: : Structuring multi-Agent toPology for Innovation through ExperieNce Sharing

------

SAPIENS is a reinforcement learning algorithm where multiple **off-policy agents** solve the same task in parallel and exchange experiences on the go. The group is characterized by its **topology**, a graph that determines who communicates with whom.

As this visualization shows in our current implementation all agents are DQNs and exchange experiences have the form of **transitions from their replay buffers**.

![Experience Sharing DQN](/home/elena/Downloads/pdf2png (2)/algorithms (1)/algorithms (1)-1.png)

Using SAPIENS we can define groups of agents that are connected with others based on a a) fully-connected topology b) small-world topology c) ring topology or d) dynamic topology



![structures-1](/home/elena/Downloads/pdf2png (2)/structures/structures-1.png)

## How to use SAPIENS

In [this google colab notebook](https://colab.research.google.com/drive/1_iwb0rkBgDUzWOcuP96BpOLdw0QuLd0c?usp=sharing) we explain how one can set up SAPIENS, launch experiments with different social network topologies and visualize various performance metrics.

## How to reproduce the paper's results

Under the [scripts](scripts) directory you will find scripts useful for rerunning the experiments and producing the plots presented in the paper.

In particular, running the [scripts/reproduce_runs.py](scripts/reproduce_runs.py) will run **all** experiments presented in the paper. This will take a lot of memory and time, as experiments are executed sequentially. It is however useful to take a look at how to configure experiments with various topologies, tasks and group sizes.

To avoid rerunning all experiments, we provide our data under the [projects/paper](projects/paper) directory. Due to space restrictions we do not include the model files, but the data and plots produced during their evaluation.





