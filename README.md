# SAPIENS: : Structuring multi-Agent toPology for Innovation through ExperieNce Sharing

------

SAPIENS is a reinforcement learning algorithm where multiple **off-policy agents** solve the same task in parallel and exchange experiences on the go. The group is characterized by its **topology**, a graph that determines who communicates with whom.

As this visualization shows in our current implementation all agents are DQNs and exchange experiences have the form of **transitions from their replay buffers**.

![Experience Sharing DQN](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2FVJlf5jueXb.png?alt=media&token=56a4f560-23bb-4098-b65c-1303ddbb8dc0)

Using SAPIENS we can define groups of agents that are connected with others based on a a) fully-connected topology b) small-world topology c) ring topology or d) dynamic topology



![topologies](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2FUn1BHk2PmM.png?alt=media&token=dd0b0588-945a-4873-9af7-605e2055d567)



## How to use SAPIENS

In [this google colab notebook](https://colab.research.google.com/drive/1_iwb0rkBgDUzWOcuP96BpOLdw0QuLd0c?usp=sharing) we explain how one can set up SAPIENS, launch experiments with different social network topologies and visualize various performance metrics.

## How to reproduce the paper's results

Under the [scripts](scripts) directory you will find scripts useful for rerunning the experiments and producing the plots presented in the paper.

In particular, running the [scripts/reproduce_runs.py](scripts/reproduce_runs.py) will run **all** experiments presented in the paper. This will take a lot of memory and time, as experiments are executed sequentially. It is however useful to take a look at how to configure experiments with various topologies, tasks and group sizes.

To avoid rerunning all experiments, we provide our data under the [projects/paper](projects/paper) directory. Due to space restrictions we do not include the model files, but the data and plots produced during their evaluation.





