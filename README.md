# SAPIENS

SAPIENS is a reinforcement learning algorithm where multiple **off-policy agents** solve the same task in parallel and exchange experiences on the go. The group is characterized by its **topology**, a graph that determines who communicates with whom.

In the current implementation all agents are DQNs and exchange experiences have the form of **transitions from their replay buffers**.

\---

This repository contains:

* our implementation of SAPIENS under the [sapiens](sapiens) directory

* a colab notebook on  [How to use SAPIENS](https://colab.research.google.com/drive/1_iwb0rkBgDUzWOcuP96BpOLdw0QuLd0c?usp=sharing) that guides you through setting up, launching experiments and visualizing performance. In particular, you will see how to run a fully-connected and dynamic topology and how to compare their innovation abilities, behavioral and mnemonic metrics.

* in directory [projects](projects) we save the results produced when finishing an experiment with SAPIENS. Each run is saved in its own sub-directory, which contains multiple trials.  We have put all projects discussed in the paper under [projects/paper](projects/paper) (we have included the data required to generate the plots and tables, model files have been excluded due to their large size) 

* scripts for:

  *  launching all simulations reported in the paper (for this run [scripts/reproduce_runs.py](scripts/reproduce_runs.py)). Results will be saved under [projects](project) and each project will contain visualizations and data produced during evaluation
  * reproducing all plots of the paper (for this run [scripts/reproduce_plots.py](scripts/reproduce_plots.py) )

  

