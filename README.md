# SAPIENS: : Structuring multi-Agent toPology for Innovation through ExperieNce Sharing

------

SAPIENS is a reinforcement learning algorithm where multiple **off-policy agents** solve the same task in parallel and exchange experiences on the go. The group is characterized by its **topology**, a graph that determines who communicates with whom.

As this visualization shows in our current implementation all agents are DQNs and exchange experiences have the form of **transitions from their replay buffers**.

![Experience Sharing DQN](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2FVJlf5jueXb.png?alt=media&token=56a4f560-23bb-4098-b65c-1303ddbb8dc0)

Using SAPIENS we can define groups of agents that are connected with others based on a a) fully-connected topology b) small-world topology c) ring topology or d) dynamic topology



![topologies](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2Feleni%2FUn1BHk2PmM.png?alt=media&token=dd0b0588-945a-4873-9af7-605e2055d567)



## Running experiments

### Install required packages

You can install all required python packages by creating a new conda environment containing the packages in [environment.yml](environment.yml):

`conda env create -f environment.yml`

And then activating the environment:

`conda activate sapiens`

### Example usages

Under [notebooks](notebooks) there is a Jupyter notebook that will guide you through setting up simulations with a fully-connected and a dynamic social network structure for solving Wordcraft tasks. It also explains how you can access visualizations of the metrics produced during the experiment. You can also directly open it in [google colab](https://colab.research.google.com/drive/1_iwb0rkBgDUzWOcuP96BpOLdw0QuLd0c?usp=sharing).

### Reproducing the paper results

Scripts under the [scripts](scripts) directory are useful for this:

With [scripts/reproduce_runs.py](scripts/reproduce_runs.py) you can run all simulations presented in the paper from scratch.

This file is useful for looking at how the experiments were configured but better avoid running it: simulations will run locally and sequentially and will take months to complete.

Instead, you can access the output of our simulations on this [online repo](https://drive.google.com/drive/folders/1x6NZe2Aw3udhDNi-V0ljgFs_uFPPzK7l?usp=sharing). Due to space limitation we do not include all checkpoint models, but the log files produced by processing the checkpoint models.

Download this zip file and uncompress it under the [projects](projects) directory. This should create a projects/paper_done sub-directory.

You can now reproduce all visualization presented in the paper. Run:

`python scripts/reproduce_visuals.py`

This will save some general plots under [visuals](visuals) and project-specific plots are saved under the corresponding project in [projects/paper_done]([projects/paper_done])











