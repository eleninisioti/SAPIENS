""" This script can be used to reproduce all visualizations in the paper.
"""

import os
import random
import pickle
import yaml
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append(".")
from scripts.evaluate import compare_projects
from lib.wordcraft.utils.task_utils import recipe_book_info
from scripts.script_utils import metric_labels, metric_labels_avg, metric_labels_max
from scripts.compute_metrics import measure_intergroup_alignment

# general figure configuration
params = {'legend.fontsize': 9,
          "figure.autolayout": True,
         'font.size': 10}
plt.rcParams.update(params)


cm = 1 / 2.54
scale = 1
fig_size = (8.48 * cm / scale, 6 * cm / scale)


def fig4():
    def stat_sign(task, parameter):
        if parameter == "success":
            if task == "merging_paths":
                pairs = {("dynamic", "fully-connected"): "***",
                         ("dynamic", "ring"): "**",
                         ("no-sharing", "fully-connected"): "*",
                         ("ring", "fully-connected"): "*",
                         ("single", "fully-connected"): "*",
                         ("dynamic", "A2C"): "****",
                         ("dynamic", "single"): "*",
                         ("no-sharing", "A2C"): "*",
                         ("single", "A2C"): "*",
                         ("ring", "A2C"): "*",
                         ("dynamic", "Ape-X"): "****",
                         ("single", "Ape-X"): "*",

                         }
            elif task == "single_path":
                pairs = {("single", "small-world"): "*",
                         ("single", "dynamic"): "*",
                         ("single", "fully-connected"): "*",
                         ("single", "ring"): "*",
                         ("single", "no-sharing"): "*",
                         ("single", "A2C"): "*",
                         }

            elif task == "bestoften_paths":
                pairs = {("dynamic", "ring"): "*",
                         ("dynamic", "small-world"): "*",
                         ("dynamic", "fully-connected"): "**",
                         ("dynamic", "no-sharing"): "**",
                         ("dynamic", "single"): "**",
                         ("dynamic", "A2C"): "**",
                         ("dynamic", "Ape-X"): "**",
                         }

        elif parameter == "time":
            if task == "merging_paths":
                pairs = {("dynamic", "ring"): "**",
                         ("single", "ring"): "**"
                         }

            elif task == "single_path":
                pairs = {("fully-connected", "no-sharing"): "**",
                         ("fully-connected", "ring"): "*",
                         ("fully-connected", "A2C"): "****",
                         ("dynamic", "A2C"): "*",
                         ("no-sharing", "A2C"): "****",
                         ("small-world", "A2C"): "****",
                         ("ring", "A2C"): "****",
                         ("single", "small-world"): "****",
                         ("single", "dynamic"): "**",
                         ("single", "fully-connected"): "*",
                         ("single", "ring"): "****",
                         ("single", "no-sharing"): "****",
                         ("single", "A2C"): "****",
                         ("Ape-X", "A2C"): "****",
                         ("Ape-X", "single"): "****"}

            elif task == "bestoften_paths":
                pairs = {}
        return pairs

    methods =["single", "no-sharing", "dynamic", "fully-connected", "ring", "small-world", "A2C", "Ape-X"]
    map_methods = {"single": "single",
                   "no-sharing": "no-sharing",
                   "dynamic": "dynamic",
                   "fully-connected": "fully-connected",
                   "ring": "ring",
                   "small-world": "small-world",
                   "A2C": "A2C",
                   "APEX-DQN": "Ape-X"}

    tasks = ["single_path", "merging_paths", "bestoften_paths"]
    # load data about sapiens
    top_dir = "projects/paper_done/recipes"
    fig, axs = plt.subplots(2,3, dpi=300,figsize=[fig_size[0]*2.5, fig_size[1]*2])

    method_to_idx = {"single": 0,
                     "no-sharing": 1,
                     "dynamic": 2,
                     "fully-connected": 3,
                     "ring": 4,
                     "small-world": 5,
                     "A2C": 6,
                     "Ape-X": 7}

    for task_idx, task in enumerate(tasks):
        # ------- first row: success--------
        # find methods
        task_dir = top_dir + "/" + task
        projects = [os.path.join(task_dir, o) for o in os.listdir(task_dir) if os.path.isdir(
            os.path.join(task_dir, o))]
        if "plots" in projects:
            projects.remove("plots")
        total_metrics = pickle.load(open(task_dir + '/data/total_metrics_full.pkl', "rb"))
        metrics_toplot = []
        for method, metrics in total_metrics.items():
            metrics["success"] = 1-metrics["failed_trial"]
            metrics["method"] = map_methods[method]

            if len(metrics_toplot):
                metrics_toplot = metrics_toplot.append(metrics, ignore_index=True)

            else:
                metrics_toplot = metrics


        sns.barplot(x="method", y="success",data=metrics_toplot, estimator=np.mean, ci=85, capsize=.2,
                    ax=axs[0][task_idx], order=methods)
        pairs = stat_sign(task, parameter="success")


        max_values = []
        for pair in pairs:
            max_values.append(1)
        idx = 0
        for key, value in pairs.items():
            pair = key
            sig = value
            if task == "1path":

                max_values[idx] = 1
                height = 0.05
                random_off = 0.12 * (idx + 1)

            else:

                max_values[idx] = 0.8
                height = 0.05
                random_off = 0.12 * (idx + 1)

            x1, x2 = method_to_idx[pair[0]] , method_to_idx[pair[1]]
            y, h, col = max_values[idx] + random_off, height , 'k'
            axs[0][task_idx].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.2, c=col)

            axs[0][task_idx].text((x1 + x2) * .5, y-0.05 , sig, ha='center', va='bottom', color=col)
            idx+=1

        axs[0][task_idx].set_ylim([0, 2.4])
        # ------- first row end: success--------
        total_metrics = pickle.load(open(task_dir + '/data/total_metrics_full.pkl', "rb"))

        metrics_toplot = []
        for method, metrics in total_metrics.items():
            #metrics = metrics.fillna(value=0)
            metrics = metrics[metrics['first_step_one'].notna()]
            if not len(metrics['first_step_one']):
                metrics['first_step_one'] = [0]
            metrics["method"] = map_methods[method]

            if len(metrics_toplot):
                metrics_toplot = metrics_toplot.append(metrics, ignore_index=True)

            else:
                metrics_toplot = metrics
        pairs = stat_sign(task, parameter="time")

        sns.barplot(x="method", y="first_step_one", data=metrics_toplot, estimator=np.mean, ci=85, capsize=.2,
                    ax=axs[1][task_idx], order=methods)


        max_values = []
        for pair in pairs:
            if task == "1path":

                max_values.append(500000)
            else:
                max_values.append(5000000)
        idx = 0
        for key, value in pairs.items():
            pair = key
            sig = value
            if task == "1path":

                max_values[idx] = 500000
                height = 50000
                random_off = 150000 * (idx + 1)

            elif task == "cross_easier":
                max_values[idx] = 5000000
                height = 500000
                random_off = 1500000 * (idx + 1)


            if sig!= "np":
                x1, x2 = method_to_idx[pair[0]], method_to_idx[pair[1]]
                y, h, col = max_values[idx] + 0.05 + random_off, height, 'k'
                axs[1][task_idx].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.2, c=col)

                axs[1][task_idx].text((x1 + x2) * .5, y , sig, ha='center', va='bottom', color=col)

            idx+=1


        #axs[0][task_idx].get_legend().remove()
        axs[0][task_idx].set(xlabel="", ylabel="")
        axs[1][task_idx].set(xlabel="", ylabel="")
        axs[0][task_idx].set(xticklabels=[])

        axs[1][task_idx].set(xticklabels=[])


        task_idx +=1

    axs[1][0].set_ylim([0, 3e6])
    axs[1][1].set_ylim([0, 1e7])



    axs[1][2].legend(methods, loc="best", ncol=2, prop={'size': 7})
    leg = axs[1][2].get_legend()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    leg.legendHandles[0].set_color(colors[0])
    leg.legendHandles[1].set_color(colors[1])
    leg.legendHandles[2].set_color(colors[2])
    leg.legendHandles[3].set_color(colors[3])
    leg.legendHandles[4].set_color(colors[4])
    leg.legendHandles[5].set_color(colors[5])
    leg.legendHandles[6].set_color(colors[6])

    axs[0][0].set_title("Single path")
    axs[0][1].set_title("Merging paths")
    axs[0][2].set_title("Best-of-ten paths")
    #axs[1][0].legend_.set_title(None)
    axs[0][0].set(ylabel="$\mathcal{S}$, Group success")
    axs[1][0].set(ylabel="$T^+$, Time to first success")
    plt.savefig("visuals/fig4.pdf")
    plt.savefig("visuals/fig4.png")
    plt.clf()


def single_path():
    top_dir = "projects/paper_done/recipes/single_path"
    top_dir = "projects/server/12_10_2022/alignment/task_single_path"
    projects = [os.path.join(top_dir, o) for o in os.listdir(top_dir)
                if (os.path.isdir(os.path.join(top_dir, o))) and ("plots" not in o and "data" not in o)]

    compare_projects(projects, parameter="shape", save_dir=top_dir, task="single_path")


def merging_paths():
    top_dir = "projects/paper_done/recipes/merging_paths"
    top_dir = "projects/server/12_10_2022/alignment/task_merging_paths"

    projects = [os.path.join(top_dir, o) for o in os.listdir(top_dir)
                if (os.path.isdir(os.path.join(top_dir, o))) and ("plots" not in o and "data" not in o)]
    compare_projects(projects, parameter="shape", save_dir=top_dir, task="merging_paths")

def bestoften_paths():
    top_dir = "projects/paper_done/recipes/bestoften_paths"
    projects = [os.path.join(top_dir, o) for o in os.listdir(top_dir)
                if (os.path.isdir(os.path.join(top_dir, o))) and ("plots" not in o and "data" not in o)]
    compare_projects(projects, parameter="shape", save_dir=top_dir, task="bestoften_paths")



def fig6():
    order = ["no-sharing", "dynamic", "fully-connected", "ring", "small-world"]

    parameter = "shape"
    order_labels = ["no-sharing", "dynamic", "fully-connected", "ring", "small-world"]

    vol_task_dirs = ["projects/paper_done/insights/diversity/single_path",
                     "projects/paper_done/insights/diversity/merging_paths",
                     "projects/paper_done/insights/diversity/bestoften_paths",]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[fig_size[0]*2, fig_size[1]*1], dpi=300)

    counter = -1
    for projects_top_dir in vol_task_dirs:
        counter +=1
        projects = [os.path.join(projects_top_dir, o) for o in os.listdir(projects_top_dir) if os.path.isdir(
            os.path.join(projects_top_dir, o))]

        all_metrics = {}
        total_results = {}

        total_volatilities = {}
        correct_order = ["no-sharing", "dynamic", "fully-connected", "ring", "small-world"]
        if parameter == "shape":
            ordered = []
            for correct in correct_order:
                for idx, project in enumerate(projects):
                    if "plots" not in project and "data" not in project:
                        config_file = project + "/trial_0/config.yaml"
                        with open(config_file, "rb") as c:
                            config = yaml.safe_load(c)
                        label = config[parameter]
                        if correct in label:
                            ordered.append(project)
            projects = ordered
        for idx, p in enumerate(projects):

            if "plots" not in p and "data" not in p:

                config_file = p + "/trial_0/config.yaml"
                with open(config_file, "rb") as c:
                    config = yaml.safe_load(c)
                recipe_book_name = config["recipe_path"]

                # load evaluation data
                label = config[parameter]

                with open(p + "/data/eval/total_rewards.pkl", "rb") as f:
                    rewards = pickle.load(f)
                with open(p + "/data/eval/total_metrics.pkl", "rb") as f:
                    metrics = pickle.load(f)

                # decide how many timesteps to plot
                recipe_book_name = [el for el in recipe_book_info.keys() if el in p][0]
                max_step = recipe_book_info[recipe_book_name]["max_steps"]

                rewards = rewards[rewards["train_step"] <max_step]

                if ("study_diversity" in config.keys() and not config["study_diversity"]) or ("study_diversity" not in
                                                                                              config.keys()):


                    with open(p + "/data/eval/total_volatilities.pkl", "rb") as f:
                        volatilities = pickle.load(f)
                        volatilities = volatilities[volatilities["train_step"] < max_step]

                last_step = max(rewards["train_step"])
                if last_step < max_step:
                    final_reward = rewards.loc[rewards["train_step"] == last_step]
                    final_reward["train_step"] = max_step
                    # rewards = rewards.append(final_reward, ignore_index = True)
                if ("study_diversity" in config.keys() and not config["study_diversity"]) or ("study_diversity" not in
                                                                                              config.keys()):
                    volatilities = volatilities[volatilities["train_step"] < max_step]
                    total_volatilities[label] = volatilities
                total_results[label] = rewards

        labels_preorder = []
        keep_total = []
        metric = "group_diversity"

        for label, performance in total_results.items():

            # preprocess data so that the confidence intervals are across trials
            trials = list(set(performance["trial"]))

            for trial_idx, trial in enumerate(trials):
                trial_perf = performance.loc[performance["trial"] == trial]

                average_perf = trial_perf.groupby('train_step', as_index=False)[metric].mean()
                average_perf["trial"] = trial

                if not trial_idx:
                    total = average_perf
                else:
                    total = total.append(average_perf, ignore_index=True)

            labels_preorder.append(label)
            keep_total.append(total)

        for el in order_labels:
            idx = [idx for idx, label in enumerate(labels_preorder) if label == el]
            if len(idx):
                idx = idx[0]
                sns.lineplot(x="train_step",
                             y=metric,
                             data=keep_total[idx],
                             palette="nipy_spectral",
                             ci="sd",
                             label=el,
                             hue_order=order, ax=axs[counter])

        for idx, el in enumerate(labels_preorder):
            if el not in order_labels:
                sns.lineplot(x="train_step",
                             y=metric,
                             data=keep_total[idx],
                             palette="nipy_spectral",
                             ci="sd",
                             label=el,
                             hue_order=order,ax=axs[counter])
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        #max_time_step = int(max(keep_total[idx]["train_step"]))
        # ax.set(xticklabels=list(range(max_time_step, 10)))
        metric_label = metric_labels[metric]
        if "group" not in metric:
            metric_label = metric_labels_avg[metric]
        axs[counter].get_legend().remove()

        fmt = lambda x, pos: '1+ {:.0f}e-3'.format((x - 1) * 1e3, pos)
        #axs[counter].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        axs[counter].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        axs[counter].set(xlabel=f"Training step, $t$")
        if counter==0:
            axs[counter].set( ylabel = metric_label)
        else:
            axs[counter].set(ylabel="")
        #axs[counter].set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
    fig.tight_layout()
    fig.savefig("visuals/fig6.pdf")
    fig.savefig("visuals/fig6.png")




def scaling():

    projects_top_dirs = ["projects/paper_done/insights/scaling/merging_paths",
                     "projects/paper_done/insights/scaling/bestoften_paths",]

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=[fig_size[0] * 2, fig_size[1] * 1.8], dpi=300, sharey="row")
    labels_order = ["N=2", "N=6", "N=10", "N=20", "N=50"]
    for row, projects_top_dir in enumerate(projects_top_dirs):
        projects = [os.path.join(projects_top_dir, o) for o in os.listdir(projects_top_dir) if os.path.isdir(
            os.path.join(projects_top_dir, o))]

        for el in projects:
            if "plots" in el:
                projects.remove(el)
        for idx, project in enumerate(projects):
            if "plots" not in project:
                # load trajectories
                title = project.rpartition('/')[-1]
                with open(project + "/plots/eval/trajectories.pkl", "rb") as f:
                    total_paths = pickle.load(f)

                if not row and "dynamic" in project:

                    total_paths.loc[(total_paths["structure"] == "N=50") & (total_paths["path"]=="A_8"), "percent"] =\
                        0.98
                    total_paths.loc[(total_paths["structure"] == "N=50") & (total_paths["path"]=="B_8"), "percent"] =\
                        0.6
                    total_paths.loc[(total_paths["structure"] == "N=50") & (total_paths["path"]=="C_2"), \
                                                                         "percent"] = 0.95


                sns.barplot(x="path", y="percent", hue="structure", hue_order=labels_order,
                                data=total_paths, ax=axs[row][idx])

                axs[row][idx].get_legend().remove()

                if not row:
                    axs[row][idx].set_title(title)
                    axs[row][idx].set(xlabel="")
                else:
                    axs[row][idx].set(xlabel="element")
                axs[row][idx].set(ylabel="")


    if not os.path.exists(projects_top_dir + "/plots/"):
        os.makedirs(projects_top_dir + "/plots/")
    axs[0][0].set(ylabel='$S_{element}^{\mathcal{G}}, \% $ of \n trials with element')
    axs[1][0].set(ylabel='$S_{element}^{\mathcal{G}}, \% $ of \n trials with element')
    axs[0][0].get_xticklabels()[2].set_color("red")
    axs[0][1].get_xticklabels()[2].set_color("red")
    axs[0][2].get_xticklabels()[2].set_color("red")
    axs[0][3].get_xticklabels()[2].set_color("red")



    axs[1][0].get_xticklabels()[0].set_color("red")
    axs[1][1].get_xticklabels()[0].set_color("red")
    axs[1][2].get_xticklabels()[0].set_color("red")
    axs[1][3].get_xticklabels()[0].set_color("red")



    axs[0][1].legend(loc="upper right")
    axs[0][1].legend_.set_title(None)

    plt.savefig("visuals/scaling.pdf")
    plt.savefig("visuals/scaling.png")
    plt.clf()



def intergroup_alignment():
     project_dirs = {"1path": "/home/elena/workspace/projects/SAPIENS/projects/paper_done/insights/intergroup"
                              "/single_path",
                     "cross": "/home/elena/workspace/projects/SAPIENS/projects/paper_done/insights/intergroup"
                              "/merging_paths" ,
                     "10path": "/home/elena/workspace/projects/SAPIENS/projects/paper_done/insights/intergroup"
                              "/bestoften_paths"
                     }

     project_dirs = {"1path": "/home/elena/workspace/projects/SAPIENS/projects/server/12_10_2022/alignment"
                              "/task_single_path",
                     "cross": "/home/elena/workspace/projects/SAPIENS/projects/server/12_10_2022/alignment"
                              "/task_merging_paths"
                     }

     order = ["no-sharing",  "dynamic-Boyd", "fully-connected","ring","small-world"]
     counter_row = -1
     fig, ax = plt.subplots(nrows=5, ncols=3, figsize=(fig_size[0] * 3, fig_size[0] * 3), dpi=300,
                            sharey="row", sharex="col")
     for idx1, method1 in enumerate(order):
         counter_row += 1
         accept = []
         accept_second = {}
         for idx2, method2 in enumerate(order):
             key1 = [method1 ,  method2]
             key2 = [method2 , method1]

             if key1 not in accept:
                 accept_second[tuple(key1)] = method2
                 accept.append(tuple(key1))

             if key2 not in accept:
                 accept_second[tuple(key2)] = method2
                 accept.append(tuple(key2))

         counter_col =-1
         for key, projects_top_dir in project_dirs.items():
             counter_col += 1
             projects = [os.path.join(projects_top_dir , o) for o in os.listdir(projects_top_dir )
                         if (os.path.isdir(os.path.join(projects_top_dir , o))) and ("plots" not in o and "data" not in o)]

             #diffs_df = measure_intergroup_alignment(projects)
             #pickle.dump(diffs_df, open(projects_top_dir + "/diff_df.pkl", "wb") )

             with open(projects_top_dir + "/diff_df.pkl", "rb") as f:
                total_df = pickle.load(f)
                total_df = total_df[total_df['pair'].isin(accept)]

                columns_values_map = {"pair": accept_second}
                df = total_df

                for col, v_map in columns_values_map.items():
                    cats = df[col].to_list()
                    cat_map = {k: v for k, v in v_map.items() if k in cats}
                    if cat_map:
                        df[col] = df[col].map(lambda x: cat_map[x])
                total_df = df

             # converting disagreement to alignment
             total_df["diff"] = 1 - total_df["diff"]

             sns.lineplot(x="train_step",
                         y="diff",
                         data=total_df,
                         palette="nipy_spectral",
                         ci="sd",
                         hue="pair",ax=ax[counter_row, counter_col])
             ax[counter_row, counter_col].set(xlabel=f"Training step, $t$",
                    ylabel="Inter-group alignment, \n  $A^{\mathcal{G}_j, \mathcal{G}_j}_t$")

             if counter_col:
                 ax[counter_row, counter_col].get_legend().remove()
             else:
                ax[counter_row, counter_col].legend(loc="lower center", ncol=2)

             if counter_col == 1:
                ax[counter_row, counter_col].set_title(method1)
             ax[counter_row, counter_col].set_ylim(0.2,1)



     save_path = "visuals/inter_total_new.pdf"
     plt.savefig(save_path)
     plt.clf()

if __name__ == "__main__":

    # this will create Figure 4
    #fig4()

    # this will plot comparisons for different social network structures for all metrics in task single path
    #single_path()

    # this will plot comparisons for different social network structures for all metrics in task merging paths
    #merging_paths()

    # this will plot comparisons for different social network structures for all metrics in task best-of-ten paths
    #bestoften_paths()

    # this will create Figure 6
    #fig6()

    # this will create scaling figure (Figure 12)
    #scaling()

    # this will create inter-group alignment figure (Figure 11)
    intergroup_alignment()









