import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.script_utils import metric_labels, metric_labels_avg, metric_labels_max

cm = 1 / 2.54
FIG_SIZE = (8.48 * cm, 6 * cm)
ORDER = ["no-sharing", "dynamic-Boyd", "fully-connected", "small-world", "ring"]


def plot_intergroup_alignment(alignment, save_dir):
    """ Plot inter-group alignment

     alignment: Dataframe
        contains infromation collected during the evaluation of the project

    save_dir: str
        directory in which to save the alignment plot

    """
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

    sns.lineplot(x="train_step",
                 y="diff",
                 data=alignment,
                 palette="nipy_spectral",
                 ci="sd",
                 hue="pair")

    plt.set(xlabel=f"Training step, $t$", ylabel="Inter-group alignment, \n  $A^{\mathcal{G}_j, \mathcal{G}_j}_t$")
    save_path = save_dir + "/plots"
    if not os.path.exists(save_path):
        os.path.makedirs(save_path)

    plt.savefig(save_path + "inter_total.pdf")
    plt.savefig(save_path + "inter_total.png")
    plt.clf()


def plot_project(eval_info, volatilities, conformities, measure_mnemonic, project):
    """ Produce all plots related to a project.

    eval_info: Dataframe
        contains infromation collected during the evaluation of the project

    volatilities: Dataframe
        contains infromation about volatility

    conformities: Dataframe
        contains infromation about conformity

    measure_mnemonic: bool
        indicates whether to plot mnemonic metrics


    project: str
        project directory

    """
    plot_metric_with_time(eval_info, "norm_reward", project)
    plot_metric_with_time(volatilities, "volatility", project)
    plot_metric_with_time(conformities, "group_conformity", project)

    if measure_mnemonic:
        plot_metric_with_time(eval_info, "diversity", project)
        plot_metric_with_time(eval_info, "group_diversity", project)
        plot_metric_with_time(eval_info, "intragroup_alignment", project)


def plot_metric_with_time(data, metric, project):
    """ Plot a specific metric with training time.

    data: Dataframe
        contains evaluation information

    metric: str
        name of metric

    project: str
        project directory
    """
    # ----- plot average and maximum performance across population -----
    if "group" in metric:
        modes = ["avg"]
    else:
        modes = ["avg", "max"]
    for mode in modes:
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        all_labels = []
        for label, performance in data.items():

            # preprocess data so that the confidence intervals are across trials
            trials = list(set(performance["trial"]))
            total = []

            for trial_idx, trial in enumerate(trials):
                trial_perf = performance.loc[performance["trial"] == trial]

                if mode == "avg":
                    average_perf = trial_perf.groupby('train_step', as_index=False)[metric].mean()
                else:
                    average_perf = trial_perf.groupby('train_step', as_index=False)[metric].max()
                average_perf["trial"] = trial

                if len(total):
                    total = total.append(average_perf, ignore_index=True)
                else:
                    total = average_perf

            all_labels.append(total)
        labels = list(data.keys())

        for idx, el in enumerate(all_labels):
            sns.lineplot(x="train_step",
                         y=metric,
                         data=all_labels[idx],
                         palette="nipy_spectral",
                         ci="sd",
                         label=labels[idx],
                         hue_order=ORDER)

        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        metric_label = metric_labels[metric]
        if "group" not in metric and mode == "avg":
            metric_label = metric_labels_avg[metric]
        elif mode == "max":
            metric_label = metric_labels_max[metric]
        ax.set(xlabel=f"Training step, $t$", ylabel=metric_label)
        fig.tight_layout()

        save_dir = project + "/plots/"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + mode + "_" + metric + ".pdf")
        plt.savefig(save_dir + mode + "_" + metric + ".png")

        plt.clf()
