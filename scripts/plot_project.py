import matplotlib.pyplot as plt
import seaborn as sns

from scripts.script_utils import metric_labels, metric_labels_avg, metric_labels_max

cm = 1 / 2.54
FIG_SIZE = (8.48 * cm, 6 * cm)
ORDER = ["no-sharing", "dynamic-Boyd", "fully-connected", "small-world", "ring"]


def plot(eval_info, volatilities, conformities, measure_mnemonic, intergroup_alignment, project):
    plot_metric_with_time(eval_info, "norm_reward", project)
    plot_metric_with_time(volatilities, "volatility", project)
    plot_metric_with_time(conformities, "group_conformity", project)

    if measure_mnemonic:
        plot_metric_with_time(eval_info, "diversity", project)
        plot_metric_with_time(eval_info, "group_diversity", project)
        plot_metric_with_time(eval_info, "intragroup_alignment", project)

    if len(intergroup_alignment):
        plot_metric_with_time(intergroup_alignment, "intergroup_alignment", project)


def plot_metric_with_time(data, metric, project):
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
        if "group" not in metric and mode=="avg":
            metric_label = metric_labels_avg[metric]
        elif mode=="max":
            metric_label = metric_labels_max[metric]
        ax.set(xlabel=f"Training step, $t$", ylabel=metric_label)
        fig.tight_layout()

        save_path = project + "/plots/" + mode + "_" + metric + ".pdf"
        plt.savefig(save_path)
        plt.clf()
