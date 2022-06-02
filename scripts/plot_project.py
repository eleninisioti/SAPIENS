
def plot(data, measure_mnemonic, measure_intergroup_alignment):

    plot_label_with_time(data, label="norm_reward")
    plot_label_with_time(data, label="volatility")
    plot_label_with_time(data, label="group_conformity")

    if measure_mnemonic:
        plot_label_with_time(data, label="diversity")
        plot_label_with_time(data, label="group_diversity")

    if measure_integroup_alignment:







def plot_label_with_time(data, label):
    # ----- plot average performance across population -----

