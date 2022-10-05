""" Contains various utilities useful for scripts.
"""


metric_labels = {"norm_reward": "Reward, $R_t$",
                 "diversity": "Diversity, $D_t$",
                 "level": "Level, $L_t$",
                 "group_diversity": "Group divesrity, $D^{\\mathcal{G}}_t$",
                 "intragroup_alignment": "Intra-group alignment, $A^{\mathcal{G}}_t$",
                 "group_conformity": "Conformity, $C_t$",
                 "volatility": "Average Volatility, $\\bar{V}_t$" }

metric_labels_avg = {"norm_reward": "Average Reward, $R^+_t$",
                     "diversity": "Average Diversity, $\\bar{D}_t$",
                     "level": "Average Level, $\\bar{L}_t$",
                     "volatility": "Average Volatility, $\\bar{V}_t$" }

metric_labels_max = {"norm_reward": "Maximum Reward, $R^*_t$",
                     "diversity": "Maximum Diversity, $\\hat{D}_t$",
                     "level": "Maximum Level, $\\hat{L}_t$",
                     "volatility": "Maximum Volatility, $\\hat{V}_t$"}
