import pickle
import os
import shutil

top_dir = "projects/paper_done/recipes/merging_paths"
projects = [os.path.join(top_dir, o) for o in os.listdir(top_dir)
                if (os.path.isdir(os.path.join(top_dir, o))) and ("plots" not in o and "data" not in o)]

for project in projects:

    volatilities = pickle.load(open(project + "/data/eval/total_volatilities.pkl", "rb"))
    conformities = pickle.load(open(project + "/data/eval/total_conformities.pkl", "rb"))

    with open(project + "/data/behavioral_metrics.pkl", "wb") as f:
        pickle.dump({"volatility": volatilities, "conformity": conformities}, f)

    src = project + "/data/eval/total_rewards.pkl"
    dst = project + "/data/eval_info.pkl"
    shutil.copyfile(src, dst)