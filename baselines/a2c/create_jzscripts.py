
import os

def save_script(lr, gamma, n_steps, task, trial,vf_coef, ent_coef,
                                                    max_grad_norm, n_agents):
    scripts_dir = "jz_scripts/insights"
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
    script_path = scripts_dir + "/" + task + "_g_" + str(gamma) + "_nsteps_" + str(n_steps) + "_lr_"\
                  + str(
        lr) + \
                  "_trial_" + str(trial) + "_cf_" + str(vf_coef) + "_ent_" + str(ent_coef) + \
                  "_grad_" + str(max_grad_norm) +"_nagents_" + str(n_agents)
    mode = "server"
    with open(script_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        fh.writelines("#SBATCH --job-name=sing0\n")
        fh.writelines("#SBATCH --account=imi@v100\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --time=20:00:00\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --gres=gpu:1\n")
        fh.writelines("#SBATCH --cpus-per-task=10\n")
        fh.writelines("#SBATCH --hint=nomultithread\n" )
        jz_file = "finaltune_" +  task + "_g_" + str(gamma) + "_nsteps_" + str(n_steps) + "_lr_" + str(lr) + \
                  "_trial_" + str(trial)  + "_cf_" + str(vf_coef) + "_ent_" + str(ent_coef) + \
                  "_grad_" + str(max_grad_norm) +"_nagents_" + str(n_agents)
        fh.writelines("#SBATCH --output=/gpfsscratch/rech/imi/utw61ti/a2c_log/jz_logs/" + jz_file + ".out\n" )
        fh.writelines("#SBATCH --error=/gpfsscratch/rech/imi/utw61ti/a2c_log/jz_logs/" + jz_file + ".err\n")
        fh.writelines("module load pytorch-gpu/py3/1.7.1\n")
        python_command = "python scripts/train/train.py --task " + task + " --lr " + str(lr) + " " \
                                                                                                             "--gamma " + str(
            gamma) + " --nsteps " + str(n_steps)  + " --trial " + str(trial)  + " --vf_coef " + str(
            vf_coef) + " --ent_coef " + str(ent_coef)  + " --grad " + str(max_grad_norm) + " --n_agents " + str(
            n_agents) + " --mode " + str(mode)
        fh.writelines(python_command)

lr = 0.001
gamma = 0.99
n_steps = 5
vf_coef = 0.25
ent_coef_values = [0.1,1]
max_grad_norm= 0.5
trials = list(range(10))
n_agents_values = [1,10]
tasks = [ "merging_paths"]
for task in tasks:
    for trial in trials:
        for ent_coef in ent_coef_values:
            for n_agents in n_agents_values:
                save_script(lr, gamma, n_steps, task, trial, vf_coef, ent_coef,
                            max_grad_norm, n_agents)
