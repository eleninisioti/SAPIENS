import os
import sys
sys.path.append(os.getcwd())
print(os.getcwd())

import server_scripts.experiments


def run_server(job_name, trial, gpu=False, time="20:00:00", long_run=False, account="jeanzay_id"):
    """
    TODO
    """
    script = "scripts/experiments.py " + str(trial)
    if account == "jeanzay_id":
        print("You must replace jeanzay_id with your own jeanzay id to run experiments on the server")

    logs_dir = f"/gpfsscratch/rech/imi/{account}/sapiens_log/jz_logs"
    python_path = "python"
    slurm_dir = f"/gpfsscratch/rech/imi/{account}/slurm"

    # create logging directories
    if not os.path.exists(slurm_dir + "/" + job_name):
        os.makedirs(slurm_dir + "/" + job_name)
    if not os.path.exists(logs_dir + "/" + job_name):
        os.makedirs(logs_dir + "/" + job_name)

    slurmjob_path = os.path.join(slurm_dir + "/" + job_name + "/script.sh")
    create_slurmjob_cmd = "touch {}".format(slurmjob_path)
    os.system(create_slurmjob_cmd)

    # write arguments into the slurmjob file
    with open(slurmjob_path, "w") as fh:
        fh.writelines("#!/bin/sh\n")
        if gpu:
            fh.writelines("#SBATCH --account=imi@v100\n")
            fh.writelines("#SBATCH --gres=gpu:1\n")
            if long_run:
                fh.writelines("#SBATCH --qos=qos_gpu-t4\n")
        else:
            fh.writelines("#SBATCH --account=imi@cpu\n")
            fh.writelines(f"#SBATCH --ntasks=1\n")

            if long_run:
                fh.writelines("#SBATCH --qos=qos_cpu-t4\n")
        fh.writelines(f"#SBATCH --cpus-per-task=10\n")
        fh.writelines("#SBATCH --job-name={}\n".format(job_name))
        fh.writelines("#SBATCH -o {}/{}_%j.out\n".format(logs_dir, job_name))
        fh.writelines("#SBATCH -e {}/{}_%j.err\n".format(logs_dir, job_name))
        fh.writelines(f"#SBATCH --time={time}\n")
        fh.writelines("#SBATCH --hint=nomultithread\n")
        batch_cmd = python_path + " " + script
        fh.writelines(batch_cmd)

    os.chdir(os.getcwd())
    os.system("sbatch %s" % slurmjob_path)


if __name__== "__main__":
    job_name = "run_mnemonic"
    for trial in range(5):
        run_server(job_name=job_name +"_trial_" + str(trial), trial=trial, account="utw61ti")