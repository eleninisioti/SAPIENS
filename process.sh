#!/bin/sh
#SBATCH --account=imi@cpu
#SBATCH --cpus-per-task=20
#SBATCH --job-name=Test
#SBATCH -o /gpfsscratch/rech/imi/utw61ti/sapiens_log/jz_logs/test.out
#SBATCH -e /gpfsscratch/rech/imi/utw61ti/sapiens_log/jz_logs/test.err
#SBATCH --time=12:00:00

python3 process_projects.py 

