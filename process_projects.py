from server_scripts.evaluate import evaluate_project

project="/gpfsscratch/rech/imi/utw61ti/sapiens_log/projects/19_10_2022/alignment/task_bestoften_paths/shape_ring"
evaluate_project(project)

project="/gpfsscratch/rech/imi/utw61ti/sapiens_log/projects/19_10_2022/alignment/task_bestoften_paths/shape_dynamic" \
        "-Boyd"
evaluate_project(project)