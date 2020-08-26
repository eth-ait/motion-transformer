from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 2
MEMORY = 10000
NUM_GPU = 1
WALL_TIME = 3
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceGTX1080Ti]" '

model_ids = ["1598363954.1", "1598363954.2", "1598363954.3"]

# experiments_command = "python spl/evaluation.py --seq_length_out 60 --glog_entry "
experiments_command = "python spl/evaluation.py --seq_length_out 600 --visualize --to_video --dynamic_test_split --eval_dir /cluster/work/hilliges/kamanuel/trained_models/motion-modelling/reproducing_experiments_08_20_eval "

# Create a unique experiment timestamp.
for work_id, model_id in enumerate(model_ids):
    time.sleep(1)
    experiment_command = experiments_command + ' --model_id ' + model_id

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    model_id + "_eval",
                                                    MEMORY,
                                                    NUM_GPU)
    call([cluster_command + experiment_command], shell=True)
