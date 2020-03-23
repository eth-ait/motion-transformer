from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 4
MEMORY = 10000
NUM_GPU = 1
WALL_TIME = 3
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceGTX1080Ti]" '

model_ids = ["1584014132.1", "1584014132.2", "1584014132.3", "1584014132.4"]

experiments_command = "python spl/evaluation.py --seq_length_out 60 --glog_entry "

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
