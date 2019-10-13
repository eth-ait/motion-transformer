from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 8
MEMORY = 3000
NUM_GPU = 1
WALL_TIME = 23
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '

github_reproduce = [
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875869-Seq2seq/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419538-Seq2seq-SPL/config.json',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875871-Seq2seq-dropout/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419548-Seq2seq-dropout-SPL/config.json',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564415781-Seq2seq-sampling-SPL/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json',
]

for i, experiment in enumerate(github_reproduce):
    # print(experiment)
    # Create a unique experiment timestamp.
    time.sleep(2)
    experiment_timestamp = str(int(time.time()))
    print(experiment_timestamp)
    experiment_command = experiment + ' --new_experiment_id ' + experiment_timestamp

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    experiment_timestamp,
                                                    MEMORY,
                                                    NUM_GPU)
    call([cluster_command + experiment_command], shell=True)
