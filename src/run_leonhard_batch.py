from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 8
MEMORY = 3072
NUM_GPU = 1
WALL_TIME = 16
cluster_command_format = 'bsub -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '

experiment_list = [
    'python amass_training.py --experiment_name lr1e3_sgd --optimizer sgd --learning_rate 0.001 '
    '--action_loss none --angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input sampling_based '
    '--seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --experiment_name lr1e3_sgd --optimizer sgd --learning_rate 0.001 '
    '--angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input supervised '
    '--seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --experiment_name lr1e3_adam --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input sampling_based '
    '--seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --experiment_name lr1e3_adam --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input supervised '
    '--seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 256 '
    '--joint_prediction_model plain --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 '
    '--joint_prediction_model separate_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',

    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 '
    '--joint_prediction_model fk_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60']

for experiment in experiment_list:
    # Create a unique experiment timestamp.
    experiment_timestamp = str(int(time.time()))
    experiment_command = experiment + ' --new_experiment_id ' + experiment_timestamp

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    experiment_timestamp,
                                                    MEMORY,
                                                    NUM_GPU)
    print(experiment_timestamp)
    call([cluster_command + experiment_command], shell=True)
    # Make sure that we get unique timestamps :)
    time.sleep(1)

