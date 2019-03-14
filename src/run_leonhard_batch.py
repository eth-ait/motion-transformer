from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 8
MEMORY = 4096
NUM_GPU = 1
WALL_TIME = 16
cluster_command_format = 'bsub -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '

experiment_list_seq2seq = [
    # seq2seq
    'python amass_training.py --experiment_name lr1e3_sgd --optimizer sgd --learning_rate 0.005 '
    '--angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input sampling_based '
    '--seq_length_in 100 --seq_length_out 60 --use_quat --no_normalization --dynamic_validation_split',

    # no velocity
    'python amass_training.py --experiment_name lr1e3_sgd --optimizer sgd --learning_rate 0.005 '
    '--angle_loss joint_sum --joint_prediction_model plain '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input sampling_based '
    '--seq_length_in 100 --seq_length_out 60 --use_quat --no_normalization --dynamic_validation_split',

    # memory error.
    'python amass_training.py --experiment_name lr1e3_sgd --optimizer sgd --learning_rate 0.005 '
    '--angle_loss joint_sum --joint_prediction_model plain --residual_velocities '
    '--batch_size 64 --model_type seq2seq --architecture tied --autoregressive_input supervised '
    '--seq_length_in 100 --seq_length_out 60 --use_quat --no_normalization --dynamic_validation_split']

experiment_list = [
    # gru 1024x1
    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --cell_type gru --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model separate_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60 '
    '--use_quat --no_normalization --dynamic_validation_split',

    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --cell_type gru --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60 '
    '--use_quat --no_normalization --dynamic_validation_split',

    # lstm 1024x1
    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model separate_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60 '
    '--use_quat --no_normalization --dynamic_validation_split',

    'python amass_training.py --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --residual_velocities --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60 '
    '--use_quat --no_normalization --dynamic_validation_split']

for experiment in experiment_list_seq2seq:
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

