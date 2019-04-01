from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 8
MEMORY = 1024
NUM_GPU = 1
WALL_TIME = 4
cluster_command_format = 'bsub -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '

"""
--joint_prediction_model [plain, fk_joints]
--output_layer_size 64, --output_layer_number 1 -> if fk_joints
--output_layer_size 256, --output_layer_number 1 -> if plain and --model_type rnn
--output_layer_size 64, --output_layer_number 1 -> if plain and --model_type wavenet or stcn
--input_dropout_rate [0, 0.1]
--seq_length_in 50 --seq_length_out [10, 25]  (assuming that the data is downsampled.)

--action_loss none
--omit_one_hot if you are Emre.

Example command:
--input_dropout_rate 0.1 --learning_rate 0.001 --angle_loss joint_sum --joint_prediction_model fk_joints --output_layer_size 64, --output_layer_number 1 --batch_size 64 --model_type rnn  --seq_length_in 50 --seq_length_out 10 --residual_velocities --cell_type lstm
"""

best_configuration = [
    # wavenet 128x35
    'python amass_training.py --dynamic_validation_split --input_dropout_rate 0.1 '
    '--optimizer adam --learning_rate 0.0005 --angle_loss joint_sum '
    '--joint_prediction_model plain --batch_size 64 --model_type wavenet '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',

    # stcn 128x35
    'python amass_training.py --dynamic_validation_split --input_dropout_rate 0.1 '
    '--optimizer adam --learning_rate 0.0005 --angle_loss joint_sum '
    '--joint_prediction_model plain --batch_size 64 --model_type stcn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',

    # lstm 512X2, out_256, fk_joints
    'python amass_training.py --dynamic_validation_split '
    '--input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 256 --cell_type lstm --cell_size 512 --cell_layers 2 '
    '--joint_prediction_model fk_joints --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',

    # lstm 1024x1, out_64, fk_joints
    'python amass_training.py --dynamic_validation_split '
    '--input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --batch_size 64 --model_type rnn '
    '--architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60',
]

# --no_normalization --dynamic_validation_split --output_layer_size 64 --optimizer adam --input_dropout_rate 0.1 --learning_rate 0.001 --angle_loss joint_sum --joint_prediction_model plain --batch_size 16 --model_type wavenet --cell_type lstm --cell_size 1024 --cell_layers 1 --architecture tied --autoregressive_input supervised --seq_length_in 100 --seq_length_out 60
# --dynamic_validation_split --no_normalization --input_dropout_rate 0.1 --optimizer adam --learning_rate 0.0005 --angle_loss joint_sum --joint_prediction_model plain --residual_velocities --batch_size 64 --model_type wavenet --seq_length_in 100 --seq_length_out 60
# --experiment_name 400ms --dynamic_validation_split --input_dropout_rate 0.1 --optimizer sgd --learning_rate 0.005 --angle_loss all_mean --batch_size 16 --model_type seq2seq --autoregressive_input sampling_based --seq_length_in 100 --seq_length_out 24 --residual_velocities

experiment_list = [
    'python amass_training.py --dynamic_validation_split '
    '--input_dropout_rate 0.0 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --output_layer_number 1 '
    '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --batch_size 64 --model_type seq2seq --residual_velocities '
    '--seq_length_in 120 --seq_length_out 24 --max_gradient_norm 1 --use_aa --autoregressive_input sampling_based ',

    'python amass_training.py --dynamic_validation_split '
    '--input_dropout_rate 0.0 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --output_layer_number 1 '
    '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --batch_size 64 --model_type seq2seq --residual_velocities '
    '--seq_length_in 120 --seq_length_out 24 --max_gradient_norm 1 --use_aa --autoregressive_input supervised ',

    'python amass_training.py --dynamic_validation_split '
    '--input_dropout_rate 0.1 --optimizer adam --learning_rate 0.001 '
    '--angle_loss joint_sum --output_layer_size 64 --output_layer_number 1 '
    '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--joint_prediction_model fk_joints --batch_size 64 --model_type seq2seq --residual_velocities '
    '--seq_length_in 120 --seq_length_out 24 --max_gradient_norm 1 --use_aa --autoregressive_input supervised ',
]


for i, experiment in enumerate(experiment_list):
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
