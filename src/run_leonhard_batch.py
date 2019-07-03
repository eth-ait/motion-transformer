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

experiment_list = [
    # Exponential lr decay
    # # Seq2seq
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss all_mean --architecture tied --input_dropout_rate 0.0 '
    # '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model plain '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 960 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential '
    # '--learning_rate_decay_rate 0.98 ',
    #
    # # Seq2seq-SPL
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.0 '
    # '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model fk_joints '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 64 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential --experiment_name SPL '
    # '--learning_rate_decay_rate 0.98 ',
    #
    # # Seq2seq-dropout
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss all_mean --architecture tied --input_dropout_rate 0.1 '
    # '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model plain '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 960 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential --experiment_name dropout '
    # '--learning_rate_decay_rate 0.98 ',
    #
    # # Seq2seq-dropout-SPL
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.1 '
    # '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model fk_joints '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 64 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential --experiment_name dropout-SPL '
    # '--learning_rate_decay_rate 0.98 ',
    #
    # # Seq2seq-sampling
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss all_mean --architecture tied --input_dropout_rate 0.0 '
    # '--autoregressive_input sampling_based --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model plain '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 960 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential --experiment_name sampling_based '
    # '--learning_rate_decay_rate 0.98 ',
    #
    # # Seq2seq-sampling-SPL
    # 'python amass_training.py --dynamic_validation_split '
    # '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.0 '
    # '--autoregressive_input sampling_based --batch_size 64 --cell_type lstm '
    # '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    # '--model_type seq2seq --joint_prediction_model fk_joints '
    # '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    # '--output_layer_size 64 --output_layer_number 1 '
    # '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    # '--learning_rate_decay_type exponential --experiment_name sampling_based-SPL '
    # '--learning_rate_decay_rate 0.98 ',
    
    
    # Seq2seq
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss all_mean --architecture tied --input_dropout_rate 0.0 '
    '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model plain '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 960 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',
    #
    # Seq2seq-SPL
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.0 '
    '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model fk_joints '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 64 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--experiment_name SPL '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',
    #
    # Seq2seq-dropout
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss all_mean --architecture tied --input_dropout_rate 0.1 '
    '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model plain '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 960 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--experiment_name dropout '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',

    # Seq2seq-dropout-SPL
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.1 '
    '--autoregressive_input supervised --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model fk_joints '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 64 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--experiment_name dropout-SPL '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',

    # Seq2seq-sampling
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss all_mean --architecture tied --input_dropout_rate 0.0 '
    '--autoregressive_input sampling_based --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model plain '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 960 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--experiment_name sampling_based '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',

    # Seq2seq-sampling-SPL
    'python amass_training.py --dynamic_validation_split '
    '--angle_loss joint_sum --architecture tied --input_dropout_rate 0.0 '
    '--autoregressive_input sampling_based --batch_size 64 --cell_type lstm '
    '--cell_layers 1 --cell_size 1024 --max_gradient_norm 1.0 '
    '--model_type seq2seq --joint_prediction_model fk_joints '
    '--learning_rate 0.001 --optimizer adam --early_stopping_tolerance 40 '
    '--output_layer_size 64 --output_layer_number 1 '
    '--residual_velocities --seq_length_in 120 --seq_length_out 24 --use_aa '
    '--experiment_name sampling_based-SPL '
    '--learning_rate_decay_type piecewise '
    '--learning_rate_decay_rate 0.95 ',
    
]

experiment_list_h36m = [
    'python amass_training.py --input_dropout_rate 0.0 --learning_rate 1e-04 '
    '--batch_size 256 --optimizer adam --angle_loss joint_sum '
    '--model_type aged --joint_prediction_model fk_joints '
    '--output_layer_size 64 --output_layer_number 2 '
    '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--test_every 500 --seq_length_in 120 --seq_length_out 24 '
    '--residual_velocities --dynamic_validation_split '
    '--autoregressive_input sampling_based --early_stopping_tolerance 40 '
    '--aged_input_layer_size 1024 --use_aa',

    'python amass_training.py --input_dropout_rate 0.0 --learning_rate 1e-04 '
    '--batch_size 256 --optimizer adam --angle_loss joint_sum '
    '--model_type aged --joint_prediction_model fk_joints '
    '--output_layer_size 128 --output_layer_number 1 '
    '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    '--test_every 500 --seq_length_in 120 --seq_length_out 24 '
    '--residual_velocities --dynamic_validation_split '
    '--autoregressive_input sampling_based --early_stopping_tolerance 40 '
    '--aged_input_layer_size 1024 --use_aa',

    # 'python amass_training.py --input_dropout_rate 0.0 --learning_rate 5e-04 '
    # '--batch_size 128 --optimizer adam --angle_loss joint_sum '
    # '--model_type aged --joint_prediction_model fk_joints '
    # '--output_layer_size 64 --output_layer_number 2 '
    # '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    # '--test_every 500 --seq_length_in 120 --seq_length_out 24 '
    # '--residual_velocities --dynamic_validation_split '
    # '--autoregressive_input sampling_based --early_stopping_tolerance 40 '
    # '--aged_input_layer_size 1024 --use_aa',
    #
    # 'python amass_training.py --input_dropout_rate 0.0 --learning_rate 5e-04 '
    # '--batch_size 128 --optimizer adam --angle_loss joint_sum '
    # '--model_type aged --joint_prediction_model fk_joints '
    # '--output_layer_size 128 --output_layer_number 1 '
    # '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    # '--test_every 500 --seq_length_in 120 --seq_length_out 24 '
    # '--residual_velocities --dynamic_validation_split '
    # '--autoregressive_input sampling_based --early_stopping_tolerance 40 '
    # '--aged_input_layer_size 1024 --use_aa',
    #
    # 'python amass_training.py --input_dropout_rate 0.1 --learning_rate 1e-03 '
    # '--batch_size 64 --optimizer adam --angle_loss joint_sum '
    # '--model_type rnn --joint_prediction_model fk_joints '
    # '--output_layer_size 64 --output_layer_number 1 '
    # '--cell_type lstm --cell_size 1024 --cell_layers 1 '
    # '--test_every 500 --seq_length_in 120 --seq_length_out 24 '
    # '--residual_velocities --dynamic_validation_split '
    # '--early_stopping_tolerance 50 ',
]

for i, experiment in enumerate(experiment_list_h36m):
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
