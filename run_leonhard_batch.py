from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 2
MEMORY = 8000
NUM_GPU = 1
WALL_TIME = 23
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceGTX1080Ti]" '
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:50 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '


transformer_experiments = [
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 1 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 3 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 4 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 5 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 6 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 7 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 9 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "ablation_nlayers" '
    '--transformer_window_length 120 --transformer_num_layers 10 '
    '--transformer_d_model 128 --transformer_dff 256 ',
]

transformer_h36m = [
    'python spl/training.py '
    '--glog_comment "full_len-idrop01-tr_lr" '
    '--input_dropout_rate 0.1 '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0.1 --transformer_lr 1 ',
    
    'python spl/training.py '
    '--glog_comment "full_len-idrop02-tr_lr" '
    '--input_dropout_rate 0.2 '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0.1 --transformer_lr 1 ',
    
    'python spl/training.py '
    '--glog_comment "full_len-idrop01-tr_drop0-tr_lr" '
    '--input_dropout_rate 0.1 '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0 --transformer_lr 1 ',
    
    'python spl/training.py '
    '--glog_comment "full_len-idrop02-tr_drop0-tr_lr" '
    '--input_dropout_rate 0.2 '
    '--random_window_min 0 --temporal_mask_drop 0 --batch_size 32 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0 --transformer_lr 1 ',
    
    # 'python spl/training.py '
    # '--glog_comment "idrop02-clip1" '
    # '--input_dropout_rate 0.2 '
    # '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    # '--data_type rotmat --batch_size 64 '
    # '--model_type rnn --cell_type gru '
    # '--joint_prediction_layer spl ',
    #
    # 'python spl/training.py '
    # '--glog_comment "idrop02-clip1" '
    # '--input_dropout_rate 0.2 '
    # '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    # '--data_type rotmat --batch_size 100 '
    # '--model_type rnn --cell_type gru '
    # '--joint_prediction_layer spl ',
    #
    # 'python spl/training.py '
    # '--glog_comment "idrop01" '
    # '--input_dropout_rate 0.1 '
    # '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    # '--data_type rotmat --batch_size 64 '
    # '--model_type rnn --cell_type gru '
    # '--joint_prediction_layer spl_sparse ',
]

# Create a unique experiment timestamp.
experiment_timestamp = str(int(time.time()))
for work_id, experiment in enumerate(transformer_h36m):
    experiment_id = "{}.{}".format(experiment_timestamp, work_id+1)
    time.sleep(1)
    print(experiment_id)
    experiment_command = experiment + ' --new_experiment_id ' + experiment_id

    cluster_command = cluster_command_format.format(NUM_CPU,
                                                    WALL_TIME,
                                                    experiment_id,
                                                    MEMORY,
                                                    NUM_GPU)
    call([cluster_command + experiment_command], shell=True)
