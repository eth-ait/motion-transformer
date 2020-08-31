from subprocess import call
import time


"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 2
MEMORY = 10000
NUM_GPU = 1
WALL_TIME = 23
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceRTX2080Ti]" '
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:50 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '


transformer_experiments = [
    'python spl/training.py -seed 1237 '
    '--glog_comment "idrop01-full_len-temp_abs_rel-norm_all_var" '
    '--batch_size 16 --num_epochs 1000 --normalization_dim all '
    '--temp_abs_pos_encoding --temp_rel_pos_encoding '
    '--input_dropout_rate 0.1 --source_seq_len 120 --target_seq_len 24 '
    '--transformer_lr 1 --transformer_window_length 144 '
    '--model_type transformer2d --transformer_dropout_rate 0.1 '
    '--transformer_num_heads_spacial 8 --transformer_num_heads_temporal 8 '
    '--transformer_num_layers 8 --transformer_d_model 128 '
    '--transformer_dff 256 --data_type rotmat ',
    
    'python spl/training.py -seed 1237 '
    '--glog_comment "idrop01-full_len-temp_abs_rel-norm_all_var" '
    '--batch_size 16 --num_epochs 1000 --normalization_dim all '
    '--temp_abs_pos_encoding --temp_rel_pos_encoding '
    '--input_dropout_rate 0.1 --source_seq_len 120 --target_seq_len 60 '
    '--transformer_lr 1 --transformer_window_length 180 '
    '--model_type transformer2d --transformer_dropout_rate 0.1 '
    '--transformer_num_heads_spacial 8 --transformer_num_heads_temporal 8 '
    '--transformer_num_layers 8 --transformer_d_model 128 '
    '--transformer_dff 256 --data_type rotmat ',
    
    'python spl/training.py -seed 1237 '
    '--glog_comment "idrop01-full_len-norm_var" '
    '--batch_size 16 --num_epochs 1000 --normalization_dim channel '
    '--abs_pos_encoding  '
    '--input_dropout_rate 0.1 --source_seq_len 120 --target_seq_len 24 '
    '--transformer_lr 1 --transformer_window_length 144 '
    '--model_type transformer2d --transformer_dropout_rate 0.1 '
    '--transformer_num_heads_spacial 8 --transformer_num_heads_temporal 8 '
    '--transformer_num_layers 8 --transformer_d_model 128 '
    '--transformer_dff 256 --data_type rotmat ',
    
    'python spl/training.py -seed 1237 '
    '--glog_comment "idrop01-full_len-norm_var" '
    '--batch_size 16 --num_epochs 1000 --normalization_dim channel '
    '--abs_pos_encoding  '
    '--input_dropout_rate 0.1 --source_seq_len 120 --target_seq_len 60 '
    '--transformer_lr 1 --transformer_window_length 180 '
    '--model_type transformer2d --transformer_dropout_rate 0.1 '
    '--transformer_num_heads_spacial 8 --transformer_num_heads_temporal 8 '
    '--transformer_num_layers 8 --transformer_d_model 128 '
    '--transformer_dff 256 --data_type rotmat ',
    
    # 'python spl/training.py -seed 1237 '
    # '--glog_comment "idrop01-temp_abs_rel-shared_kv-norm_std" '
    # '--batch_size 16 --num_epochs 1000 --normalization_dim channel '
    # '--temp_abs_pos_encoding --temp_rel_pos_encoding '
    # '--input_dropout_rate 0.1 --source_seq_len 120 --target_seq_len 24 '
    # '--transformer_lr 1 --transformer_window_length 144 '
    # '--model_type transformer2d --transformer_dropout_rate 0.1 '
    # '--transformer_num_heads_spacial 8 --transformer_num_heads_temporal 8 '
    # '--transformer_num_layers 8 --transformer_d_model 128 '
    # '--transformer_dff 256 --data_type rotmat --shared_templ_kv ',
    ]

transformer_h36m = [
    'python spl/training.py '
    '--glog_comment "full_len-idrop01-temp_abs_rel-shared_kv-norm_all_std-w1000" '
    '--input_dropout_rate 0.1 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0.1 --transformer_lr 1 '
    '--temp_pos_encoding --temp_rel_pos_encoding --data_type rotmat '
    '--normalization_dim all --shared_templ_kv --warm_up_steps 1000 ',
    
    'python spl/training.py '
    '--glog_comment "full_len-idrop01-temp_abs_rel-norm_all_std-w1000" '
    '--input_dropout_rate 0.1 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0.1 --transformer_lr 1 '
    '--temp_pos_encoding --temp_rel_pos_encoding --data_type rotmat '
    '--normalization_dim all --warm_up_steps 1000 ',
    
    'python spl/training.py '
    '--glog_comment "full_len-idrop01-temp_abs_rel-norm_std-w1000" '
    '--input_dropout_rate 0.1 '
    '--num_epochs 15000 --use_h36m --source_seq_len 50 --target_seq_len 25 '
    '--transformer_window_length 75 --transformer_num_layers 4 '
    '--transformer_d_model 64 --transformer_dff 128 '
    '--transformer_num_heads_temporal 4 '
    '--transformer_num_heads_spacial 4 '
    '--transformer_dropout_rate 0.1 --transformer_lr 1 '
    '--temp_pos_encoding --temp_rel_pos_encoding --data_type rotmat '
    '--normalization_dim channel --warm_up_steps 1000 ',
    ]

reproducing_exp = [
    'python spl/training.py '
    '--from_config ./pretrained_configs/1573450146-transformer2d/config.json '
    '--glog_comment "f7fb419fc-std" ',
    
    'python spl/training.py '
    '--from_config ./pretrained_configs/1573450146-transformer2d/config.json '
    '--glog_comment "f7fb419fc-std" ',
    ]

# Create a unique experiment timestamp.
experiment_timestamp = str(int(time.time()))
for work_id, experiment in enumerate(reproducing_exp):
  experiment_id = "{}.{}".format(experiment_timestamp, work_id + 1)
  time.sleep(1)
  print(experiment_id)
  experiment_command = experiment + ' --new_experiment_id ' + experiment_id
  
  cluster_command = cluster_command_format.format(NUM_CPU,
                                                  WALL_TIME,
                                                  experiment_id,
                                                  MEMORY,
                                                  NUM_GPU)
  call([cluster_command + experiment_command], shell=True)
