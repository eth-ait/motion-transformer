from subprocess import call
import time

"""
The working directory must be the same with python file's directory.
"""

NUM_CPU = 8
MEMORY = 3000
NUM_GPU = 1
WALL_TIME = 23
# cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:00 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" -R "select[gpu_model0==GeForceGTX1080Ti]" '
cluster_command_format = 'bsub -G ls_hilli -n {} -W {}:50 -o log_{} -R "rusage[mem={}, ngpus_excl_p={}]" '


transformer_experiments = [
    'python spl/training.py '
    '--glog_comment "" '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_temp_spat_ffn" '
    '--shared_embedding_layer '
    '--shared_temporal_layer --shared_spatial_layer --shared_pw_ffn '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_temp_spat" '
    '--shared_embedding_layer '
    '--shared_temporal_layer --shared_spatial_layer '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_ffn" '
    '--shared_pw_ffn '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "rand100_mask01" '
    '--random_window_min 100 --temporal_mask_drop 0.1 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_block" '
    '--shared_attention_block '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 120 --transformer_num_layers 8 '
    '--transformer_d_model 128 --transformer_dff 256 ',
]

transformer_h36m = [
    'python spl/training.py '
    '--glog_comment "shared_emb_blocks" '
    '--shared_embedding_layer --shared_attention_block '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_blocks" '
    '--shared_embedding_layer --shared_attention_block '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 0 --temporal_mask_drop 0 '
    '--transformer_window_length 50 --transformer_num_layers 6 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 6 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_ffn_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--shared_pw_ffn '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_ffn_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--shared_pw_ffn '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 6 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_temp_spat_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--shared_temporal_layer --shared_spatial_layer '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "shared_emb_temp_spat_blocks-rand40_mask01" '
    '--shared_embedding_layer --shared_attention_block '
    '--shared_temporal_layer --shared_spatial_layer '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 6 '
    '--transformer_d_model 128 --transformer_dff 256 ',
        
    'python spl/training.py '
    '--glog_comment "shared_emb_temp_spat-rand40_mask01" '
    '--shared_embedding_layer '
    '--shared_temporal_layer --shared_spatial_layer '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py '
    '--glog_comment "rand40_mask01" '
    '--num_epochs 10000 --use_h36m --source_seq_len 50 --target_seq_len 10 '
    '--random_window_min 40 --temporal_mask_drop 0.1 '
    '--transformer_window_length 50 --transformer_num_layers 2 '
    '--transformer_d_model 128 --transformer_dff 256 ',
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
