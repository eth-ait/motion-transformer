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

experiment_list = [
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875869-Seq2seq/config.json',
    # # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419538-Seq2seq-SPL/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875871-Seq2seq-dropout/config.json',
    # # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419548-Seq2seq-dropout-SPL/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1565875866-Seq2seq-sampling/config.json',
    # # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564415781-Seq2seq-sampling-SPL/config.json',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 10 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 20 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 30 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 40 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 50 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 60 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 70 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 80 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 90 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 100 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 110 ',
    'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419522-RNN/config.json --source_seq_len 120 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 10 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 20 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 30 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 40 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 50 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 60 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 70 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 80 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 90 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 100 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 110 ',
    # 'python spl/training.py --from_config /cluster/home/eaksan/motion-modelling-github/pretrained_configs/1564419527-RNN-SPL/config.json  --source_seq_len 120 ',
]

transformer_experiments = [
    'python spl/training.py --glog_comment "pp_relu-shared_emb_st-emb2x" '
    '--shared_spatial_layer --shared_temporal_layer --shared_embedding_layer '
    '--transformer_num_layers 8 --transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py --glog_comment "pp_relu-shared_st-emb2x" '
    '--shared_spatial_layer --shared_temporal_layer '
    '--transformer_num_layers 8 --transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py --glog_comment "pp_relu-shared_emb_blocks-emb2x" '
    '--shared_embedding_layer --shared_attention_block '
    '--transformer_num_layers 8 --transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py --glog_comment "pp_relu-shared_blocks-emb2x" '
    '--shared_attention_block '
    '--transformer_num_layers 8 --transformer_d_model 128 --transformer_dff 256 ',
    
    'python spl/training.py --glog_comment "pp_relu-shared_emb_blocks-res-emb2x" '
    '--shared_embedding_layer --shared_attention_block '
    '--transformer_num_heads_temporal 8 --transformer_num_heads_spacial 8 '
    '--transformer_num_layers 8 --transformer_d_model 128 --transformer_dff 256',
]

# Create a unique experiment timestamp.
experiment_timestamp = str(int(time.time()))
for work_id, experiment in enumerate(transformer_experiments):
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
