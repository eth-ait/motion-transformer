# create leonhard commands

# base_command = 'python translate.py --dynamic_validation_split --input_dropout_rate {} --learning_rate {} ' \
#                '--angle_loss joint_sum --output_layer_number 1 --output_layer_size {} ' \
#                '--model_type rnn --action_loss none --joint_prediction_model {} ' \
#                '--seq_length_in 50 --seq_length_out 10 --residual_velocities --cell_type {} ' \
#                '--experiment_name {} --optimizer {} {} ' \
#                '--data_dir /cluster/work/hilliges/kamanuel/data/motion-modelling/h3.6m/dataset ' \
#                '--train_dir /cluster/work/hilliges/kamanuel/trained_models/motion-modelling/experiments_h36m '

base_command = 'python martinez_translate.py --input_dropout_rate 0.0 --learning_rate {} --batch_size {} ' \
               '--optimizer adam --angle_loss joint_sum --model_type aged --action_loss none ' \
               '--joint_prediction_model {} --learning_rate_decay_rate 0.95 ' \
               '--seq_length_in 50 --seq_length_out 10 --residual_velocities --cell_type gru ' \
               '--autoregressive_input sampling_based --aged_input_layer_size 1024 ' \
               '--aged_adversarial --aged_d_weight {} --aged_log_loss --aged_min_g --experiment_name {}'


# base_command = 'python amass_training.py --input_dropout_rate 0.1 --learning_rate {} --batch_size {} ' \
#                '--optimizer adam --angle_loss joint_sum --model_type aged --output_layer_size 64 --output_layer_number {} ' \
#                '--joint_prediction_model {} --cell_type gru --cell_size 1024 --cell_layers 1 --test_every 500 ' \
#                '--seq_length_in 120 --seq_length_out 24 --residual_velocities --dynamic_validation_split ' \
#                '--autoregressive_input sampling_based --aged_input_layer_size 1024 {} --use_aa'


lrs = [0.001, 0.0001, 0.0005]
batch_sizes = [64, 128]
aged_weight = [0.6, 1.0]
joint_prediction = ['fk_joints']

all_commands = []

for lr in lrs:
    for bs in batch_sizes:
        for d_weight in aged_weight:
            for jpr in joint_prediction:
                exp_name = "spl_logloss_ming_lr{}_bs{}_d{}".format(lr, bs, d_weight)
                c = base_command.format(lr, bs, jpr, d_weight, exp_name)
                print(c)
                all_commands.append(c)

print("{} commands".format(len(all_commands)))
print(all_commands)


#