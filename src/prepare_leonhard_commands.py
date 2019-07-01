def prepare_martinez():
    base_command = 'python martinez_translate.py --input_dropout_rate 0.0 --learning_rate {} --batch_size {} ' \
                   '--optimizer {} --angle_loss joint_sum --model_type aged --action_loss none ' \
                   '--joint_prediction_model {} --learning_rate_decay_rate 0.95 ' \
                   '--seq_length_in 50 --seq_length_out 10 --residual_velocities --cell_type gru ' \
                   '--autoregressive_input supervised --aged_input_layer_size 1024 ' \
                   '{} --experiment_name {}'

    lrs = [0.005]
    batch_sizes = [16, 64]
    aged_weight = [0.6, 0.0]
    joint_prediction = ['plain']
    optimizer = ['adam', 'sgd']

    all_commands = []

    for lr in lrs:
        for bs in batch_sizes:
            for d_weight in aged_weight:
                for jpr in joint_prediction:
                    for optim in optimizer:
                        exp_name = "supervised_idrop0_logloss_ming_lr{}_bs{}_d{}".format(lr, bs, d_weight)
                        if d_weight > 0.0:
                            aged = '--aged_adversarial --aged_d_weight {} --aged_log_loss --aged_min_g'.format(d_weight)
                        else:
                            aged = ''

                        # if jpr == 'plain':
                        #     out_size = 0
                        # else:
                        #     out_size = 1
                        c = base_command.format(lr, bs, optim, jpr, aged, exp_name)
                        print(c)
                        all_commands.append(c)

    return all_commands


def prepare_amass():
    base_command = 'python amass_training.py --input_dropout_rate 0.0 --learning_rate {} --batch_size {} ' \
                   '--optimizer {} --angle_loss joint_sum --model_type aged --output_layer_size 64 --output_layer_number {} ' \
                   '--joint_prediction_model {} --cell_type {} --cell_size 1024 --cell_layers 1 --test_every 500 ' \
                   '--seq_length_in 120 --seq_length_out 24 --residual_velocities --dynamic_validation_split ' \
                   '--autoregressive_input sampling_based --aged_input_layer_size 1024 {} --use_aa ' \
                   '--early_stopping_tolerance 50 '

    lrs = [0.0001]
    batch_sizes = [64, 128]
    aged_weight = [0.0]
    joint_prediction = ['plain']
    cell_type = ['gru']
    optimizer = ['adam']

    all_commands = []

    for lr in lrs:
        for bs in batch_sizes:
            for d_weight in aged_weight:
                for jpr in joint_prediction:
                    for ct in cell_type:
                        for optim in optimizer:
                            if d_weight > 0.0:
                                aged = '--aged_adversarial --aged_d_weight {} --aged_log_loss --aged_min_g'.format(d_weight)
                            else:
                                aged = ''

                            # if lr == 0.0001 and optim == 'adam':
                            #     continue

                            if jpr == 'plain':
                                out_size = 0
                            else:
                                out_size = 1
                            c = base_command.format(lr, bs, optim, out_size, jpr, ct, aged)
                            print(c)
                            all_commands.append(c)

    return all_commands


if __name__ == '__main__':
    # commands = prepare_martinez()
    commands = prepare_amass()
    print("{} commands".format(len(commands)))
    print(commands)


#