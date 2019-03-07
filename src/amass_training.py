from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import amass_models as models
from amass_tf_data import TFRecordMotionDataset

# ETH imports
from constants import Constants as C
import glob
import json

# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.95, "Learning rate mutiplier. 1 means no decay.")
tf.app.flags.DEFINE_string("learning_rate_decay_type", "piecewise", "Learning rate decay type.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
# Directories
tf.app.flags.DEFINE_string("meta_data_path", "../data/amass/stats.npz", "Path to meta-data file.")
tf.app.flags.DEFINE_string("train_data_path", "../data/amass/training/amass-?????-of-?????", "Path to train data folder.")
tf.app.flags.DEFINE_string("valid_data_path", "../data/amass/validation/amass-?????-of-?????", "Path to valid data folder.")
tf.app.flags.DEFINE_string("test_data_path", None, "Path to test data folder.")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("../experiments_amass/"), "Training directory.")

tf.app.flags.DEFINE_string("autoregressive_input", "sampling_based", "The type of decoder inputs, supervised or sampling_based")
tf.app.flags.DEFINE_integer("print_every", 100, "How often to log training error.")
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 2000, "How often to save the model.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_string("experiment_name", None, "A descriptive name for the experiment.")
tf.app.flags.DEFINE_string("experiment_id", None, "Unique experiment timestamp to load a pre-trained model.")
tf.app.flags.DEFINE_string("model_type", "seq2seq", "Model type: seq2seq, seq2seq_feedback, wavenet, stcn, structured_stcn or vrnn")
tf.app.flags.DEFINE_boolean("feed_error_to_encoder", True, "If architecture is not tied, can choose to feed error in encoder or not")
tf.app.flags.DEFINE_boolean("new_preprocessing", True, "Only discard entire joints not single DOFs per joint")
tf.app.flags.DEFINE_string("joint_prediction_model", "plain", "plain, separate_joints or fk_joints.")
tf.app.flags.DEFINE_string("angle_loss", "joint_sum", "joint_sum, joint_mean or all_mean.")

args = tf.app.flags.FLAGS

# Unique timestamp to distinguish experiments with the same configuration.
experiment_timestamp = str(int(time.time()))


def create_model(session):
    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    if args.model_type == "seq2seq":
        model_cls, config, experiment_name = get_seq2seq_config(args)
    elif args.model_type == "stcn":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "wavenet":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "seq2seq_feedback":
        model_cls, config, experiment_name = get_seq2seq_config(args)
    elif args.model_type == "structured_stcn":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "vrnn":
        model_cls, config, experiment_name = get_rnn_config(args)
    else:
        raise Exception("Unknown model type.")

    with tf.name_scope("training_data"):
        windows_length = args.seq_length_in + args.seq_length_out
        train_data = TFRecordMotionDataset(data_path=args.train_data_path,
                                           meta_data_path=args.meta_data_path,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_epochs=args.num_epochs,
                                           extract_windows_of=windows_length,
                                           num_parallel_calls=16)
        train_pl = train_data.get_tf_samples()

    with tf.name_scope("validation_data"):
        eval_data = TFRecordMotionDataset(data_path=args.valid_data_path,
                                          meta_data_path=args.meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_epochs=args.num_epochs,
                                          extract_windows_of=windows_length,
                                          num_parallel_calls=16)
        eval_pl = eval_data.get_tf_samples()

    with tf.name_scope(C.TRAIN):
        train_model = model_cls(
            config=config,
            data_pl=train_pl,
            mode=C.TRAIN,
            reuse=False,
            dtype=tf.float32)
        train_model.build_graph()

    with tf.name_scope(C.SAMPLE):
        eval_model = model_cls(
            config=config,
            data_pl=eval_pl,
            mode=C.SAMPLE,
            reuse=True,
            dtype=tf.float32)
        eval_model.build_graph()

    num_param = 0
    for v in tf.global_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))
    config["num_parameters"] = num_param

    if args.experiment_id is None:
        experiment_dir = os.path.normpath(os.path.join(args.train_dir, experiment_name))
    else:
        experiment_dir = glob.glob(os.path.join(args.train_dir, args.experiment_id + "-*"), recursive=False)[0]
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)
    print("Experiment directory " + experiment_dir)

    train_model.optimization_routines()
    train_model.summary_routines()
    eval_model.summary_routines()

    # Create saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    if args.experiment_id is None:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return train_model, eval_model, train_data, eval_data, saver, global_step, experiment_dir

    # Load a pre-trained model.
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")
    print("Experiment directory: ", experiment_dir)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if args.load > 0:
            if os.path.isfile(os.path.join(experiment_dir, "checkpoint-{0}.index".format(args.load))):
                ckpt_name = os.path.normpath(
                    os.path.join(os.path.join(experiment_dir, "checkpoint-{0}".format(args.load))))
            else:
                raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(args.load))
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading model {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
        return train_model, eval_model, train_data, eval_data, saver, global_step, experiment_dir
    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def get_rnn_config(args):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['seed'] = 1234
    config['learning_rate'] = 5e-4
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_type'] = 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['cell'] = dict()
    config['cell']['kld_weight'] = dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
    config['cell']['type'] = C.LATENT_GAUSSIAN
    config['cell']['latent_size'] = 64
    config['cell']["hidden_activation_fn"] = C.RELU
    config['cell']["num_hidden_units"] = 256
    config['cell']["num_hidden_layers"] = 2
    config['cell']['latent_sigma_threshold'] = 5.0
    config['cell']['cell_type'] = C.LSTM
    config['cell']['cell_size'] = 512
    config['cell']['cell_num_layers'] = 1
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = 0
    config['input_layer']['num_layers'] = 1
    config['input_layer']['size'] = 256
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 2
    config['output_layer']['size'] = 256
    config['output_layer']['activation_fn'] = C.RELU

    config['grad_clip_by_norm'] = 1
    config['loss_on_encoder_outputs'] = True
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss

    if args.model_type == "vrnn":
        model_cls = models.RNNLatentCellModel
    else:
        raise Exception()

    experiment_name_format = "{}-{}{}-{}-{}-b{}-l{}_{}@{}{}-in{}_out{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    config['batch_size'],
                                                    config['cell']['latent_size'],
                                                    config['cell']['cell_size'],
                                                    config['cell']['cell_type'],
                                                    '-residual_vel' if args.residual_velocities else '',
                                                    args.seq_length_in,
                                                    args.seq_length_out)
    return model_cls, config, experiment_name


def get_stcn_config(args):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['seed'] = 1234
    config['learning_rate'] = 1e-3
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_type'] = 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['latent_layer'] = dict()
    config['latent_layer']['kld_weight'] = dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
    config['latent_layer']['latent_size'] = [256, 128, 64, 32, 16, 8, 4]
    config['latent_layer']['type'] = C.LATENT_LADDER_GAUSSIAN
    config['latent_layer']['layer_structure'] = C.LAYER_CONV1
    config['latent_layer']["hidden_activation_fn"] = C.RELU
    config['latent_layer']["num_hidden_units"] = 128
    config['latent_layer']["num_hidden_layers"] = 1
    config['latent_layer']['vertical_dilation'] = 4
    config['latent_layer']['use_fixed_pz1'] = False
    config['latent_layer']['use_same_q_sample'] = False
    config['latent_layer']['dynamic_prior'] = True
    config['latent_layer']['precision_weighted_update'] = True
    config['latent_layer']['recursive_q'] = True
    config['latent_layer']["top_down_latents"] = True
    config['latent_layer']['dense_z'] = True
    config['latent_layer']['latent_sigma_threshold'] = 5.0
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = 0
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 2
    config['output_layer']['size'] = 256
    config['output_layer']['type'] = C.LAYER_TCN
    config['output_layer']['filter_size'] = 2
    config['output_layer']['activation_fn'] = C.RELU
    config['cnn_layer'] = dict()
    config['cnn_layer']['num_encoder_layers'] = 28
    config['cnn_layer']['num_decoder_layers'] = 0
    config['cnn_layer']['num_filters'] = 128
    config['cnn_layer']['filter_size'] = 2
    config['cnn_layer']['dilation_size'] = [1, 2, 4, 8]*7
    config['cnn_layer']['activation_fn'] = C.RELU
    config['cnn_layer']['use_residual'] = True
    config['cnn_layer']['zero_padding'] = True
    config['decoder_use_enc_skip'] = False
    config['decoder_use_enc_last'] = False
    config['decoder_use_raw_inputs'] = False
    config['grad_clip_by_norm'] = 1
    config['use_future_steps_in_q'] = False
    config['loss_on_encoder_outputs'] = True

    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss

    if args.model_type == "stcn":
        model_cls = models.STCN
    elif args.model_type == "wavenet":
        model_cls = models.Wavenet
        if not(config['decoder_use_enc_skip'] or config['decoder_use_enc_last'] or config['decoder_use_raw_inputs']):
            config['decoder_use_enc_last'] = True
        del config["latent_layer"]
    elif args.model_type == "structured_stcn":
        model_cls = models.StructuredSTCN
    else:
        raise Exception()

    experiment_name_format = "{}-{}{}-{}-{}-b{}-{}x{}@{}{}-in{}_out{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    config['batch_size'],
                                                    config['cnn_layer']['num_encoder_layers'],
                                                    config['cnn_layer']['num_filters'],
                                                    config['cnn_layer']['filter_size'],
                                                    '-residual_vel' if args.residual_velocities else '',
                                                    args.seq_length_in,
                                                    args.seq_length_out)
    return model_cls, config, experiment_name


def get_seq2seq_config(args):
    """Create translation model and initialize or load parameters in session."""

    config = dict()
    config['seed'] = 1234
    config['loss_on_encoder_outputs'] = False  # Only valid for Wavenet variants.
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model  # "plain", "separate_joints", "fk_joints"
    config['architecture'] = args.architecture
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['rnn_size'] = args.size
    config['num_layers'] = args.num_layers
    config['grad_clip_by_norm'] = args.max_gradient_norm
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['learning_rate_decay_rate'] = args.learning_rate_decay_rate
    config['learning_rate_decay_type'] = args.learning_rate_decay_type
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model  # currently ignored by seq2seq models
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 0
    config['output_layer']['size'] = 128
    config['output_layer']['activation_fn'] = C.RELU
    config['angle_loss_type'] = args.angle_loss

    if args.model_type == "seq2seq":
        model_cls = models.Seq2SeqModel
    elif args.model_type == "seq2seq_feedback":
        model_cls = models.Seq2SeqFeedbackModel
        config['feed_error_to_encoder'] = args.feed_error_to_encoder
    else:
        raise ValueError("'{}' model unknown".format(args.model_type))

    experiment_name_format = "{}-{}-{}-{}-b{}-in{}_out{}-{}-enc{}feed-{}-depth{}-size{}-{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "" if args.experiment_name is None else args.experiment_name,
                                                    config['angle_loss_type'],
                                                    config['batch_size'],
                                                    args.seq_length_in,
                                                    args.seq_length_out,
                                                    args.architecture,
                                                    '' if args.feed_error_to_encoder else 'no',
                                                    config['autoregressive_input'],
                                                    args.num_layers,
                                                    args.size,
                                                    'residual_vel' if args.residual_velocities else 'not_residual_vel')
    return model_cls, config, experiment_name


def train():
    """Train a seq2seq model on human motion"""
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    device_count = {"GPU": 0} if args.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        # Create the model
        train_model, eval_model, train_data, eval_data, saver, global_step, experiment_dir = create_model(sess)

        # Summary writers for train and test runs
        summaries_dir = os.path.normpath(os.path.join(experiment_dir, "log"))
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        test_writer = train_writer
        print("Model created")

        # Training loop
        print("Running Training Loop.")
        time_counter = 0.0
        stop_signal = False
        step = 1
        epoch = 0
        train_loss = 0.0
        eval_loss = 0.0
        train_iter = train_data.get_iterator()
        eval_iter = eval_data.get_iterator()

        # Assuming that we use initializable iterators.
        sess.run(train_iter.initializer)
        sess.run(eval_iter.initializer)
        while True:
            if stop_signal:
                break

            # Training.
            for i in range(args.test_every):
                try:
                    start_time = time.perf_counter()
                    step += 1

                    step_loss, summary, _ = train_model.step(sess)
                    train_writer.add_summary(summary, step)
                    train_loss += step_loss

                    time_counter += (time.perf_counter() - start_time)
                    if step % args.print_every == 0:
                        train_loss_avg = train_loss / args.print_every
                        train_loss = 0
                        time_elapsed = time_counter/args.print_every
                        time_counter = 0
                        print("Train [{:04d}] \t Loss: {:.3f} \t time/batch = {:.3f}".format(step,
                                                                                             train_loss_avg,
                                                                                             time_elapsed))
                    # Learning rate decay
                    if step % args.learning_rate_decay_steps == 0 and train_model.learning_rate_decay_type == "piecewise":
                        sess.run(train_model.learning_rate_scheduler)

                    # Save the model
                    if step % args.save_every == 0:
                        print("Saving the model to {} ...".format(experiment_dir))
                        saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=step)

                except tf.errors.OutOfRangeError:
                    sess.run(train_iter.initializer)
                    epoch += 1
                    if epoch >= args.num_epochs:
                        stop_signal = True
                        print("End of Training.")
                        break

            # Evaluation: make a full pass on the evaluation data.
            eval_step = 0
            eval_loss = 0
            try:
                while True:
                    prediction, targets, seed_sequence = eval_model.sampled_step(sess)
                    step_loss = np.mean(np.square(prediction - targets))  # Dummy loss calculation.
                    eval_loss += step_loss
                    eval_step += 1
            except tf.errors.OutOfRangeError:
                # test_writer.add_summary(loss_summary, step)  # TODO Accumulate evaluation error.
                sess.run(eval_iter.initializer)
                eval_loss_avg = eval_loss / eval_step
                print("Eval [{:04d}] \t Loss: {:.3f}".format(step - 1, eval_loss_avg))


def sample():
    raise Exception("Not implemented.")


def main(_):
    if args.sample:
        sample()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
