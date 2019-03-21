"""Simple code for training an RNN for motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import h5py
import copy

import numpy as np
import tensorflow as tf

import data_utils
import cv2
import models

# ETH imports
from constants import Constants as C
import glob
import json
from logger import GoogleSheetLogger


# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.95, "Learning rate mutiplier. 1 means no decay.")
tf.app.flags.DEFINE_string("learning_rate_decay_type", "piecewise", "Learning rate decay type.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", 100000, "Iterations to train for.")
tf.app.flags.DEFINE_integer("early_stopping_tolerance", 20, "# of waiting steps until the validation loss improves.")
tf.app.flags.DEFINE_string("optimizer", "adam", "Optimization algorithm: adam or sgd.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("omit_one_hot", False, "Whether to remove one-hot encoding from the data")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
tf.app.flags.DEFINE_float("input_dropout_rate", 0.1, "Dropout rate on the model inputs.")
tf.app.flags.DEFINE_integer("output_layer_size", 64, "Number of units in the output layer.")
tf.app.flags.DEFINE_integer("output_layer_number", 1, "Number of output layer.")
tf.app.flags.DEFINE_string("cell_type", C.LSTM, "RNN cell type: gru, lstm, layernormbasiclstmcell")
tf.app.flags.DEFINE_integer("cell_size", 1024, "RNN cell size.")
tf.app.flags.DEFINE_integer("cell_layers", 1, "Number of cells in the RNN model.")
# Directories
tf.app.flags.DEFINE_string("data_dir", os.path.normpath("../data/h3.6m/dataset"), "Data directory")
tf.app.flags.DEFINE_string("train_dir", os.path.normpath("../experiments_h36m/"), "Training directory.")

tf.app.flags.DEFINE_string("action", "all", "The action to train on. all actions")
tf.app.flags.DEFINE_string("autoregressive_input", "sampling_based", "The type of decoder inputs, supervised or sampling_based")
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 2000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_string("experiment_name", None, "A descriptive name for the experiment.")
tf.app.flags.DEFINE_string("experiment_id", None, "Unique experiment timestamp to load a pre-trained model.")
tf.app.flags.DEFINE_string("model_type", "seq2seq", "Model type: seq2seq, seq2seq_feedback, wavenet, stcn, structured_stcn or rnn")
tf.app.flags.DEFINE_boolean("feed_error_to_encoder", True, "If architecture is not tied, can choose to feed error in encoder or not")
tf.app.flags.DEFINE_boolean("new_preprocessing", True, "Only discard entire joints not single DOFs per joint")
tf.app.flags.DEFINE_string("joint_prediction_model", "plain", "plain, separate_joints or fk_joints.")
tf.app.flags.DEFINE_boolean("use_sparse_fk_joints", False, "Sparse or dense fk_joints.")
tf.app.flags.DEFINE_string("angle_loss", "joint_sum", "joint_sum, joint_mean or all_mean.")
tf.app.flags.DEFINE_string("action_loss", "none", "cross_entropy, l2 or none.")
tf.app.flags.DEFINE_boolean("use_rotmat", False, "Convert everything to rotation matrices.")
tf.app.flags.DEFINE_boolean("force_valid_rot", False, "Forces a rotation matrix to be valid before feeding it back to the model")  # TODO(kamanuel) implement this for all models

FLAGS = tf.app.flags.FLAGS

# Unique timestamp to distinguish experiments with the same configuration.
experiment_timestamp = str(int(time.time()))


def create_model(session, actions, sampling=False):
    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    if FLAGS.model_type == "seq2seq":
        model_cls, config, experiment_name = create_seq2seq_model(actions, sampling)
    elif FLAGS.model_type == "stcn":
        model_cls, config, experiment_name = create_stcn_model(actions, sampling)
    elif FLAGS.model_type == "wavenet":
        model_cls, config, experiment_name = create_stcn_model(actions, sampling)
    elif FLAGS.model_type == "seq2seq_feedback":
        model_cls, config, experiment_name = create_seq2seq_model(actions, sampling)
    elif FLAGS.model_type == "structured_stcn":
        model_cls, config, experiment_name = create_stcn_model(actions, sampling)
    elif FLAGS.model_type == "rnn":
        model_cls, config, experiment_name = create_rnn_model(actions, sampling)
    else:
        raise Exception("Unknown model type.")

    with tf.name_scope(C.TRAIN):
        train_model = model_cls(
            config=config,
            session=session,
            mode=C.TRAIN,
            reuse=False,
            dtype=tf.float32)
        train_model.build_graph()

    with tf.name_scope(C.SAMPLE):
        eval_model = model_cls(
            config=config,
            session=session,
            mode=C.SAMPLE,
            reuse=True,
            dtype=tf.float32)
        eval_model.build_graph()

    experiment_name += "{}{}".format("-rotmat" if FLAGS.use_rotmat else "",
                                     "-force_mat" if FLAGS.force_valid_rot else "")

    num_param = 0
    for v in tf.global_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(int(num_param)))
    config["num_parameters"] = num_param

    if FLAGS.experiment_id is None:
        experiment_dir = os.path.normpath(os.path.join(FLAGS.train_dir, experiment_name))
    else:
        experiment_dir = glob.glob(os.path.join(FLAGS.train_dir, FLAGS.experiment_id + "-*"), recursive=False)[0]
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    if not sampling:
        json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)
    print("Experiment directory: ", experiment_dir)

    train_model.optimization_routines()
    train_model.summary_routines()
    eval_model.summary_routines()

    # Create saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    if FLAGS.experiment_id is None:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return train_model, eval_model, saver, global_step, experiment_dir

    # Load a pre-trained model.
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")
    print("Experiment directory: ", experiment_dir)

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        if FLAGS.load > 0:
            if os.path.isfile(os.path.join(experiment_dir, "checkpoint-{0}.index".format(FLAGS.load))):
                ckpt_name = os.path.normpath(
                    os.path.join(os.path.join(experiment_dir, "checkpoint-{0}".format(FLAGS.load))))
            else:
                raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
        else:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print("Loading model {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
        return train_model, eval_model, saver, global_step, experiment_dir
    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def create_rnn_model(actions, sampling=False):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['seed'] = 1234
    config['learning_rate'] = 1e-3
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_type'] = 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['cell'] = dict()
    config['cell']['cell_type'] = FLAGS.cell_type
    config['cell']['cell_size'] = FLAGS.cell_size
    config['cell']['cell_num_layers'] = FLAGS.cell_layers
    if FLAGS.model_type == 'vrnn':
        config['cell']['kld_weight'] = 1  # dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
        config['cell']['type'] = C.LATENT_GAUSSIAN
        config['cell']['latent_size'] = 64
        config['cell']["hidden_activation_fn"] = C.RELU
        config['cell']["num_hidden_units"] = 256
        config['cell']["num_hidden_layers"] = 1
        config['cell']['latent_sigma_threshold'] = 5.0
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = FLAGS.input_dropout_rate
    config['input_layer']['num_layers'] = 1
    config['input_layer']['size'] = 256
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = FLAGS.output_layer_number
    config['output_layer']['size'] = FLAGS.output_layer_size
    config['output_layer']['activation_fn'] = C.RELU

    config['grad_clip_by_norm'] = 1
    config['loss_on_encoder_outputs'] = True
    config['source_seq_len'] = FLAGS.seq_length_in
    config['target_seq_len'] = FLAGS.seq_length_out
    config['batch_size'] = FLAGS.batch_size
    config['autoregressive_input'] = FLAGS.autoregressive_input if not sampling else "sampling_based",
    config['number_of_actions'] = 0 if FLAGS.omit_one_hot else len(actions)
    config['one_hot'] = not FLAGS.omit_one_hot
    config['residual_velocities'] = FLAGS.residual_velocities
    config['joint_prediction_model'] = FLAGS.joint_prediction_model
    config['use_sparse_fk_joints'] = FLAGS.use_sparse_fk_joints
    config['angle_loss_type'] = FLAGS.angle_loss
    config['action_loss_type'] = FLAGS.action_loss
    config['rep'] = "rot_mat" if FLAGS.use_rotmat else "aa"

    if FLAGS.model_type == "rnn":
        model_cls = models.RNN
    else:
        raise Exception()

    input_dropout = config['input_layer'].get('dropout_rate', 0)
    experiment_name_format = "{}-{}{}-{}-{}{}-b{}-{}@{}{}-in{}_out{}-{}-{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    FLAGS.model_type,
                                                    "-"+FLAGS.experiment_name if FLAGS.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    "sparse_" + config['joint_prediction_model'] if FLAGS.use_sparse_fk_joints and config['joint_prediction_model'] == "fk_joints" else config['joint_prediction_model'],
                                                    "-idrop_" + str(input_dropout) if input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    config['cell']['cell_size'],
                                                    config['cell']['cell_type'],
                                                    '-residual_vel' if FLAGS.residual_velocities else '',
                                                    FLAGS.seq_length_in,
                                                    FLAGS.seq_length_out,
                                                    'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
                                                    config['action_loss_type'])
    return model_cls, config, experiment_name


def create_stcn_model(actions, sampling=False):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['seed'] = 1234
    config['learning_rate'] = FLAGS.learning_rate
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_type'] = 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['latent_layer'] = dict()
    config['latent_layer']['kld_weight'] = dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
    config['latent_layer']['latent_size'] = [64, 32, 16, 8, 4, 2, 1]
    config['latent_layer']['type'] = C.LATENT_LADDER_GAUSSIAN
    config['latent_layer']['layer_structure'] = C.LAYER_CONV1
    config['latent_layer']["hidden_activation_fn"] = C.RELU
    config['latent_layer']["num_hidden_units"] = 128
    config['latent_layer']["num_hidden_layers"] = 2
    config['latent_layer']['vertical_dilation'] = 5
    config['latent_layer']['use_fixed_pz1'] = False
    config['latent_layer']['use_same_q_sample'] = False
    config['latent_layer']['dynamic_prior'] = True
    config['latent_layer']['precision_weighted_update'] = True
    config['latent_layer']['recursive_q'] = True
    config['latent_layer']["top_down_latents"] = True
    config['latent_layer']['dense_z'] = True
    config['latent_layer']['latent_sigma_threshold'] = 5.0
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = FLAGS.input_dropout_rate
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = FLAGS.output_layer_number
    config['output_layer']['size'] = FLAGS.output_layer_size
    config['output_layer']['type'] = C.LAYER_TCN
    config['output_layer']['filter_size'] = 2
    config['output_layer']['activation_fn'] = C.RELU
    config['cnn_layer'] = dict()
    config['cnn_layer']['num_encoder_layers'] = 35
    config['cnn_layer']['num_decoder_layers'] = 0
    config['cnn_layer']['num_filters'] = 128
    config['cnn_layer']['filter_size'] = 2
    config['cnn_layer']['dilation_size'] = [1, 2, 4, 8, 16]*7
    config['cnn_layer']['activation_fn'] = C.RELU
    config['cnn_layer']['use_residual'] = True
    config['cnn_layer']['zero_padding'] = True
    config['decoder_use_enc_skip'] = False
    config['decoder_use_enc_last'] = False
    config['decoder_use_raw_inputs'] = False
    config['grad_clip_by_norm'] = 1
    config['use_future_steps_in_q'] = False
    config['loss_on_encoder_outputs'] = True

    config['source_seq_len'] = FLAGS.seq_length_in
    config['target_seq_len'] = FLAGS.seq_length_out
    config['batch_size'] = FLAGS.batch_size
    config['autoregressive_input'] = FLAGS.autoregressive_input if not sampling else "sampling_based",
    config['number_of_actions'] = 0 if FLAGS.omit_one_hot else len(actions)
    config['one_hot'] = not FLAGS.omit_one_hot
    config['residual_velocities'] = FLAGS.residual_velocities
    config['joint_prediction_model'] = FLAGS.joint_prediction_model
    config['use_sparse_fk_joints'] = FLAGS.use_sparse_fk_joints
    config['angle_loss_type'] = FLAGS.angle_loss
    config['action_loss_type'] = FLAGS.action_loss
    config['rep'] = "rot_mat" if FLAGS.use_rotmat else "aa"

    if FLAGS.model_type == "stcn":
        model_cls = models.STCN
    elif FLAGS.model_type == "wavenet":
        model_cls = models.Wavenet
        if not(config['decoder_use_enc_skip'] or config['decoder_use_enc_last'] or config['decoder_use_raw_inputs']):
            config['decoder_use_enc_last'] = True
        del config["latent_layer"]
    elif FLAGS.model_type == "structured_stcn":
        model_cls = models.StructuredSTCN
    else:
        raise Exception()

    input_dropout = config['input_layer'].get('dropout_rate', 0)
    experiment_name_format = "{}-{}{}-{}-{}{}-b{}-{}x{}@{}{}-in{}_out{}-{}-{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    FLAGS.model_type,
                                                    "-"+FLAGS.experiment_name if FLAGS.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    "sparse_" + config['joint_prediction_model'] if FLAGS.use_sparse_fk_joints and config['joint_prediction_model'] == "fk_joints" else config['joint_prediction_model'],
                                                    "-idrop_" + str(input_dropout) if input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    config['cnn_layer']['num_encoder_layers'],
                                                    config['cnn_layer']['num_filters'],
                                                    config['cnn_layer']['filter_size'],
                                                    '-residual_vel' if FLAGS.residual_velocities else '',
                                                    FLAGS.seq_length_in,
                                                    FLAGS.seq_length_out,
                                                    'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
                                                    config['action_loss_type'])
    return model_cls, config, experiment_name


def create_seq2seq_model(actions, sampling=False):
    """Create translation model and initialize or load parameters in session."""

    config = dict()
    config['seed'] = 1234
    config['optimizer'] = FLAGS.optimizer
    config['loss_on_encoder_outputs'] = False  # Only valid for Wavenet variants.
    config['residual_velocities'] = FLAGS.residual_velocities
    config['joint_prediction_model'] = FLAGS.joint_prediction_model  # "plain", "separate_joints", "fk_joints"
    config['use_sparse_fk_joints'] = FLAGS.use_sparse_fk_joints  # "plain", "separate_joints", "fk_joints"
    config['architecture'] = FLAGS.architecture
    config['source_seq_len'] = FLAGS.seq_length_in
    config['target_seq_len'] = FLAGS.seq_length_out
    config['rnn_size'] = FLAGS.size
    config['cell_type'] = FLAGS.cell_type
    config['num_layers'] = FLAGS.num_layers
    config['grad_clip_by_norm'] = FLAGS.max_gradient_norm
    config['batch_size'] = FLAGS.batch_size
    config['learning_rate'] = FLAGS.learning_rate
    config['learning_rate_decay_rate'] = FLAGS.learning_rate_decay_rate
    config['learning_rate_decay_type'] = FLAGS.learning_rate_decay_type
    config['autoregressive_input'] = FLAGS.autoregressive_input
    config['number_of_actions'] = 0 if FLAGS.omit_one_hot else len(actions)
    config['one_hot'] = not FLAGS.omit_one_hot
    config['residual_velocities'] = FLAGS.residual_velocities
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = FLAGS.input_dropout_rate
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = FLAGS.output_layer_number
    config['output_layer']['size'] = FLAGS.output_layer_size
    config['output_layer']['activation_fn'] = C.RELU
    config['angle_loss_type'] = FLAGS.angle_loss
    config['rep'] = "rot_mat" if FLAGS.use_rotmat else "aa"
    config['action_loss_type'] = C.LOSS_ACTION_L2
    if FLAGS.action_loss != C.LOSS_ACTION_L2:
        print("!!!Only L2 action loss is implemented for seq2seq models!!!")

    if FLAGS.model_type == "seq2seq":
        model_cls = models.Seq2SeqModel
    elif FLAGS.model_type == "seq2seq_feedback":
        model_cls = models.Seq2SeqFeedbackModel
        config['feed_error_to_encoder'] = FLAGS.feed_error_to_encoder
    else:
        raise ValueError("'{}' model unknown".format(FLAGS.model_type))

    if not sampling:
        autoregressive_input = FLAGS.autoregressive_input
    else:
        autoregressive_input = "sampling_based"

    experiment_name_format = "{}-{}-{}-{}-{}-{}-b{}-in{}_out{}-{}-enc{}feed-{}-{}-depth{}-size{}-{}-{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    FLAGS.model_type,
                                                    FLAGS.action if FLAGS.experiment_name is None else FLAGS.experiment_name + "_" + FLAGS.action,
                                                    config['angle_loss_type'],
                                                    "sparse_" + config['joint_prediction_model'] if FLAGS.use_sparse_fk_joints and config['joint_prediction_model'] == "fk_joints" else config['joint_prediction_model'],
                                                    config["cell_type"],
                                                    config['batch_size'],
                                                    FLAGS.seq_length_in,
                                                    FLAGS.seq_length_out,
                                                    FLAGS.architecture,
                                                    '' if FLAGS.feed_error_to_encoder else 'no',
                                                    autoregressive_input,
                                                    'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
                                                    FLAGS.num_layers,
                                                    FLAGS.size,
                                                    'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel',
                                                    config['action_loss_type'])
    return model_cls, config, experiment_name


def train():
    """Train a seq2seq model on human motion"""

    actions = define_actions(FLAGS.action)

    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions,
                                                                                        FLAGS.seq_length_in,
                                                                                        FLAGS.seq_length_out,
                                                                                        FLAGS.data_dir,
                                                                                        not FLAGS.omit_one_hot,
                                                                                        FLAGS.new_preprocessing)
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:
        # === Create the model ===
        train_model, eval_model, saver, global_step, experiment_dir = create_model(sess, actions)
        # Summary writers for train and test runs
        summaries_dir = os.path.normpath(os.path.join(experiment_dir, "log"))
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        test_writer = train_writer
        print("Model created")

        rep = "rot_mat" if FLAGS.use_rotmat else "aa"

        # === Read and denormalize the gt with srnn's seeds, as we'll need them
        # many times for evaluation in Euler Angles ===
        srnn_gts_euler = get_srnn_gts(actions, eval_model, test_set, data_mean, data_std, dim_to_ignore, not FLAGS.omit_one_hot, rep)

        # === This is the training loop ===
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        previous_losses = []
        step_time, loss = 0, 0

        # Early stopping configuration.
        improvement_ratio = 0.005
        best_valid_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False
        best_google_sheet_data = None

        while not stop_signal:
            if current_step >= FLAGS.iterations:
                stop_signal = True
                break

            start_time = time.time()
            # === Training step ===
            encoder_inputs, decoder_inputs, decoder_outputs = train_model.get_batch(train_set, not FLAGS.omit_one_hot)
            step_loss, summary, _ = train_model.step(encoder_inputs, decoder_inputs, decoder_outputs)
            train_writer.add_summary(summary, current_step)
            # train_writer.add_summary(lr_summary, current_step)
            # train_writer.add_summary(grad_summary, current_step)

            if current_step % 100 == 0:
                print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss))

            step_time += (time.time() - start_time)/FLAGS.test_every
            loss += step_loss/FLAGS.test_every
            current_step += 1

            # === step decay ===
            if current_step % FLAGS.learning_rate_decay_steps == 0 and train_model.learning_rate_decay_type == "piecewise":
                sess.run(train_model.learning_rate_scheduler)

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.test_every == 0:

                # === Validation with randomly chosen seeds ===
                encoder_inputs, decoder_inputs, decoder_outputs = eval_model.get_batch(test_set, not FLAGS.omit_one_hot)
                step_loss, loss_summary, _ = eval_model.step(encoder_inputs, decoder_inputs, decoder_outputs)
                val_loss = step_loss  # Loss book-keeping

                test_writer.add_summary(loss_summary, current_step)

                print()
                print("{0: <16} |".format("milliseconds"), end="")
                for ms in [80, 160, 320, 400, 560, 1000]:
                    print(" {0:5d} |".format(ms), end="")
                print()

                all_actions_mean_error = []
                selected_actions_mean_error = []
                # dictionary {action -> {ms -> error}}
                google_sheet_data = dict()
                # === Validation with srnn's seeds ===
                for action in actions:

                    # Evaluate the model on the test batches
                    encoder_inputs, decoder_inputs, decoder_outputs = eval_model.get_batch_srnn(test_set, action)
                    srnn_poses = eval_model.sampled_step(encoder_inputs, decoder_inputs, decoder_outputs)

                    # Denormalize the output
                    srnn_pred_expmap = data_utils.revert_output_format(srnn_poses,
                                                                       data_mean, data_std, dim_to_ignore, actions,
                                                                       not FLAGS.omit_one_hot,
                                                                       rep=rep)

                    # Save the errors here
                    mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))

                    # Training is done in exponential map, but the error is reported in
                    # Euler angles, as in previous work.
                    # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-247769197
                    N_SEQUENCE_TEST = 8
                    for i in np.arange(N_SEQUENCE_TEST):
                        eulerchannels_pred = np.zeros([srnn_pred_expmap[i].shape[0], 99])

                        # Convert from exponential map to Euler angles
                        for j in np.arange(eulerchannels_pred.shape[0]):
                            if FLAGS.use_rotmat:
                                n_joints = eulerchannels_pred.shape[-1] // 9
                                for joint in range(n_joints):
                                    if joint == 0 or joint == 1:
                                        # this is global translation or global rotation, ignore
                                        continue
                                    else:
                                        init_rot = np.reshape(srnn_pred_expmap[i][j, joint*9:(joint+1)*9], [3, 3])
                                        # make sure rotation matrix is valid
                                        init_rot = data_utils.get_closest_rotmat(init_rot)
                                        eulerchannels_pred[j, joint*3:(joint+1)*3] = data_utils.rotmat2euler(init_rot)
                            else:
                                for k in np.arange(3, 97, 3):
                                    eulerchannels_pred[j, k:k + 3] = data_utils.rotmat2euler(
                                        data_utils.expmap2rotmat(srnn_pred_expmap[i][j, k:k + 3]))

                        # The global translation (first 3 entries) and global rotation
                        # (next 3 entries) are also not considered in the error, so they are set to zero.
                        # See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
                        gt_i = np.copy(srnn_gts_euler[action][i])
                        gt_i[:, 0:6] = 0

                        # Now compute the l2 error. The following is numpy port of the error
                        # function provided by Ashesh Jain (in matlab), available at
                        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m
                        # /dataParser/Utils/motionGenerationError.m#L40-L54
                        idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

                        euc_error = np.power(gt_i[:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
                        euc_error = np.sum(euc_error, 1)
                        euc_error = np.sqrt(euc_error)
                        mean_errors[i, :] = euc_error

                    # This is simply the mean error over the N_SEQUENCE_TEST examples
                    mean_mean_errors = np.mean(mean_errors, 0)
                    all_actions_mean_error.append(mean_errors)
                    if action in ["walking", "discussion", "smoking", "eating"]:
                        selected_actions_mean_error.append(mean_errors)
                    # Pretty print of the results for 80, 160, 320, 400, 560 and 1000 ms
                    print("{0: <16} |".format(action), end="")
                    for ms in [1, 3, 7, 9, 13, 24]:
                        if FLAGS.seq_length_out >= ms + 1:
                            print(" {0:.3f} |".format(mean_mean_errors[ms]), end="")
                        else:
                            print("   n/a |", end="")
                    print()

                    google_sheet_data[action] = {"80": mean_mean_errors[1],
                                                 "160": mean_mean_errors[3],
                                                 "320": mean_mean_errors[7],
                                                 "400": mean_mean_errors[9]}

                    # Ugly massive if-then to log the error to tensorboard :shrug:
                    if action == "walking":
                        summaries = sess.run(
                            [eval_model.walking_err80_summary,
                             eval_model.walking_err160_summary,
                             eval_model.walking_err320_summary,
                             eval_model.walking_err400_summary,
                             eval_model.walking_err560_summary,
                             eval_model.walking_err1000_summary],
                            {eval_model.walking_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.walking_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.walking_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.walking_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.walking_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.walking_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "eating":
                        summaries = sess.run(
                            [eval_model.eating_err80_summary,
                             eval_model.eating_err160_summary,
                             eval_model.eating_err320_summary,
                             eval_model.eating_err400_summary,
                             eval_model.eating_err560_summary,
                             eval_model.eating_err1000_summary],
                            {eval_model.eating_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.eating_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.eating_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.eating_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.eating_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.eating_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "smoking":
                        summaries = sess.run(
                            [eval_model.smoking_err80_summary,
                             eval_model.smoking_err160_summary,
                             eval_model.smoking_err320_summary,
                             eval_model.smoking_err400_summary,
                             eval_model.smoking_err560_summary,
                             eval_model.smoking_err1000_summary],
                            {eval_model.smoking_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.smoking_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.smoking_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.smoking_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.smoking_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.smoking_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "discussion":
                        summaries = sess.run(
                            [eval_model.discussion_err80_summary,
                             eval_model.discussion_err160_summary,
                             eval_model.discussion_err320_summary,
                             eval_model.discussion_err400_summary,
                             eval_model.discussion_err560_summary,
                             eval_model.discussion_err1000_summary],
                            {eval_model.discussion_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.discussion_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.discussion_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.discussion_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.discussion_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.discussion_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "directions":
                        summaries = sess.run(
                            [eval_model.directions_err80_summary,
                             eval_model.directions_err160_summary,
                             eval_model.directions_err320_summary,
                             eval_model.directions_err400_summary,
                             eval_model.directions_err560_summary,
                             eval_model.directions_err1000_summary],
                            {eval_model.directions_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.directions_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.directions_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.directions_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.directions_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.directions_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "greeting":
                        summaries = sess.run(
                            [eval_model.greeting_err80_summary,
                             eval_model.greeting_err160_summary,
                             eval_model.greeting_err320_summary,
                             eval_model.greeting_err400_summary,
                             eval_model.greeting_err560_summary,
                             eval_model.greeting_err1000_summary],
                            {eval_model.greeting_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.greeting_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.greeting_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.greeting_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.greeting_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.greeting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "phoning":
                        summaries = sess.run(
                            [eval_model.phoning_err80_summary,
                             eval_model.phoning_err160_summary,
                             eval_model.phoning_err320_summary,
                             eval_model.phoning_err400_summary,
                             eval_model.phoning_err560_summary,
                             eval_model.phoning_err1000_summary],
                            {eval_model.phoning_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.phoning_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.phoning_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.phoning_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.phoning_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.phoning_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "posing":
                        summaries = sess.run(
                            [eval_model.posing_err80_summary,
                             eval_model.posing_err160_summary,
                             eval_model.posing_err320_summary,
                             eval_model.posing_err400_summary,
                             eval_model.posing_err560_summary,
                             eval_model.posing_err1000_summary],
                            {eval_model.posing_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.posing_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.posing_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.posing_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.posing_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.posing_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "purchases":
                        summaries = sess.run(
                            [eval_model.purchases_err80_summary,
                             eval_model.purchases_err160_summary,
                             eval_model.purchases_err320_summary,
                             eval_model.purchases_err400_summary,
                             eval_model.purchases_err560_summary,
                             eval_model.purchases_err1000_summary],
                            {eval_model.purchases_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.purchases_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.purchases_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.purchases_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.purchases_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.purchases_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "sitting":
                        summaries = sess.run(
                            [eval_model.sitting_err80_summary,
                             eval_model.sitting_err160_summary,
                             eval_model.sitting_err320_summary,
                             eval_model.sitting_err400_summary,
                             eval_model.sitting_err560_summary,
                             eval_model.sitting_err1000_summary],
                            {eval_model.sitting_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.sitting_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.sitting_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.sitting_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.sitting_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.sitting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "sittingdown":
                        summaries = sess.run(
                            [eval_model.sittingdown_err80_summary,
                             eval_model.sittingdown_err160_summary,
                             eval_model.sittingdown_err320_summary,
                             eval_model.sittingdown_err400_summary,
                             eval_model.sittingdown_err560_summary,
                             eval_model.sittingdown_err1000_summary],
                            {eval_model.sittingdown_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.sittingdown_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.sittingdown_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.sittingdown_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.sittingdown_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.sittingdown_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "takingphoto":
                        summaries = sess.run(
                            [eval_model.takingphoto_err80_summary,
                             eval_model.takingphoto_err160_summary,
                             eval_model.takingphoto_err320_summary,
                             eval_model.takingphoto_err400_summary,
                             eval_model.takingphoto_err560_summary,
                             eval_model.takingphoto_err1000_summary],
                            {eval_model.takingphoto_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.takingphoto_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.takingphoto_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.takingphoto_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.takingphoto_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.takingphoto_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "waiting":
                        summaries = sess.run(
                            [eval_model.waiting_err80_summary,
                             eval_model.waiting_err160_summary,
                             eval_model.waiting_err320_summary,
                             eval_model.waiting_err400_summary,
                             eval_model.waiting_err560_summary,
                             eval_model.waiting_err1000_summary],
                            {eval_model.waiting_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.waiting_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.waiting_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.waiting_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.waiting_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.waiting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "walkingdog":
                        summaries = sess.run(
                            [eval_model.walkingdog_err80_summary,
                             eval_model.walkingdog_err160_summary,
                             eval_model.walkingdog_err320_summary,
                             eval_model.walkingdog_err400_summary,
                             eval_model.walkingdog_err560_summary,
                             eval_model.walkingdog_err1000_summary],
                            {eval_model.walkingdog_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.walkingdog_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.walkingdog_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.walkingdog_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.walkingdog_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.walkingdog_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
                    elif action == "walkingtogether":
                        summaries = sess.run(
                            [eval_model.walkingtogether_err80_summary,
                             eval_model.walkingtogether_err160_summary,
                             eval_model.walkingtogether_err320_summary,
                             eval_model.walkingtogether_err400_summary,
                             eval_model.walkingtogether_err560_summary,
                             eval_model.walkingtogether_err1000_summary],
                            {eval_model.walkingtogether_err80  : mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
                             eval_model.walkingtogether_err160 : mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
                             eval_model.walkingtogether_err320 : mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
                             eval_model.walkingtogether_err400 : mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
                             eval_model.walkingtogether_err560 : mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
                             eval_model.walkingtogether_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})

                    for i in np.arange(len(summaries)):
                        test_writer.add_summary(summaries[i], current_step)

                all_actions_mean_error = np.mean(np.concatenate(all_actions_mean_error, axis=0), axis=0)
                summaries = sess.run(
                    [eval_model.all_mean_err80_summary,
                     eval_model.all_mean_err160_summary,
                     eval_model.all_mean_err320_summary,
                     eval_model.all_mean_err400_summary,
                     eval_model.all_mean_err560_summary,
                     eval_model.all_mean_err1000_summary],
                    {eval_model.all_mean_err80  : all_actions_mean_error[1] if FLAGS.seq_length_out >= 2 else None,
                     eval_model.all_mean_err160 : all_actions_mean_error[3] if FLAGS.seq_length_out >= 4 else None,
                     eval_model.all_mean_err320 : all_actions_mean_error[7] if FLAGS.seq_length_out >= 8 else None,
                     eval_model.all_mean_err400 : all_actions_mean_error[9] if FLAGS.seq_length_out >= 10 else None,
                     eval_model.all_mean_err560 : all_actions_mean_error[13] if FLAGS.seq_length_out >= 14 else None,
                     eval_model.all_mean_err1000: all_actions_mean_error[24] if FLAGS.seq_length_out >= 25 else None})
                for i in np.arange(len(summaries)):
                    test_writer.add_summary(summaries[i], current_step)

                summaries = sess.run([eval_model.all_mean_err_summary], {eval_model.all_mean_err: np.mean(all_actions_mean_error)})
                for i in np.arange(len(summaries)):
                    test_writer.add_summary(summaries[i], current_step)

                valid_loss = np.mean(np.concatenate(selected_actions_mean_error, axis=0))
                print()
                print("============================\n"
                      "Global step:         %d\n"
                      "Learning rate:       %.4f\n"
                      "Step-time (ms):     %.4f\n"
                      "Train loss avg:      %.4f\n"
                      "--------------------------\n"
                      "Val loss:            %.4f\n"
                      "All avg loss:            %.4f\n"
                      "Early stopping loss:  %.4f\n"
                      "============================" % (current_step, train_model.learning_rate.eval(), step_time*1000,
                                                        loss, val_loss, np.mean(all_actions_mean_error), valid_loss))
                print()

                previous_losses.append(loss)

                # Early stopping check.
                if (best_valid_loss - valid_loss) > np.abs(best_valid_loss*improvement_ratio):
                    num_steps_wo_improvement = 0
                else:
                    num_steps_wo_improvement += 1
                if num_steps_wo_improvement == FLAGS.early_stopping_tolerance:
                    stop_signal = True

                # Save the model
                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss
                    best_google_sheet_data = copy.deepcopy(google_sheet_data)
                    print("Saving the model to {}".format(experiment_dir))
                    saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=current_step)

                step_time, loss = 0, 0
                sys.stdout.flush()

        # store best eval results in google sheet data
        g_logger = GoogleSheetLogger(credential_file=C.LOGGER_MANU,
                                     workbook_name="martinez_setting_experiments")
        glog_data = {'Model ID': [os.path.split(experiment_dir)[-1].split('-')[0]],
                     'Model Name': ['-'.join(os.path.split(experiment_dir)[-1].split('-')[1:])],
                     'Comment': [""]}

        which_actions = ["walking", "eating", "discussion", "smoking"]
        for action in which_actions:
            best_euler = best_google_sheet_data[action]
            for ms in best_euler:
                glog_data[action[0] + ms] = [best_euler[ms]]

        g_logger.append_row(glog_data, sheet_name="logs")


def get_srnn_gts(actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot, rep, to_euler=True):
    """
    Get the ground truths for srnn's sequences, and convert to Euler angles.
    (the error is always computed in Euler angles).

    Args
      actions: a list of actions to get ground truths for.
      model: training model we are using (we only use the "get_batch" method).
      test_set: dictionary with normalized training data.
      data_mean: d-long vector with the mean of the training data.
      data_std: d-long vector with the standard deviation of the training data.
      dim_to_ignore: dimensions that we are not using to train/predict.
      one_hot: whether the data comes with one-hot encoding indicating action.
      to_euler: whether to convert the angles to Euler format or keep thm in exponential map

    Returns
      srnn_gts_euler: a dictionary where the keys are actions, and the values
        are the ground_truth, denormalized expected outputs of srnns's seeds.
    """
    assert rep in ["rot_mat", "aa"]
    srnn_gts_euler = {}

    for action in actions:

        srnn_gt_euler = []
        _, _, srnn_expmap = model.get_batch_srnn(test_set, action)

        # expmap -> rotmat -> euler
        for i in np.arange(srnn_expmap.shape[0]):
            denormed = data_utils.unNormalizeData(srnn_expmap[i, :, :], data_mean, data_std, dim_to_ignore, actions,
                                                  one_hot)

            if to_euler:
                if rep == "rot_mat":
                    euler_rep = np.zeros([denormed.shape[0], 99])
                    for j in np.arange(denormed.shape[0]):
                        n_joints = denormed.shape[1] // 9
                        for joint in range(n_joints):
                            if joint == 0 or joint == 1:
                                # this is global translation or rotation, ignore
                                continue
                            else:
                                init_rot = np.reshape(denormed[j, joint * 9:(joint + 1) * 9], [3, 3])
                                # make sure rotation matrix is valid
                                init_rot = data_utils.get_closest_rotmat(init_rot)
                                euler_rep[j, joint * 3:(joint + 1) * 3] = data_utils.rotmat2euler(init_rot)
                else:
                    euler_rep = denormed.copy()
                    for j in np.arange(denormed.shape[0]):
                        for k in np.arange(3, 97, 3):
                            euler_rep[j, k:k + 3] = data_utils.rotmat2euler(data_utils.expmap2rotmat(denormed[j, k:k + 3]))

                srnn_gt_euler.append(euler_rep)
            else:
                srnn_gt_euler.append(denormed)

        # Put back in the dictionary
        srnn_gts_euler[action] = srnn_gt_euler

    return srnn_gts_euler


def sample():
    """Sample predictions for srnn's seeds"""

    if FLAGS.experiment_id is None:
        raise ValueError("Must give an experiment id to read parameters from")

    actions = define_actions(FLAGS.action)

    # Use the CPU if asked to
    device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(device_count=device_count)) as sess:

        # === Create the model ===
        sampling = True
        train_model, eval_model, saver, global_step, experiment_dir = create_model(sess, actions, sampling)
        print("Model created")

        rep = "rot_mat" if FLAGS.use_rotmat else "aa"

        # Load all the data
        train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
            actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot,
            FLAGS.new_preprocessing)

        # === Read and denormalize the gt with srnn's seeds, as we'll need them
        # many times for evaluation in Euler Angles ===
        srnn_gts_euler = get_srnn_gts(actions, eval_model, test_set, data_mean, data_std, dim_to_ignore,
                                      not FLAGS.omit_one_hot, rep, to_euler=True)

        def _to_expmap(_list_of_samples):
            """list of samples expected in shape (seq_len, n_joints*3*3)"""
            _converted = []
            for _the_sample in _list_of_samples:
                _seq_len = _the_sample.shape[0]
                _rots = np.reshape(_the_sample, [-1, 3, 3])
                _aas = np.zeros([_rots.shape[0], 3])
                for _r in range(_rots.shape[0]):
                    _aas[_r] = np.squeeze(cv2.Rodrigues(_rots[_r])[0])
                _converted.append(np.reshape(_aas, [_seq_len, 99]))
            return _converted

        # Clean and create a new h5 file of samples
        SAMPLES_FNAME = os.path.join(experiment_dir, 'samples.h5')
        try:
            os.remove(SAMPLES_FNAME)
        except OSError:
            pass

        # Predict and save for each action
        for action in actions:

            # Make prediction with srnn' seeds
            encoder_inputs, decoder_inputs, decoder_outputs = eval_model.get_batch_srnn(test_set, action)
            srnn_poses = eval_model.sampled_step(encoder_inputs, decoder_inputs, decoder_outputs)

            srnn_seeds = np.concatenate([encoder_inputs, decoder_inputs[:, 0:1]], axis=1)  # first frame of decoder input is gt
            srnn_seeds = np.transpose(srnn_seeds, [1, 0, 2])  # transpose so that revert output format works correctly
            srnn_seeds_expmap = data_utils.revert_output_format(srnn_seeds, data_mean, data_std, dim_to_ignore, actions,
                                                                not FLAGS.omit_one_hot, rep)

            # denormalizes too
            srnn_pred_expmap = data_utils.revert_output_format(srnn_poses, data_mean, data_std, dim_to_ignore, actions,
                                                               not FLAGS.omit_one_hot, rep)

            # Save the conditioning seeds
            if rep == "rot_mat":
                # convert back to exponential map
                srnn_pred_expmap = _to_expmap(srnn_pred_expmap)
                srnn_seeds_expmap = _to_expmap(srnn_seeds_expmap)

            # Save the samples
            with h5py.File(SAMPLES_FNAME, 'a') as hf:
                for i in np.arange(8):
                    # Save conditioning ground truth
                    node_name = 'expmap/gt/{1}_{0}'.format(i, action)
                    hf.create_dataset(node_name, data=srnn_seeds_expmap[i])
                    # Save prediction
                    node_name = 'expmap/preds/{1}_{0}'.format(i, action)
                    hf.create_dataset(node_name, data=srnn_pred_expmap[i])

            # # Compute and save the errors here
            # mean_errors = np.zeros((len(srnn_pred_expmap), srnn_pred_expmap[0].shape[0]))
            #
            # Not interested in this for the moment
            # for i in np.arange(8):
            #
            #     eulerchannels_pred = srnn_pred_expmap[i]
            #
            #     for j in np.arange(eulerchannels_pred.shape[0]):
            #         for k in np.arange(3, 97, 3):
            #             eulerchannels_pred[j, k:k + 3] = data_utils.rotmat2euler(
            #                 data_utils.expmap2rotmat(eulerchannels_pred[j, k:k + 3]))
            #
            #     eulerchannels_pred[:, 0:6] = 0
            #
            #     # Pick only the dimensions with sufficient standard deviation. Others are ignored.
            #     idx_to_use = np.where(np.std(eulerchannels_pred, 0) > 1e-4)[0]
            #
            #     euc_error = np.power(srnn_gts_euler[action][i][:, idx_to_use] - eulerchannels_pred[:, idx_to_use], 2)
            #     euc_error = np.sum(euc_error, 1)
            #     euc_error = np.sqrt(euc_error)
            #     mean_errors[i, :] = euc_error
            #
            # mean_mean_errors = np.mean(mean_errors, 0)
            # # print(action)
            # # print(','.join(map(str, mean_mean_errors.tolist())))
            #
            # with h5py.File(SAMPLES_FNAME, 'a') as hf:
            #     node_name = 'mean_{0}_error'.format(action)
            #     hf.create_dataset(node_name, data=mean_mean_errors)


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d"%action)


def read_all_data(actions, seq_length_in, seq_length_out, data_dir, one_hot, new_pp=True):
    """
    Loads data for training/testing and normalizes it.

    Args
      actions: list of strings (actions) to load
      seq_length_in: number of frames to use in the burn-in sequence
      seq_length_out: number of frames to use in the output sequence
      data_dir: directory to load the data from
      one_hot: whether to use one-hot encoding per action
      new_pp: ignores entire joints instead of single DOFs
    Returns
      train_set: dictionary with normalized training data
      test_set: dictionary with test data
      data_mean: d-long vector with the mean of the training data
      data_std: d-long vector with the standard dev of the training data
      dim_to_ignore: dimensions that are not used becaused stdev is too small
      dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
        seq_length_in, seq_length_out))

    train_subject_ids = [1, 6, 7, 8, 9, 11]
    # train_subject_ids = [1]
    test_subject_ids = [5]

    rep = "rot_mat" if FLAGS.use_rotmat else "aa"

    train_set, complete_train = data_utils.load_data(data_dir, train_subject_ids, actions, one_hot, FLAGS.use_rotmat)
    test_set, complete_test = data_utils.load_data(data_dir, test_subject_ids, actions, one_hot, FLAGS.use_rotmat)

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = data_utils.normalization_stats(complete_train, rep, new_pp)

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(train_set, data_mean, data_std, dim_to_use, actions, one_hot, rep)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std, dim_to_use, actions, one_hot, rep)
    print("done reading data.")

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def main(_):
    if FLAGS.sample:
        sample()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
