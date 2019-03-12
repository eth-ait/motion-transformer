from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# dirty hack so that it works on the server
import sys
sys.path.append('../eth_source')

import os
import time

import numpy as np
import tensorflow as tf

import amass_models as models
from amass_tf_data import TFRecordMotionDataset
from logger import GoogleSheetLogger

# ETH imports
from constants import Constants as C
import glob
import json
from motion_metrics import MetricsEngine


# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.95, "Learning rate mutiplier. 1 means no decay.")
tf.app.flags.DEFINE_string("learning_rate_decay_type", "piecewise", "Learning rate decay type.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs.")
tf.app.flags.DEFINE_string("optimizer", "adam", "Optimization algorithm: adam or sgd.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
tf.app.flags.DEFINE_float("input_dropout_rate", 0.0, "Dropout rate on the model inputs.")
tf.app.flags.DEFINE_integer("output_layer_size", 128, "Number of units in the output layer.")

tf.app.flags.DEFINE_string("new_experiment_id", None, "10 digit unique experiment id given externally.")
tf.app.flags.DEFINE_string("autoregressive_input", "sampling_based", "The type of decoder inputs, supervised or sampling_based")
tf.app.flags.DEFINE_integer("print_every", 100, "How often to log training error.")
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")
tf.app.flags.DEFINE_string("experiment_name", None, "A descriptive name for the experiment.")
tf.app.flags.DEFINE_string("experiment_id", None, "Unique experiment timestamp to load a pre-trained model.")
tf.app.flags.DEFINE_string("model_type", "seq2seq", "Model type: seq2seq, wavenet, stcn, simple_baseline, rnn or vrnn.")
tf.app.flags.DEFINE_boolean("feed_error_to_encoder", True, "If architecture is not tied, can choose to feed error in encoder or not")
tf.app.flags.DEFINE_boolean("new_preprocessing", True, "Only discard entire joints not single DOFs per joint")
tf.app.flags.DEFINE_string("joint_prediction_model", "plain", "plain, separate_joints or fk_joints.")
tf.app.flags.DEFINE_string("angle_loss", "joint_sum", "joint_sum, joint_mean or all_mean.")
tf.app.flags.DEFINE_boolean("no_normalization", False, "If set, do not use zero-mean unit-variance normalization.")
tf.app.flags.DEFINE_boolean("force_valid_rot", False, "If set, forces predicted outputs to be valid rotations")
tf.app.flags.DEFINE_integer("early_stopping_tolerance", 20, "# of waiting steps until the evaluation loss improves.")

args = tf.app.flags.FLAGS

# Unique timestamp to distinguish experiments with the same configuration.
experiment_timestamp = str(int(time.time()))
if args.new_experiment_id is not None:
    if len(args.new_experiment_id) != 10:
        raise Exception("Experiment ID must be 10 digits.")
    experiment_timestamp = args.new_experiment_id


def create_model(session):
    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    train_data_path = os.path.join(os.environ["AMASS_TRAIN"], "amass-?????-of-?????")
    valid_data_path = os.path.join(os.environ["AMASS_VALID"], "amass-?????-of-?????")
    test_data_path = os.path.join(os.environ["AMASS_TEST"], "amass-?????-of-?????")
    meta_data_path = os.environ["AMASS_META"]
    train_dir = os.environ["AMASS_EXPERIMENTS"]

    if args.force_valid_rot:
        assert args.no_normalization, 'normalization does not make sense when enforcing valid rotations'

    if args.model_type == "seq2seq":
        model_cls, config, experiment_name = get_seq2seq_config(args)
    elif args.model_type == "simple_baseline":
        # get a dummy config from seq2seq
        model_cls, config, _ = get_seq2seq_config(args)
        experiment_name = "25041990-a-simple-yet-effective-baseline"
    elif args.model_type == "stcn":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "wavenet":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "seq2seq_feedback":
        model_cls, config, experiment_name = get_seq2seq_config(args)
    elif args.model_type == "structured_stcn":
        model_cls, config, experiment_name = get_stcn_config(args)
    elif args.model_type == "rnn":
        model_cls, config, experiment_name = get_rnn_config(args)
    elif args.model_type == "vrnn":
        model_cls, config, experiment_name = get_rnn_config(args)
    else:
        raise Exception("Unknown model type.")

    experiment_name += '{}-norm'.format('-no' if args.no_normalization else '')

    with tf.name_scope("training_data"):
        windows_length = args.seq_length_in + args.seq_length_out
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           extract_windows_of=windows_length,
                                           num_parallel_calls=16,
                                           normalize=not args.no_normalization)
        train_pl = train_data.get_tf_samples()

    assert windows_length == 160, "TFRecords are hardcoded with length of 160."
    with tf.name_scope("validation_data"):
        eval_data = TFRecordMotionDataset(data_path=valid_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          extract_windows_of=0,
                                          num_parallel_calls=16,
                                          normalize=not args.no_normalization)
        eval_pl = eval_data.get_tf_samples()

    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          extract_windows_of=0,
                                          num_parallel_calls=16,
                                          normalize=not args.no_normalization)
        test_pl = test_data.get_tf_samples()

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

    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=True,
            dtype=tf.float32)
        test_model.build_graph()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))
    config["num_parameters"] = int(num_param)

    if args.experiment_id is None:
        experiment_dir = os.path.normpath(os.path.join(train_dir, experiment_name))
    else:
        experiment_dir = glob.glob(os.path.join(train_dir, args.experiment_id + "-*"), recursive=False)[0]
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)
    print("Experiment directory " + experiment_dir)

    train_model.optimization_routines()
    train_model.summary_routines()
    eval_model.summary_routines()

    # Create saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    models = (train_model, eval_model, test_model)
    data = (train_data, eval_data, test_data)

    if args.experiment_id is None:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return models, data, saver, global_step, experiment_dir

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
        return models, data, saver, global_step, experiment_dir
    else:
        print("Could not find checkpoint. Aborting.")
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def load_latest_checkpoint(sess, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError("could not load checkpoint")


def get_rnn_config(args):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['model_type'] = args.model_type
    config['seed'] = 1234
    config['learning_rate'] = args.learning_rate
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_type'] = 'exponential'
    config['learning_rate_decay_steps'] = 1000
    config['cell'] = dict()
    config['cell']['cell_type'] = C.LayerNormLSTM
    config['cell']['cell_size'] = 1024
    config['cell']['cell_num_layers'] = 1
    if args.model_type == 'vrnn':
        config['cell']['kld_weight'] = 1  # dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
        config['cell']['type'] = C.LATENT_GAUSSIAN
        config['cell']['latent_size'] = 8
        config['cell']["hidden_activation_fn"] = C.RELU
        config['cell']["num_hidden_units"] = 256
        config['cell']["num_hidden_layers"] = 1
        config['cell']['latent_sigma_threshold'] = 5.0
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = args.input_dropout_rate
    config['input_layer']['num_layers'] = 1
    config['input_layer']['size'] = 256
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 1
    config['output_layer']['size'] = args.output_layer_size
    config['output_layer']['activation_fn'] = C.RELU

    config['optimizer'] = args.optimizer
    config['grad_clip_by_norm'] = 1
    config['loss_on_encoder_outputs'] = True
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss
    config['force_valid_rot'] = args.force_valid_rot

    model_exp_name = ""
    if args.model_type == "rnn":
        model_cls = models.RNN
    elif args.model_type == "vrnn":
        model_cls = models.VRNN
        kld_weight = config['cell']['kld_weight']
        kld_txt = str(int(kld_weight)) if isinstance(kld_weight, float) or isinstance(kld_weight, int) else "a"
        model_exp_name = "-kld_{}-l{}".format(kld_txt, config['cell']['latent_size'])
    else:
        raise Exception()

    input_dropout = config['input_layer'].get('dropout_rate', 0)

    experiment_name_format = "{}-{}{}-{}-{}{}-b{}{}-{}@{}{}-in{}_out{}{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    "-idrop_" + str(input_dropout) if input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    model_exp_name,
                                                    config['cell']['cell_size'],
                                                    config['cell']['cell_type'],
                                                    '-residual_vel' if args.residual_velocities else '',
                                                    args.seq_length_in,
                                                    args.seq_length_out,
                                                    '-force_rot' if args.force_valid_rot else '')
    return model_cls, config, experiment_name


def get_stcn_config(args):
    """Create translation model and initialize or load parameters in session."""
    config = dict()
    config['model_type'] = args.model_type
    config['seed'] = 1234
    config['learning_rate'] = args.learning_rate
    config['learning_rate_decay_rate'] = 0.98
    config['learning_rate_decay_steps'] = 1000
    config['learning_rate_decay_type'] = 'exponential'
    config['latent_layer'] = dict()
    config['latent_layer']['kld_weight'] = dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
    config['latent_layer']['latent_size'] = [128, 64, 32, 16, 8, 4, 2]
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
    config['input_layer']['dropout_rate'] = args.input_dropout_rate
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 2
    config['output_layer']['size'] = 128
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

    config['optimizer'] = args.optimizer
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss
    config['force_valid_rot'] = args.force_valid_rot

    input_dropout = config['input_layer'].get('dropout_rate', 0)
    model_exp_name = ""
    if args.model_type == "stcn":
        model_cls = models.STCN
        kld_weight = config['latent_layer']['kld_weight']
        kld_txt = str(int(kld_weight)) if isinstance(kld_weight, float) or isinstance(kld_weight, int) else "a"
        model_exp_name = "-kld_{}".format(kld_txt)
    elif args.model_type == "wavenet":
        model_cls = models.Wavenet
        if not(config['decoder_use_enc_skip'] or config['decoder_use_enc_last'] or config['decoder_use_raw_inputs']):
            config['decoder_use_enc_last'] = True
        del config["latent_layer"]
    elif args.model_type == "structured_stcn":
        model_cls = models.StructuredSTCN
    else:
        raise Exception()

    experiment_name_format = "{}-{}{}-{}-{}{}{}-b{}-{}x{}@{}{}-in{}_out{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    model_exp_name,
                                                    "-idrop_" + str(input_dropout) if input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    config['cnn_layer']['num_encoder_layers'],
                                                    config['cnn_layer']['num_filters'],
                                                    config['cnn_layer']['filter_size'],
                                                    '-residual_vel' if args.residual_velocities else '',
                                                    args.seq_length_in,
                                                    args.seq_length_out,
                                                    '-force_rot' if args.force_valid_rot else '')
    return model_cls, config, experiment_name


def get_seq2seq_config(args):
    """Create translation model and initialize or load parameters in session."""

    config = dict()
    config['model_type'] = args.model_type
    config['seed'] = 1234
    config['loss_on_encoder_outputs'] = False  # Only valid for Wavenet variants.
    config['optimizer'] = args.optimizer
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
    config['force_valid_rot'] = args.force_valid_rot

    if args.model_type == "seq2seq":
        model_cls = models.Seq2SeqModel
    elif args.model_type == "seq2seq_feedback":
        model_cls = models.Seq2SeqFeedbackModel
        config['feed_error_to_encoder'] = args.feed_error_to_encoder
    elif args.model_type == "simple_baseline":
        model_cls = models.ASimpleYetEffectiveBaseline
    else:
        raise ValueError("'{}' model unknown".format(args.model_type))

    experiment_name_format = "{}-{}-{}-{}-b{}-in{}_out{}-{}-enc{}feed-{}-depth{}-size{}-{}{}"
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
                                                    'residual_vel' if args.residual_velocities else 'not_residual_vel',
                                                    '-force_rot' if args.force_valid_rot else '')
    return model_cls, config, experiment_name


def train():
    """Train a seq2seq model on human motion"""
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    device_count = {"GPU": 0} if args.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        # Create the model
        models, data, saver, global_step, experiment_dir = create_model(sess)
        train_model, eval_model, test_model = models
        train_data, eval_data, test_data = data

        # Create metrics engine including summaries
        # in milliseconds: 83.3, 166.7, 316.7, 400, 566.7, 1000]
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS if x <= train_model.target_seq_len]
        pck_threshs = C.METRIC_PCK_THRESHS  # thresholds for pck, in meters
        metrics_engine = MetricsEngine("../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl",
                                       target_lengths,
                                       pck_threshs=pck_threshs,
                                       force_valid_rot=True)
        # create the necessary summary placeholders and ops
        metrics_engine.create_summaries()
        # reset computation of metrics
        metrics_engine.reset()

        # Summary writers for train and test runs
        summaries_dir = os.path.normpath(os.path.join(experiment_dir, "log"))
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        test_writer = train_writer
        print("Model created")

        # Early stopping configuration.
        early_stopping_metric_key = C.METRIC_JOINT_ANGLE
        improvement_ratio = 0.01
        best_eval_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False

        # Training loop configuration.
        time_counter = 0.0
        step = 1
        epoch = 0
        train_loss = 0.0
        train_iter = train_data.get_iterator()
        eval_iter = eval_data.get_iterator()
        test_iter = test_data.get_iterator()

        print("Running Training Loop.")
        # Assuming that we use initializable iterators.
        sess.run(train_iter.initializer)
        sess.run(eval_iter.initializer)

        def evaluate_model(_eval_model, _eval_iter, _metrics_engine):
            # make a full pass on the validation or test dataset and compute the metrics
            _metrics_engine.reset()
            sess.run(_eval_iter.initializer)
            try:
                while True:
                    # TODO(kamanuel) should we compute the validation loss here as well, if so how?
                    # get the predictions and ground truth values
                    prediction, targets, seed_sequence, data_id = _eval_model.sampled_step(sess)
                    # unnormalize - if normalization is not configured, these calls do nothing
                    p = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": prediction}, "poses")
                    t = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": targets}, "poses")
                    _metrics_engine.compute_and_aggregate(p["poses"], t["poses"])
            except tf.errors.OutOfRangeError:
                # finalize the computation of the metrics
                final_metrics = _metrics_engine.get_final_metrics()
            return final_metrics

        while not stop_signal:
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
                        time_elapsed = time_counter/args.print_every
                        train_loss, time_counter = 0., 0.
                        print("Train [{:04d}] \t Loss: {:.3f} \t time/batch = {:.3f}".format(step,
                                                                                             train_loss_avg,
                                                                                             time_elapsed))
                    # Learning rate decay
                    if step % args.learning_rate_decay_steps == 0 and train_model.learning_rate_decay_type == "piecewise":
                        sess.run(train_model.learning_rate_scheduler)

                except tf.errors.OutOfRangeError:
                    sess.run(train_iter.initializer)
                    epoch += 1
                    if epoch >= args.num_epochs:
                        stop_signal = True
                        break

            # Evaluation: make a full pass on the evaluation data.
            eval_metrics = evaluate_model(eval_model, eval_iter, metrics_engine)
            # print an informative string to the console
            print("Eval [{:04d}] \t {}".format(step - 1, metrics_engine.get_summary_string(eval_metrics)))
            # get the summary feed dict
            summary_feed = metrics_engine.get_summary_feed_dict(eval_metrics)
            # get the writable summaries
            summaries = sess.run(metrics_engine.all_summaries_op, feed_dict=summary_feed)
            # write to log
            test_writer.add_summary(summaries, step)
            # reset the computation of the metrics
            metrics_engine.reset()
            # reset the evaluation iterator
            sess.run(eval_iter.initializer)

            # Early stopping check.
            eval_loss = eval_metrics[early_stopping_metric_key].sum()
            if (best_eval_loss - eval_loss) > np.abs(best_eval_loss*improvement_ratio):
                num_steps_wo_improvement = 0
            else:
                num_steps_wo_improvement += 1
            if num_steps_wo_improvement == args.early_stopping_tolerance:
                stop_signal = True

            if eval_loss <= best_eval_loss:
                best_eval_loss = eval_loss
                print("Saving the model to {}".format(experiment_dir))
                saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=step-1)

        print("End of Training.")
        load_latest_checkpoint(sess, saver, experiment_dir)

        print("Evaluating validation set ...")
        eval_metrics = evaluate_model(eval_model, eval_iter, metrics_engine)
        print("Validation [{:04d}] \t {}".format(step - 1, metrics_engine.get_summary_string(eval_metrics)))

        print("Evaluating test set ...")
        test_metrics = evaluate_model(test_model, test_iter, metrics_engine)
        print("Test [{:04d}] \t {}".format(step - 1, metrics_engine.get_summary_string(test_metrics)))

        # create logger
        gLogger = GoogleSheetLogger(credential_file=C.LOGGER_MANU,
                                    workbook_name="motion_modelling_experiments")
        glog_data = {'Model ID': [os.path.split(experiment_dir)[-1].split('-')[0]],
                     'Model Name': ['-'.join(os.path.split(experiment_dir)[-1].split('-')[1:])],
                     'Comment': [""]}

        # gather the metrics
        for t in metrics_engine.target_lengths:
            glog_eval_metrics = metrics_engine.get_summary_glogger(eval_metrics, until=t)
            glog_test_metrics = metrics_engine.get_summary_glogger(test_metrics, is_validation=False, until=t)

            glog_data["Comment"] = ["until_{}".format(t)]
            glog_data = {**glog_data, **glog_eval_metrics, **glog_test_metrics}
            gLogger.append_row(glog_data, sheet_name="until_{}".format(t))

        print("Finished.")


def sample():
    raise Exception("Not implemented.")


def main(_):
    if args.sample:
        sample()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
