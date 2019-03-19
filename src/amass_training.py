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
from amass_tf_data import SRNNTFRecordMotionDataset
from logger import GoogleSheetLogger

# ETH imports
from constants import Constants as C
import glob
import json
from motion_metrics import MetricsEngine
from motion_metrics import rotmat2euler
from motion_metrics import quat2euler
from motion_metrics import aa2rotmat
from fk import H36MForwardKinematics
from fk import SMPLForwardKinematics
from fk import H36M_MAJOR_JOINTS
from visualize import Visualizer


# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.95, "Learning rate mutiplier. 1 means no decay.")
tf.app.flags.DEFINE_string("learning_rate_decay_type", "piecewise", "Learning rate decay type.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 10000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs.")
tf.app.flags.DEFINE_string("optimizer", "adam", "Optimization algorithm: adam or sgd.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Number of frames to feed into the encoder. 25 fps")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Number of frames that the decoder has to predict. 25fps")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
tf.app.flags.DEFINE_string("residual_velocities_type", "plus", "How to combine inputs with model prediction")
tf.app.flags.DEFINE_float("input_dropout_rate", 0.0, "Dropout rate on the model inputs.")
tf.app.flags.DEFINE_integer("output_layer_size", 128, "Number of units in the output layer.")
tf.app.flags.DEFINE_integer("output_layer_number", 2, "Number of output layer.")
tf.app.flags.DEFINE_string("cell_type", C.GRU, "RNN cell type: gru, lstm, layernormbasiclstmcell")
tf.app.flags.DEFINE_integer("cell_size", 1024, "RNN cell size.")
tf.app.flags.DEFINE_integer("cell_layers", 1, "Number of cells in the RNN model.")
tf.app.flags.DEFINE_boolean("wavenet_enc_skip", False, "Wavenet model using skip connections.")
tf.app.flags.DEFINE_boolean("wavenet_enc_last", False, "Wavenet model using the last layer.")
tf.app.flags.DEFINE_boolean("wavenet_enc_raw", False, "Wavenet model using the inputs.")
tf.app.flags.DEFINE_integer("wavenet_units", 128, "Number of wavenet units generally used in the model.")

tf.app.flags.DEFINE_string("new_experiment_id", None, "10 digit unique experiment id given externally.")
tf.app.flags.DEFINE_string("autoregressive_input", "sampling_based", "The type of decoder inputs, supervised or sampling_based")
tf.app.flags.DEFINE_integer("print_every", 100, "How often to log training error.")
tf.app.flags.DEFINE_integer("test_every", 1000, "How often to compute error on the validation set.")
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
tf.app.flags.DEFINE_boolean("rot_matrix_regularization", False, "If set, apply regularization term.")
tf.app.flags.DEFINE_boolean("force_valid_rot", False, "If set, forces predicted outputs to be valid rotations")
tf.app.flags.DEFINE_boolean("use_quat", False, "Use quaternions instead of rotation matrices")
tf.app.flags.DEFINE_boolean("use_aa", False, "Use angle-axis instead of rotation matrices")
tf.app.flags.DEFINE_integer("early_stopping_tolerance", 20, "# of waiting steps until the validation loss improves.")
tf.app.flags.DEFINE_boolean("dynamic_validation_split", False, "Validation samples are extracted on-the-fly.")
tf.app.flags.DEFINE_boolean("use_h36m_only", False, "Only use H36M for training and validaton")
tf.app.flags.DEFINE_boolean("use_h36m_martinez", False, "Only use H36M coming directly from Martinez code repo")

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

    use_quat = args.use_quat
    use_aa = args.use_aa

    assert not (use_quat and use_quat), 'must choose between quat or aa'

    if use_quat:
        assert args.no_normalization, 'we normalize quaternions on the output, so it does not make sense ' \
                                      'to use normalization'

    # if use_aa:
    #    assert args.use_h36m_only or args.use_h36m_martinez, 'currently only H3.6M is in angle-axis format'

    rep = "quat" if use_quat else "rotmat"
    rep = "aa" if use_aa else rep

    data_path = os.environ["AMASS_DATA"]
    if args.use_h36m_only:
        assert not args.use_h36m_martinez
        data_path = os.path.join(data_path, '../per_db/h36m')

    if args.use_h36m_martinez:
        data_path = os.path.join(data_path, '../../h3.6m/tfrecords/')

    train_data_path = os.path.join(data_path, rep, "training", "amass-?????-of-?????")
    if args.dynamic_validation_split:
        valid_data_path = os.path.join(data_path, rep, "validation_dynamic", "amass-?????-of-?????")
    else:
        valid_data_path = os.path.join(data_path, rep, "validation", "amass-?????-of-?????")
    test_data_path = os.path.join(data_path, rep, "test", "amass-?????-of-?????")
    meta_data_path = os.path.join(data_path, rep, "training", "stats.npz")
    train_dir = os.environ["AMASS_EXPERIMENTS"]

    if args.force_valid_rot:
        assert args.no_normalization, 'Normalization does not make sense when enforcing valid rotations.'
    if not args.no_normalization:
        assert not args.rot_matrix_regularization, "The inputs and outputs must be between -1 and 1."

    if args.residual_velocities_type == "matmul":
        # this makes only sense if we use rotation matrices
        assert rep == "rotmat"

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

    # Naming.
    experiment_name += '{}_norm'.format('-no' if args.no_normalization else '')
    if args.rot_matrix_regularization:
        experiment_name += "rot_loss"

    window_length = args.seq_length_in + args.seq_length_out
    experiment_name += "{}".format("-h36m" if args.use_h36m_only else "")
    experiment_name += "{}".format("-h36martinez" if args.use_h36m_martinez else "")

    with tf.name_scope("training_data"):
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           extract_windows_of=window_length,
                                           extract_random_windows=True,
                                           num_parallel_calls=16,
                                           normalize=not args.no_normalization)
        train_pl = train_data.get_tf_samples()

    assert window_length <= 180, "TFRecords are hardcoded with length of 180."
    if args.dynamic_validation_split:
        extract_random_windows = True
    else:
        window_length = 0
        extract_random_windows = False

    with tf.name_scope("validation_data"):
        valid_data = TFRecordMotionDataset(data_path=valid_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           extract_windows_of=window_length,
                                           extract_random_windows=extract_random_windows,
                                           num_parallel_calls=16,
                                           normalize=not args.no_normalization)
        valid_pl = valid_data.get_tf_samples()

    window_length = 0 if window_length == 160 else window_length
    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          extract_random_windows=False,
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
        valid_model = model_cls(
            config=config,
            data_pl=valid_pl,
            mode=C.SAMPLE,
            reuse=True,
            dtype=tf.float32)
        valid_model.build_graph()

    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=True,
            dtype=tf.float32)
        test_model.build_graph()

    if args.use_h36m_martinez:
        # create model and data for SRNN evaluation
        with tf.name_scope("srnn_data"):
            srnn_path = os.path.join(data_path, rep, "srnn_poses", "amass-?????-of-?????")
            srnn_data = SRNNTFRecordMotionDataset(data_path=srnn_path,
                                                  meta_data_path=meta_data_path,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  extract_windows_of=120,  # 100 = 2 seconds input, 20 = 400 ms output
                                                  extract_random_windows=False,
                                                  num_parallel_calls=16,
                                                  normalize=not args.no_normalization)
            srnn_pl = srnn_data.get_tf_samples()

        with tf.name_scope("SRNN"):
            srnn_model = model_cls(
                config=config,
                data_pl=srnn_pl,
                mode=C.SAMPLE,
                reuse=True,
                dtype=tf.float32)
            srnn_model.build_graph()

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
    valid_model.summary_routines()

    # Create saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)

    models = [train_model, valid_model, test_model]
    data = [train_data, valid_data, test_data]

    if args.use_h36m_martinez:
        models.append(srnn_model)
        data.append(srnn_data)

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
    config['cell']['cell_type'] = args.cell_type
    config['cell']['cell_size'] = args.cell_size
    config['cell']['cell_num_layers'] = args.cell_layers
    if args.model_type == 'vrnn':
        config['cell']['kld_weight'] = 1  # dict(type=C.DECAY_LINEAR, values=[0, 1.0, 1e-4])
        config['cell']['type'] = C.LATENT_GAUSSIAN
        config['cell']['latent_size'] = 64
        config['cell']["hidden_activation_fn"] = C.RELU
        config['cell']["num_hidden_units"] = 256
        config['cell']["num_hidden_layers"] = 1
        config['cell']['latent_sigma_threshold'] = 5.0
    config['input_layer'] = dict()
    config['input_layer']['dropout_rate'] = args.input_dropout_rate
    config['input_layer']['num_layers'] = 1
    config['input_layer']['size'] = 256
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = args.output_layer_number
    config['output_layer']['size'] = args.output_layer_size
    config['output_layer']['activation_fn'] = C.RELU

    config['optimizer'] = args.optimizer
    config['grad_clip_by_norm'] = args.max_gradient_norm
    config['loss_on_encoder_outputs'] = True
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['residual_velocities_type'] = args.residual_velocities_type
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss
    config['force_valid_rot'] = args.force_valid_rot
    config['rot_matrix_regularization'] = args.rot_matrix_regularization
    config['use_quat'] = args.use_quat
    config['use_aa'] = args.use_aa
    config['no_normalization'] = args.no_normalization
    config['use_h36m_only'] = args.use_h36m_only
    config['use_h36m_martinez'] = args.use_h36m_martinez

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

    experiment_name_format = "{}-{}{}-{}-{}-{}{}-b{}{}-{}@{}{}-in{}_out{}{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    "quat" if args.use_quat else "aa" if args.use_aa else "rotmat",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    "-idrop_" + str(input_dropout) if input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    model_exp_name,
                                                    config['cell']['cell_size'],
                                                    config['cell']['cell_type'],
                                                    '-residual_vel_{}'.format(args.residual_velocities_type) if args.residual_velocities else '',
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
    config['latent_layer']['latent_size'] = [64, 32, 16, 8, 4, 2, 1]
    config['latent_layer']['type'] = C.LATENT_LADDER_GAUSSIAN
    config['latent_layer']['layer_structure'] = C.LAYER_CONV1
    config['latent_layer']["hidden_activation_fn"] = C.RELU
    config['latent_layer']["num_hidden_units"] = args.wavenet_units
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
    config['input_layer']['dropout_rate'] = args.input_dropout_rate  # dict(values=[0.1, 0.5, 0.1], step=5e3, type=C.DECAY_PC)
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = args.output_layer_number
    config['output_layer']['size'] = args.output_layer_size
    config['output_layer']['type'] = C.LAYER_TCN
    config['output_layer']['filter_size'] = 2
    config['output_layer']['activation_fn'] = C.RELU
    config['cnn_layer'] = dict()
    config['cnn_layer']['num_encoder_layers'] = 35
    config['cnn_layer']['num_decoder_layers'] = 0
    config['cnn_layer']['num_filters'] = args.wavenet_units
    config['cnn_layer']['filter_size'] = 2
    config['cnn_layer']['dilation_size'] = [1, 2, 4, 8, 16]*7
    config['cnn_layer']['activation_fn'] = C.RELU
    config['cnn_layer']['use_residual'] = True
    config['cnn_layer']['zero_padding'] = True
    config['decoder_use_enc_skip'] = args.wavenet_enc_skip
    config['decoder_use_enc_last'] = args.wavenet_enc_last
    config['decoder_use_raw_inputs'] = args.wavenet_enc_raw
    config['grad_clip_by_norm'] = args.max_gradient_norm
    config['use_future_steps_in_q'] = False
    config['loss_on_encoder_outputs'] = True

    config['optimizer'] = args.optimizer
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['residual_velocities_type'] = args.residual_velocities_type
    config['joint_prediction_model'] = args.joint_prediction_model
    config['angle_loss_type'] = args.angle_loss
    config['force_valid_rot'] = args.force_valid_rot
    config['use_quat'] = args.use_quat
    config['use_aa'] = args.use_aa
    config['rot_matrix_regularization'] = args.rot_matrix_regularization
    config['no_normalization'] = args.no_normalization
    config['use_h36m_only'] = args.use_h36m_only
    config['use_h36m_martinez'] = args.use_h36m_martinez

    input_dropout = config['input_layer'].get('dropout_rate', 0)
    model_exp_name = ""
    if args.use_h36m_only:
        model_exp_name = "h36m"
    elif args.use_h36m_only:
        model_exp_name = "h36m_martinez"

    if args.model_type == "stcn":
        model_cls = models.STCN
        kld_weight = config['latent_layer']['kld_weight']
        kld_txt = str(int(kld_weight)) if isinstance(kld_weight, float) or isinstance(kld_weight, int) else "a"
        model_exp_name = "-kld_{}".format(kld_txt)
    elif args.model_type == "wavenet":
        model_cls = models.Wavenet
        if not(config['decoder_use_enc_skip'] or config['decoder_use_enc_last'] or config['decoder_use_raw_inputs']):
            config['decoder_use_enc_last'] = True
        model_exp_name = "-use"
        if config['decoder_use_enc_skip']:
            model_exp_name += "_skip"
        if config['decoder_use_enc_last']:
            model_exp_name += "_last"
        if config['decoder_use_raw_inputs']:
            model_exp_name += "_raw"
        del config["latent_layer"]
    elif args.model_type == "structured_stcn":
        model_cls = models.StructuredSTCN
    else:
        raise Exception()

    experiment_name_format = "{}-{}{}-{}-{}-{}{}{}-b{}-{}x{}@{}{}-in{}_out{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    "quat" if args.use_quat else "aa" if args.use_aa else "rotmat",
                                                    config['angle_loss_type'],
                                                    config['joint_prediction_model'],
                                                    model_exp_name,
                                                    "-idrop_" + str(input_dropout) if not isinstance(input_dropout, dict) and input_dropout > 0 else "",
                                                    config['batch_size'],
                                                    config['cnn_layer']['num_encoder_layers'],
                                                    config['cnn_layer']['num_filters'],
                                                    config['cnn_layer']['filter_size'],
                                                    '-residual_vel_{}'.format(args.residual_velocities_type) if args.residual_velocities else '',
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
    config['joint_prediction_model'] = args.joint_prediction_model  # "plain", "separate_joints", "fk_joints"
    config['architecture'] = args.architecture
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['rnn_size'] = args.cell_size
    config['num_layers'] = args.cell_layers
    config['grad_clip_by_norm'] = args.max_gradient_norm
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['learning_rate_decay_rate'] = args.learning_rate_decay_rate
    config['learning_rate_decay_type'] = args.learning_rate_decay_type
    config['autoregressive_input'] = args.autoregressive_input
    config['residual_velocities'] = args.residual_velocities
    config['residual_velocities_type'] = args.residual_velocities_type
    config['joint_prediction_model'] = args.joint_prediction_model  # currently ignored by seq2seq models
    config['output_layer'] = dict()
    config['output_layer']['num_layers'] = 0
    config['output_layer']['size'] = 128
    config['output_layer']['activation_fn'] = C.RELU
    config['angle_loss_type'] = args.angle_loss
    config['force_valid_rot'] = args.force_valid_rot
    config['rot_matrix_regularization'] = args.rot_matrix_regularization
    config['no_normalization'] = args.no_normalization
    config['use_quat'] = args.use_quat
    config['use_aa'] = args.use_aa
    config['use_h36m_only'] = args.use_h36m_only
    config['use_h36m_martinez'] = args.use_h36m_martinez

    model_exp_name = ""
    if args.use_h36m_only:
        model_exp_name = "h36m"
    elif args.use_h36m_only:
        model_exp_name = "h36m_martinez"

    if args.model_type == "seq2seq":
        model_cls = models.Seq2SeqModel
    elif args.model_type == "seq2seq_feedback":
        model_cls = models.Seq2SeqFeedbackModel
        config['feed_error_to_encoder'] = args.feed_error_to_encoder
    elif args.model_type == "simple_baseline":
        model_cls = models.ASimpleYetEffectiveBaseline
    else:
        raise ValueError("'{}' model unknown".format(args.model_type))

    experiment_name_format = "{}-{}-{}-{}-{}-{}-b{}-in{}_out{}-{}-enc{}feed-{}-depth{}-size{}-{}{}"
    experiment_name = experiment_name_format.format(experiment_timestamp,
                                                    args.model_type,
                                                    "" if args.experiment_name is None else args.experiment_name,
                                                    "quat" if args.use_quat else "aa" if args.use_aa else "rotmat",
                                                    model_exp_name,
                                                    config['angle_loss_type'],
                                                    config['batch_size'],
                                                    args.seq_length_in,
                                                    args.seq_length_out,
                                                    args.architecture,
                                                    '' if args.feed_error_to_encoder else 'no',
                                                    config['autoregressive_input'],
                                                    args.cell_layers,
                                                    args.cell_size,
                                                    'residual_vel_{}'.format(args.residual_velocities_type) if args.residual_velocities else 'not_residual_vel',
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
        if args.use_h36m_martinez:
            train_model, valid_model, test_model, srnn_model = models
            train_data, valid_data, test_data, srnn_data = data

            srnn_iter = srnn_data.get_iterator()
            srnn_pl = srnn_data.get_tf_samples()
            # iterate once over data to get all ground truth samples
            try:
                sess.run(srnn_iter.initializer)
                srnn_gts = dict()
                while True:
                    srnn_batch = sess.run(srnn_pl)
                    # Store each test sample and corresponding predictions with the unique sample IDs.
                    for k in range(srnn_batch["euler_targets"].shape[0]):
                        euler_targ = srnn_batch["euler_targets"][k]  # (window_size, 96)
                        euler_targ = euler_targ[-srnn_model.target_seq_len:]
                        srnn_gts[srnn_batch[C.BATCH_ID][k].decode("utf-8")] = euler_targ
            except tf.errors.OutOfRangeError:
                pass
        else:
            train_model, valid_model, test_model = models
            train_data, valid_data, test_data = data

        # Create metrics engine including summaries
        pck_threshs = C.METRIC_PCK_THRESHS  # thresholds for pck, in meters
        if args.use_h36m_martinez:
            fk_engine = H36MForwardKinematics()
            target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_MARTINEZ if x <= train_model.target_seq_len]
        else:
            target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= train_model.target_seq_len]
            fk_engine = SMPLForwardKinematics()
        metrics_engine = MetricsEngine(fk_engine,
                                       target_lengths,
                                       pck_threshs=pck_threshs,
                                       rep="quat" if train_model.use_quat else "aa" if train_model.use_aa else "rot_mat",
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
        best_valid_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False

        # Training loop configuration.
        time_counter = 0.0
        step = 1
        epoch = 0
        train_loss = 0.0
        train_iter = train_data.get_iterator()
        valid_iter = valid_data.get_iterator()
        test_iter = test_data.get_iterator()

        print("Running Training Loop.")
        # Assuming that we use initializable iterators.
        sess.run(train_iter.initializer)
        sess.run(valid_iter.initializer)

        def evaluate_model(_eval_model, _eval_iter, _metrics_engine, _return_results=False):
            # make a full pass on the validation or test dataset and compute the metrics
            _eval_result = dict()
            _start_time = time.perf_counter()
            _metrics_engine.reset()
            sess.run(_eval_iter.initializer)
            try:
                while True:
                    # get the predictions and ground truth values
                    prediction, targets, seed_sequence, data_id = _eval_model.sampled_step(sess)

                    # unnormalize - if normalization is not configured, these calls do nothing
                    p = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": prediction}, "poses")
                    t = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": targets}, "poses")
                    _metrics_engine.compute_and_aggregate(p["poses"], t["poses"])

                    if _return_results:
                        s = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": seed_sequence}, "poses")
                        # Store each test sample and corresponding predictions with the unique sample IDs.
                        for k in range(prediction.shape[0]):
                            _eval_result[data_id[k].decode("utf-8")] = (p["poses"][k], t["poses"][k], s["poses"][k])

            except tf.errors.OutOfRangeError:
                # finalize the computation of the metrics
                final_metrics = _metrics_engine.get_final_metrics()
            return final_metrics, time.perf_counter() - _start_time, _eval_result

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
                        print("Train [{:04d}] \t Loss: {:.3f} \t time/batch: {:.3f}".format(step,
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

            # Evaluation: make a full pass on the validation split.
            valid_metrics, valid_time, _ = evaluate_model(valid_model, valid_iter, metrics_engine)
            # print an informative string to the console
            print("Valid [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                      metrics_engine.get_summary_string(valid_metrics),
                                                                      valid_time))
            # get the summary feed dict
            summary_feed = metrics_engine.get_summary_feed_dict(valid_metrics)
            # get the writable summaries
            summaries = sess.run(metrics_engine.all_summaries_op, feed_dict=summary_feed)
            # write to log
            test_writer.add_summary(summaries, step)
            # reset the computation of the metrics
            metrics_engine.reset()
            # reset the validation iterator
            sess.run(valid_iter.initializer)

            # Early stopping check.
            valid_loss = valid_metrics[early_stopping_metric_key].sum()
            if (best_valid_loss - valid_loss) > np.abs(best_valid_loss*improvement_ratio):
                num_steps_wo_improvement = 0
            else:
                num_steps_wo_improvement += 1
            if num_steps_wo_improvement == args.early_stopping_tolerance:
                stop_signal = True

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Saving the model to {}".format(experiment_dir))
                saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=step-1)

        print("End of Training.")
        load_latest_checkpoint(sess, saver, experiment_dir)

        print("Evaluating validation set ...")
        valid_metrics, valid_time, _ = evaluate_model(valid_model, valid_iter, metrics_engine)
        print("Valid [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                  metrics_engine.get_summary_string(valid_metrics),
                                                                  valid_time))

        print("Evaluating test set ...")
        test_metrics, test_time, _ = evaluate_model(test_model, test_iter, metrics_engine)
        print("Test [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                 metrics_engine.get_summary_string(test_metrics),
                                                                 test_time))

        # create logger
        workbook_name = "motion_modelling_experiments"
        if args.use_h36m_only or args.use_h36m_martinez:
            workbook_name = "h36m_motion_modelling_experiments"
        gLogger = GoogleSheetLogger(credential_file=C.LOGGER_MANU,
                                    workbook_name=workbook_name)
        glog_data = {'Model ID': [os.path.split(experiment_dir)[-1].split('-')[0]],
                     'Model Name': ['-'.join(os.path.split(experiment_dir)[-1].split('-')[1:])],
                     'Comment': [""]}

        # gather the metrics and store them in the Google Sheet
        for t in metrics_engine.target_lengths:
            glog_valid_metrics = metrics_engine.get_summary_glogger(valid_metrics, until=t)
            glog_test_metrics = metrics_engine.get_summary_glogger(test_metrics, is_validation=False, until=t)

            glog_data["Comment"] = ["until_{}".format(t)]
            glog_data = {**glog_data, **glog_valid_metrics, **glog_test_metrics}
            gLogger.append_row(glog_data, sheet_name="until_{}".format(t))

        # compute metrics on SRNN poses to compare directly to Martinez
        def _evaluate_srnn_poses(_eval_model, _srnn_iter, _gt_euler):
            # make a full pass on the validation or test dataset and compute the metrics
            _start_time = time.perf_counter()
            sess.run(_srnn_iter.initializer)
            _euler_angle_metrics = dict()  # {action -> list of mean euler angles per frame}
            try:
                while True:
                    # get the predictions and ground truth values
                    prediction, _, seed_sequence, data_id = _eval_model.sampled_step(sess)

                    # unnormalize - if normalization is not configured, these calls do nothing
                    p = train_data.unnormalize_zero_mean_unit_variance_channel({"poses": prediction}, "poses")["poses"]
                    batch_size, seq_length = p.shape[0], p.shape[1]

                    # convert to euler angles
                    if train_model.use_quat:
                        # TODO(kamanuel) using function from quaternions, may be convert to rotmat like with aa
                        p_euler = quat2euler(np.reshape(p, [batch_size, seq_length, -1, 4]))
                    elif train_model.use_aa:
                        p_euler = rotmat2euler(aa2rotmat(np.reshape(p, [batch_size, seq_length, -1, 3])))
                    else:
                        p_euler = rotmat2euler(np.reshape(p, [batch_size, seq_length, -1, 3, 3]))

                    p_euler_padded = np.zeros([batch_size, seq_length, 32, 3])
                    p_euler_padded[:, :, H36M_MAJOR_JOINTS] = p_euler
                    p_euler_padded = np.reshape(p_euler_padded, [batch_size, seq_length, -1])

                    for k in range(batch_size):
                        _d_id = data_id[k].decode("utf-8")
                        _action = _d_id.split('/')[-1]
                        _targ = _gt_euler[_d_id]  # (seq_length, 96)
                        _pred = p_euler_padded[k]  # (seq_length, 96)

                        # compute euler loss like Martinez does it, but we don't have global translation
                        gt_i = np.copy(_targ)
                        gt_i[:, 0:3] = 0.0
                        _pred[:, 0:3] = 0.0

                        # compute the error only on the joints that we use for training
                        # only do this on ground truths, predictions are already sparse
                        idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

                        euc_error = np.power(gt_i[:, idx_to_use] - _pred[:, idx_to_use], 2)
                        euc_error = np.sum(euc_error, axis=1)
                        euc_error = np.sqrt(euc_error)  # (seq_length, )
                        if _action not in _euler_angle_metrics:
                            _euler_angle_metrics[_action] = [euc_error]
                        else:
                            _euler_angle_metrics[_action].append(euc_error)

            except tf.errors.OutOfRangeError:
                pass
            return _euler_angle_metrics, time.perf_counter() - _start_time

        if args.use_h36m_martinez:
            predictions_euler, _ = _evaluate_srnn_poses(srnn_model, srnn_iter, srnn_gts)

            which_actions = ['walking', 'eating', 'discussion', 'smoking']
            google_sheet_data = dict()
            for action in which_actions:
                # get the mean over all samples for that action
                assert len(predictions_euler[action]) == 8
                euler_mean = np.mean(np.stack(predictions_euler[action]), axis=0)

                # get the metrics for the timesteps, NOTE this assumes 50 Hz!!
                google_sheet_data[action] = {"80": euler_mean[3],
                                             "160": euler_mean[7],
                                             "320": euler_mean[15],
                                             "400": euler_mean[19]}

            g_logger = GoogleSheetLogger(credential_file=C.LOGGER_MANU,
                                         workbook_name="h36m_motion_modelling_experiments")
            glog_data = {'Model ID': [os.path.split(experiment_dir)[-1].split('-')[0]],
                         'Model Name': ['-'.join(os.path.split(experiment_dir)[-1].split('-')[1:])],
                         'Comment': [""]}

            which_actions = ["walking", "eating", "discussion", "smoking"]
            for action in which_actions:
                best_euler = google_sheet_data[action]
                for ms in best_euler:
                    glog_data[action[0] + ms] = [best_euler[ms]]

            g_logger.append_row(glog_data, sheet_name="logs")

        # save some sample videos to the experiment folder
        # TODO(kamanuel) this does not work on Leonhard for some stupid reason
        # visualizer = Visualizer("../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl", experiment_dir)
        # n_samples_viz = 5
        # rng = np.random.RandomState(42)
        # idxs = rng.randint(0, len(test_res), size=n_samples_viz)
        # sample_keys = [list(sorted(test_res.keys()))[i] for i in idxs]
        # for k in sample_keys:
        #     visualizer.visualize(test_res[k][2], test_res[k][0], test_res[k][1], title=k)

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
