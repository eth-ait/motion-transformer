"""
SPL: training and evaluation of neural networks with a structured prediction layer.
Copyright (C) 2019 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import glob
import json

import quaternion
import numpy as np
import tensorflow as tf

from common.constants import Constants as C
from spl.data.amass_tf import TFRecordMotionDataset
from spl.data.srnn_tf import SRNNTFRecordMotionDataset
from spl.model.zero_velocity import ZeroVelocityBaseline
from spl.model.rnn import RNN
from spl.model.seq2seq import Seq2SeqModel
from spl.model.transformer import Transformer2d
from spl.model.transformer_h36m import Transformer2d as Transformer2dH36M
# from spl.model.transformer2d_full_baseline import Transformer2d
from spl.model.vanilla import Transformer1d

from visualization.fk import H36MForwardKinematics
from visualization.fk import SMPLForwardKinematics
from visualization.fk import H36M_MAJOR_JOINTS

from metrics.motion_metrics import MetricsEngine
from common.conversions import rotmat2euler, aa2rotmat
from common.export_code import export_code


try:
    from common.logger import GoogleSheetLogger

    if "GLOGGER_WORKBOOK_AMASS" not in os.environ:
        raise ImportError("GLOGGER_WORKBOOK_AMASS not found.")
    if "GDRIVE_API_KEY" not in os.environ:
        raise ImportError("GDRIVE_API_KEY not found.")
    GLOGGER_AVAILABLE = True
except ImportError:
    GLOGGER_AVAILABLE = False
    print("GLogger not available...")

tf.app.flags.DEFINE_integer("seed", 1234, "Seed value.")
tf.app.flags.DEFINE_string("experiment_id", None, "Unique experiment id to restore an existing model.")
tf.app.flags.DEFINE_string("new_experiment_id", None, "Unique experiment id to start an experiment.")
tf.app.flags.DEFINE_string("data_dir", None,
                           "Path to data. If not passed, then AMASS_DATA environment variable is used.")
tf.app.flags.DEFINE_string("save_dir", None,
                           "Path to experiments. If not passed, then AMASS_EXPERIMENTS environment variable is used.")
tf.app.flags.DEFINE_string("from_config", None,
                           "Path to an existing config.json to start a new experiment.")
tf.app.flags.DEFINE_integer("print_frequency", 100, "Print/log every this many training steps.")
tf.app.flags.DEFINE_integer("test_frequency", 1000, "Runs validation every this many training steps.")
tf.app.flags.DEFINE_string("glog_comment", None, "A descriptive text for Google Sheet entry.")
# If from_config is used, the rest will be ignored.
# Data
tf.app.flags.DEFINE_enum("data_type", "rotmat", ["rotmat", "aa", "quat", "euler"],
                         "Which data representation: rotmat (rotation matrix), aa (angle axis), quat (quaternion).")
tf.app.flags.DEFINE_boolean("use_h36m", False, "Use H36M for training and validation.")
tf.app.flags.DEFINE_boolean("no_normalization", False, "If set, do not use zero-mean unit-variance normalization.")
tf.app.flags.DEFINE_float("random_noise_ratio", 0, "Random uniform noise on inputs.")
tf.app.flags.DEFINE_integer("source_seq_len", 120, "Number of frames to feed into the encoder.")
tf.app.flags.DEFINE_integer("target_seq_len", 24, "Number of frames that the decoder has to predict.")
tf.app.flags.DEFINE_integer("loss_seq_len", 0, "# of frames for training objective.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
# Training loop.
tf.app.flags.DEFINE_integer("num_epochs", 1000, "Training epochs.")
tf.app.flags.DEFINE_boolean("exhaustive_validation", False, "Use entire validation samples (takes much longer).")
tf.app.flags.DEFINE_integer("early_stopping_tolerance", 20, "# of waiting steps until the validation loss improves.")
# Optimization.
tf.app.flags.DEFINE_float("learning_rate", .001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.98, "Learning rate multiplier. See tf.exponential_decay.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 1000, "Decay steps. See tf.exponential_decay.")
tf.app.flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd"], "Optimization function type.")
tf.app.flags.DEFINE_float("grad_clip_norm", 1.0, "Clip gradients to this norm. If 0, gradient clipping is not applied.")
# Model
tf.app.flags.DEFINE_enum("model_type", "transformer2d", ["rnn", "seq2seq", "zero_velocity", "transformer2d", "transformer1d"],
                         "Which model to use.")
tf.app.flags.DEFINE_float("input_dropout_rate", 0, "Dropout rate on the inputs.")
tf.app.flags.DEFINE_integer("input_hidden_layers", 1, "# of hidden layers directly on the inputs.")
tf.app.flags.DEFINE_integer("input_hidden_size", 256, "Size of hidden layers directly on the inputs.")
tf.app.flags.DEFINE_enum("cell_type", "lstm", ["lstm", "gru"], "RNN cell type: gru or lstm.")
tf.app.flags.DEFINE_integer("cell_size", 1024, "RNN cell size.")
tf.app.flags.DEFINE_integer("cell_layers", 1, "Number of cells in the RNN model.")
tf.app.flags.DEFINE_boolean("residual_velocity", True, "Add a residual connection that effectively models velocities.")
tf.app.flags.DEFINE_enum("loss_type", "joint_sum", ["joint_sum", "all_mean", "geodesic"], "Joint-wise or vanilla mean loss.")
tf.app.flags.DEFINE_enum("joint_prediction_layer", "plain", ["spl", "spl_sparse", "plain"],
                         "Whether to use structured prediction layer (sparse or dense) "
                         "or a standard dense layer to make predictions.")
tf.app.flags.DEFINE_integer("output_hidden_layers", 1, "# of hidden layers in the prediction layer.")
tf.app.flags.DEFINE_integer("output_hidden_size", 64, "Size of hidden layers in the prediction layer. It is not scaled "
                                                      "based on joint_prediction_layer. If it is `plain`, "
                                                      "then it is recommended to pass a larger value (i.e., 960)")
# Only used by Seq2seq model.
tf.app.flags.DEFINE_enum("architecture", "tied", ["tied", "basic"], "If tied, encoder and decoder use the same cell.")
tf.app.flags.DEFINE_enum("autoregressive_input", "sampling_based", ["sampling_based", "supervised"],
                         "If sampling_based, decoder is trained with its predictions. More robust.")

# Only used by Transformer2d model.
tf.app.flags.DEFINE_integer("transformer_lr", 1, "Whether to use transformer learning rate or not")
tf.app.flags.DEFINE_integer("transformer_d_model", 128, "Size of d_model of the transformer")
tf.app.flags.DEFINE_float("transformer_dropout_rate", 0.1, "Dropout rate of the transformer")
tf.app.flags.DEFINE_integer("transformer_dff", 64, "Size of feed forward layer of the transformer")
tf.app.flags.DEFINE_integer("transformer_num_layers", 8, "Number of layers of the transformer")
tf.app.flags.DEFINE_integer("transformer_num_heads_temporal", 8, "Number of heads of the transformer's temporal block")
tf.app.flags.DEFINE_integer("transformer_num_heads_spacial", 8, "Number of heads of the transformer's spatial block")
tf.app.flags.DEFINE_integer("transformer_window_length", 120, "length of attention window of the transformer")
tf.app.flags.DEFINE_integer("warm_up_steps", 10000, "number of warm-up steps")
# They are for ablations and will go away.
tf.app.flags.DEFINE_boolean("shared_embedding_layer", False, "Whether to use a shared embedding layer instead of joint-specific layers or not.")
tf.app.flags.DEFINE_boolean("shared_output_layer", False, "-")
tf.app.flags.DEFINE_boolean("shared_temporal_layer", False, "-")
tf.app.flags.DEFINE_boolean("shared_spatial_layer", False, "-")
tf.app.flags.DEFINE_boolean("shared_attention_block", False, "-")
tf.app.flags.DEFINE_boolean("shared_pw_ffn", False, "-")
tf.app.flags.DEFINE_boolean("residual_attention_block", False, "-")
tf.app.flags.DEFINE_integer("random_window_min", 0, "-")
tf.app.flags.DEFINE_float("temporal_mask_drop", 0, "-")

# Positional encoding stuff.
tf.app.flags.DEFINE_boolean("abs_pos_encoding", False, "-")
tf.app.flags.DEFINE_boolean("temp_abs_pos_encoding", False, "-")
tf.app.flags.DEFINE_boolean("temp_rel_pos_encoding", False, "-")
tf.app.flags.DEFINE_boolean("shared_templ_kv", False, "-")
tf.app.flags.DEFINE_integer("max_relative_position", 50, "-")
tf.app.flags.DEFINE_enum("normalization_dim", "channel", ["channel", "all"], "Channel-wise or global normalization.")

args = tf.app.flags.FLAGS


def load_latest_checkpoint(sess, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def get_model_cls(model_type, is_h36m=False):
    if model_type == C.MODEL_ZERO_VEL:
        return ZeroVelocityBaseline
    elif model_type == C.MODEL_RNN:
        return RNN
    elif model_type == C.MODEL_SEQ2SEQ:
        return Seq2SeqModel
    elif model_type == C.MODEL_TRANS2D and is_h36m:
        return Transformer2dH36M
    elif model_type == C.MODEL_TRANS2D:
        return Transformer2d
    elif model_type == "transformer1d":
        return Transformer1d
    else:
        raise Exception("Unknown model type.")


def create_model(session):
    # Set experiment directory.
    save_dir = args.save_dir if args.save_dir else os.environ["AMASS_EXPERIMENTS"]

    # Load an existing config or initialize with command-line arguments.
    if args.experiment_id is not None:
        experiment_dir = glob.glob(os.path.join(save_dir, args.experiment_id + "-*"), recursive=False)[0]
        config = json.load(open(os.path.join(experiment_dir, "config.json"), "r"))
        model_cls = get_model_cls(config["model_type"], config["use_h36m"])
    else:
        # Initialize config and experiment name.
        if args.from_config is not None:
            from_config = json.load(open(args.from_config, "r"))
            model_cls = get_model_cls(from_config["model_type"], from_config["use_h36m"])
            
            # TODO(emre) quick hack to run with shorter seed sequence.
            if args.source_seq_len < 120:
                from_config["source_seq_len"] = args.source_seq_len
        else:
            from_config = None
            model_cls = get_model_cls(args.model_type, args.use_h36m)
        config, experiment_name = model_cls.get_model_config(args, from_config)
        experiment_dir = os.path.normpath(os.path.join(save_dir, experiment_name))
        os.mkdir(experiment_dir)

    tf.random.set_random_seed(config["seed"])
    print("Using model " + model_cls.__name__)

    # Set data paths.
    data_dir = args.data_dir if args.data_dir else os.environ["AMASS_DATA"]
    if config["use_h36m"]:
        data_dir = os.path.join(data_dir, '../h3.6m/tfrecords/')

    train_data_path = os.path.join(data_dir, config["data_type"], "training", "amass-?????-of-?????")
    test_data_path = os.path.join(data_dir, config["data_type"], "test", "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    
    # Exhaustive validation uses all motion windows extracted from the sequences Since it takes much longer, it is
    # advised to use the default validation procedure. It basically extracts a window randomly or from the center of a
    # motion sequence. The latter one is deterministic and reproducible.
    if config.get("exhaustive_validation", False):
        valid_data_path = os.path.join(data_dir, config["data_type"], "validation", "amass-?????-of-?????")
    else:
        valid_data_path = os.path.join(data_dir, config["data_type"], "validation_dynamic", "amass-?????-of-?????")

    # Data splits.
    # Each sample in training data is a full motion clip. We extract windows
    # of seed+target length randomly.
    
    # Set a fixed seed length of 2 seconds (120 and 50 frames for AMASS and H36M) datasets, respectively.
    # If the model uses shorter, it is clipped. This is required to ensure that
    # in shorter seed sequence cases, the validation and the test samples are still the same.
    default_seed_len = 120
    if config["use_h36m"]:
        default_seed_len = 50
    beginning_index = default_seed_len - config["source_seq_len"]
    
    with tf.name_scope("training_data"):
        window_length = config["source_seq_len"] + config["target_seq_len"]
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           extract_windows_of=window_length,
                                           window_type=C.DATA_WINDOW_RANDOM,
                                           num_parallel_calls=4,
                                           normalize=not config["no_normalization"],
                                           normalization_dim=config.get("normalization_dim", "channel"))
        train_pl = train_data.get_tf_samples()
    
    with tf.name_scope("validation_data"):
        if config.get("exhaustive_validation", False):
            window_length = 0
        valid_data = TFRecordMotionDataset(data_path=valid_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"] * 2,
                                           shuffle=False,
                                           extract_windows_of=window_length,
                                           window_type=C.DATA_WINDOW_CENTER,
                                           num_parallel_calls=4,
                                           normalize=not config["no_normalization"],
                                           normalization_dim=config.get("normalization_dim", "channel"))
        valid_pl = valid_data.get_tf_samples()
    
    with tf.name_scope("test_data"):
        window_length = config["source_seq_len"] + config["target_seq_len"]
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=config["batch_size"] * 2,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          window_type=C.DATA_WINDOW_BEGINNING,
                                          num_parallel_calls=4,
                                          normalize=not config["no_normalization"],
                                          normalization_dim=config.get("normalization_dim", "channel"),
                                          beginning_index=beginning_index)
        test_pl = test_data.get_tf_samples()

    # Models.
    with tf.name_scope(C.TRAIN):
        train_model = model_cls(
            config=config,
            data_pl=train_pl,
            mode=C.TRAIN,
            reuse=False)
        train_model.build_graph()

    with tf.name_scope(C.SAMPLE):
        valid_model = model_cls(
            config=config,
            data_pl=valid_pl,
            mode=C.SAMPLE,
            reuse=True)
        valid_model.build_graph()

    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=True)
        test_model.build_graph()

    # Return of this function.
    models = [train_model, valid_model, test_model]
    data = [train_data, valid_data, test_data]

    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    if config["use_h36m"]:
        # create model and data for SRNN evaluation
        with tf.name_scope("srnn_data"):
            srnn_dir = "srnn_poses_25fps"
            extract_windows_of = config["source_seq_len"] + config["target_seq_len"]
            srnn_path = os.path.join(data_dir, config["data_type"], srnn_dir, "amass-?????-of-?????")
            srnn_data = SRNNTFRecordMotionDataset(data_path=srnn_path,
                                                  meta_data_path=meta_data_path,
                                                  batch_size=config["batch_size"],
                                                  shuffle=False,
                                                  seed_len=config["source_seq_len"],
                                                  target_len=config["target_seq_len"],
                                                  # extract_windows_of=extract_windows_of,
                                                  # extract_random_windows=False,
                                                  num_parallel_calls=4,
                                                  normalize=not config["no_normalization"],
                                                  normalization_dim=config.get("normalization_dim", "channel"))
            srnn_pl = srnn_data.get_tf_samples()

        with tf.name_scope("SRNN"):
            srnn_model = model_cls(
                config=config,
                data_pl=srnn_pl,
                mode=C.SAMPLE,
                reuse=True,
                dtype=tf.float32)
            srnn_model.build_graph()

        models.append(srnn_model)
        data.append(srnn_data)

    num_param = 0
    for v in tf.trainable_variables():
        n_params = np.prod(v.shape.as_list())
        num_param += n_params
        # print(v.name, v.shape.as_list())
    print("# of parameters:", num_param)
    config["num_parameters"] = int(num_param)

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    print("Experiment directory: " + experiment_dir)
    json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)

    train_model.optimization_routines()
    train_model.summary_routines()
    valid_model.summary_routines()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3, save_relative_paths=True)

    # Initialize a new model or load a pre-trained one.
    if args.experiment_id is None:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return models, data, saver, global_step, experiment_dir, config

    # Load a pre-trained model.
    load_latest_checkpoint(session, saver, experiment_dir)
    return models, data, saver, global_step, experiment_dir, config


def evaluate_model(sess, _eval_model, _eval_iter, _metrics_engine,
                   undo_normalization_fn, _return_results=False):
    # make a full pass on the validation or test dataset and compute the metrics
    _eval_result = dict()
    _start_time = time.perf_counter()
    _metrics_engine.reset()
    sess.run(_eval_iter.initializer)
    try:
        while True:
            # Get the predictions and ground truth values
            prediction_steps = _eval_model.target_seq_len
            res = _eval_model.sampled_step(sess, prediction_steps=prediction_steps)
            if args.model_type == "transformer2d" or args.model_type == "transformer1d":
                prediction, targets, seed_sequence, data_id, attention = res
            else:
                prediction, targets, seed_sequence, data_id = res
            # Unnormalize predictions if there normalization applied.
            p = undo_normalization_fn(
                {"poses": prediction}, "poses")
            t = undo_normalization_fn(
                {"poses": targets}, "poses")
            _metrics_engine.compute_and_aggregate(p["poses"], t["poses"][:, :prediction_steps])

            if _return_results:
                s = undo_normalization_fn(
                    {"poses": seed_sequence}, "poses")
                # Store each test sample and corresponding predictions with
                # the unique sample IDs.
                for k in range(prediction.shape[0]):
                    _eval_result[data_id[k].decode("utf-8")] = (p["poses"][k],
                                                                t["poses"][k],
                                                                s["poses"][k])
    except tf.errors.OutOfRangeError:
        # finalize the computation of the metrics
        final_metrics = _metrics_engine.get_final_metrics()
    return final_metrics, time.perf_counter() - _start_time, _eval_result


def _evaluate_srnn_poses(sess, _eval_model, _srnn_iter, _gt_euler,
                         undo_normalization_fn):
    # compute the euler angle metric on the SRNN poses
    _start_time = time.perf_counter()
    sess.run(_srnn_iter.initializer)
    # {action -> list of mean euler angles per frame}
    _euler_angle_metrics = dict()
    try:
        while True:
            # get the predictions and ground truth values
            res = _eval_model.sampled_step(sess)
            if args.model_type == "transformer2d" or args.model_type == "transformer1d":
                prediction, targets, seed_sequence, data_id, attention = res
            else:
                prediction, targets, seed_sequence, data_id = res

            # Unnormalize predictions if there normalization applied.
            p = undo_normalization_fn(
                {"poses": prediction}, "poses")["poses"]
            batch_size, seq_length = p.shape[0], p.shape[1]

            # Convert to euler angles to calculate the error.
            # NOTE: these ground truth euler angles come from Martinez et al.,
            # so we shouldn't use quat2euler as this uses a different convention
            if _eval_model.use_quat:
                rot = quaternion.as_rotation_matrix(quaternion.from_float_array(
                    np.reshape(p, [batch_size, seq_length, -1, 4])))
                p_euler = rotmat2euler(rot)
            elif _eval_model.use_aa:
                p_euler = rotmat2euler(
                    aa2rotmat(np.reshape(p, [batch_size, seq_length, -1, 3])))
            elif _eval_model.use_rotmat:
                p_euler = rotmat2euler(
                    np.reshape(p, [batch_size, seq_length, -1, 3, 3]))
            else:
                p_euler = np.reshape(p, [batch_size, seq_length, -1, 3])

            p_euler_padded = np.zeros([batch_size, seq_length, 32, 3])
            p_euler_padded[:, :, H36M_MAJOR_JOINTS] = p_euler
            p_euler_padded = np.reshape(p_euler_padded,
                                        [batch_size, seq_length, -1])

            for k in range(batch_size):
                _d_id = data_id[k].decode("utf-8")
                _action = _d_id.split('/')[-1]
                _targ = _gt_euler[_d_id]  # (seq_length, 96)
                _pred = p_euler_padded[k]  # (seq_length, 96)

                # compute euler loss like Martinez does it,
                # but we don't have global translation
                gt_i = np.copy(_targ)
                gt_i[:, 0:3] = 0.0
                _pred[:, 0:3] = 0.0

                # compute the error only on the joints that we use for training
                # only do this on ground truths, predictions are already sparse
                idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]

                euc_error = np.power(gt_i[:, idx_to_use] - _pred[:, idx_to_use],
                                     2)
                euc_error = np.sum(euc_error, axis=1)
                euc_error = np.sqrt(euc_error)  # (seq_length, )
                if _action not in _euler_angle_metrics:
                    _euler_angle_metrics[_action] = [euc_error]
                else:
                    _euler_angle_metrics[_action].append(euc_error)
    except tf.errors.OutOfRangeError:
        pass
    return _euler_angle_metrics, time.perf_counter() - _start_time


def train():
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Create the model
        models, data, saver, global_step, experiment_dir, config = create_model(sess)
        code_files = glob.glob('**/*.py', recursive=True)
        export_code(code_files, os.path.join(experiment_dir, 'code.zip'))

        # If it is h36m data, iterate once over entire dataset to load all
        # ground-truth samples.
        if config["use_h36m"]:
            train_model, valid_model, test_model, srnn_model = models
            train_data, valid_data, test_data, srnn_data = data

            srnn_iter = srnn_data.get_iterator()
            srnn_pl = srnn_data.get_tf_samples()

            try:
                sess.run(srnn_iter.initializer)
                srnn_gts = dict()
                while True:
                    srnn_batch = sess.run(srnn_pl)
                    # Store each test sample and corresponding predictions
                    # with the unique sample IDs.
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
        pck_thresholds = C.METRIC_PCK_THRESHS  # thresholds for pck, in meters
        if config["use_h36m"]:
            fk_engine = H36MForwardKinematics()
            tls = C.METRIC_TARGET_LENGTHS_H36M_25FPS
            target_lengths = [x for x in tls if x <= valid_model.target_seq_len]
        else:
            target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= valid_model.target_seq_len]
            fk_engine = SMPLForwardKinematics()
        
        if not config["use_h36m"]:
            metrics_engine = MetricsEngine(fk_engine,
                                           target_lengths,
                                           pck_threshs=pck_thresholds,
                                           rep=config["data_type"],
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

        # Google logging.
        if GLOGGER_AVAILABLE:
            glogger_workbook = os.environ["GLOGGER_WORKBOOK_AMASS"]
            gdrive_key = os.environ["GDRIVE_API_KEY"]
            model_name = '-'.join(
                os.path.split(experiment_dir)[-1].split('-')[1:])
            static_values = dict()
            static_values["Model ID"] = config["experiment_id"]
            static_values["Model Name"] = model_name
            if args.glog_comment is not None:
                static_values["Comment"] = args.glog_comment

            if config["use_h36m"]:
                sheet_name = "h36m"
            else:
                sheet_name = "until_{}".format(18)
            
            credentials = tf.gfile.Open(gdrive_key, "r")
            glogger = GoogleSheetLogger(
                credentials,
                glogger_workbook,
                sheet_names=[sheet_name],
                model_identifier=config["experiment_id"],
                static_values=static_values)

        # Early stopping configuration.
        early_stopping_metric_key = C.METRIC_JOINT_ANGLE
        # Defines the ratio of improvement required wrt the previous evaluation.
        improvement_ratio = 0.01
        best_valid_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False
        checkpoint_step = 0
        # Print results for 400 ms horizon.
        summary_at_frame = 10 if config["use_h36m"] else 24

        # Training loop configuration.
        time_counter = 0.0
        step = 0
        epoch = 0
        print_steps = 0
        train_loss = 0.0
        train_iter = train_data.get_iterator()
        valid_iter = valid_data.get_iterator()
        test_iter = test_data.get_iterator()

        print("Running Training Loop.")
        # Assuming that we use initializable iterators.
        sess.run(train_iter.initializer)
        sess.run(valid_iter.initializer)

        undo_norm_fn = train_data.unnormalization_func
        train_str = "Train [{:04d}] \t Loss: {:.3f} \t time/batch: {:.3f}"
        valid_str = "Valid [{:04d}] \t {} \t total_time: {:.3f}"
        while not stop_signal:
            # Training.
            for i in range(args.test_frequency):
                try:
                    start_time = time.perf_counter()
                    step += 1
                    print_steps += 1
                    
                    if step % args.print_frequency == 0:
                        train_loss_avg = train_loss / print_steps
                        time_elapsed = time_counter / print_steps
                        train_loss, time_counter, print_steps = 0., 0., 0
                        print(train_str.format(step,
                                               train_loss_avg,
                                               time_elapsed))
                    
                    step_loss, summary, _ = train_model.step(sess)
                    train_writer.add_summary(summary, step)
                    train_loss += step_loss

                    time_counter += (time.perf_counter() - start_time)
                except tf.errors.OutOfRangeError:
                    sess.run(train_iter.initializer)
                    epoch += 1
                    if epoch >= config["num_epochs"]:
                        stop_signal = True
                        break
            
            if config["use_h36m"]:
                # do early stopping based on euler angle loss
                predictions_euler, _ = _evaluate_srnn_poses(sess, srnn_model,
                                                            srnn_iter, srnn_gts,
                                                            undo_norm_fn)
                selected_actions_mean_error = []
                es_actions = ['walking', 'eating', 'discussion', 'smoking']
                for action in es_actions:
                    selected_actions_mean_error.append(np.stack(predictions_euler[action]))

                srnn_valid_loss = np.mean(np.concatenate(selected_actions_mean_error, axis=0), 0)[[1, 3, 7, 9]].mean()
                print("Euler angle valid loss on SRNN samples: {}".format(srnn_valid_loss))
                valid_loss = srnn_valid_loss
            
            else:
                # Evaluation: make a full pass on the validation split.
                valid_metrics, valid_time, _ = evaluate_model(sess, valid_model,
                                                              valid_iter,
                                                              metrics_engine,
                                                              undo_norm_fn)
                # print an informative string to the console
                valid_log = metrics_engine.get_summary_string_all(valid_metrics,
                                                                  [summary_at_frame],
                                                                  pck_thresholds)
                print(valid_str.format(step,
                                       valid_log,
                                       valid_time))
                # get the summary feed dict
                summary_feed = metrics_engine.get_summary_feed_dict(valid_metrics)
                # get the writable summaries
                summaries = sess.run(metrics_engine.all_summaries_op,
                                     feed_dict=summary_feed)
                # write to log
                test_writer.add_summary(summaries, step)
                # reset the computation of the metrics
                metrics_engine.reset()
                # reset the validation iterator
                sess.run(valid_iter.initializer)
    
                # Early stopping check.
                # valid_loss = valid_metrics[early_stopping_metric_key].sum()
                valid_loss = valid_metrics[early_stopping_metric_key][:summary_at_frame].sum()

            # Check if the improvement is good enough. If not, we wait to see
            # if there is an improvement (i.e., early_stopping_tolerance).
            if (best_valid_loss - valid_loss) > np.abs(best_valid_loss * improvement_ratio):
                num_steps_wo_improvement = 0
            else:
                num_steps_wo_improvement += 1
            if num_steps_wo_improvement == config.get("early_stopping_tolerance", 20):
                stop_signal = True

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Saving the model to {}".format(experiment_dir))
                saver.save(sess, os.path.normpath(
                    os.path.join(experiment_dir, 'checkpoint')),
                           global_step=step)
                checkpoint_step = step

                # If there is a new checkpoint, log the result.
                if GLOGGER_AVAILABLE:
                    # Note that h3.6m validation performance can be logged wrt.
                    # time-steps similar to AMASS.
                    if config["use_h36m"]:
                        which_actions = ['walking', 'eating', 'discussion', 'smoking']
                        log_data = dict()
                        for action in which_actions:
                            # get the mean over all samples for that action
                            assert len(predictions_euler[action]) == 8
                            euler_mean = np.mean(
                                np.stack(predictions_euler[action]), axis=0)

                            log_data[action[0] + "80"] = euler_mean[1]
                            log_data[action[0] + "160"] = euler_mean[3]
                            log_data[action[0] + "320"] = euler_mean[7]
                            log_data[action[0] + "400"] = euler_mean[9]
                            if euler_mean.shape[0] > 12:
                                log_data[action[0] + "560"] = euler_mean[13]
                                log_data[action[0] + "1000"] = euler_mean[24]
                                
                        log_data["Step"] = checkpoint_step
                        glogger.update_or_append_row(log_data, "h36m")
                    else:
                        for t in metrics_engine.target_lengths:
                            valid_ = metrics_engine.get_metrics_until(
                                valid_metrics,
                                t,
                                pck_thresholds,
                                prefix="val ")
                            valid_["Step"] = checkpoint_step
                            glogger.update_or_append_row(valid_,
                                                         "until_{}".format(t))

        print("End of Training.")
        load_latest_checkpoint(sess, saver, experiment_dir)

        if not config["use_h36m"]:
            print("Evaluating validation set...")
            valid_metrics, valid_time, _ = evaluate_model(sess,
                                                          valid_model,
                                                          valid_iter,
                                                          metrics_engine,
                                                          undo_norm_fn)
            valid_log = metrics_engine.get_summary_string_all(valid_metrics,
                                                              [summary_at_frame],
                                                              pck_thresholds)
            print(valid_str.format(step,
                                   valid_log,
                                   valid_time))

            print("Evaluating test set...")
            test_metrics, test_time, _ = evaluate_model(sess,
                                                        test_model,
                                                        test_iter,
                                                        metrics_engine,
                                                        undo_norm_fn)
            test_str = "Test [{:04d}] \t {} \t total_time: {:.3f}"
            print(test_str.format(step,
                                  metrics_engine.get_summary_string_all(
                                      test_metrics,
                                      [summary_at_frame],
                                      pck_thresholds),
                                  test_time))
            if GLOGGER_AVAILABLE:
                for t in metrics_engine.target_lengths:
                    test_ = metrics_engine.get_metrics_until(test_metrics,
                                                             t,
                                                             pck_thresholds,
                                                             prefix="test ")
                    test_["Step"] = checkpoint_step
                    glogger.update_or_append_row(test_,
                                                 "until_{}".format(t))

        else:
            predictions_euler, _ = _evaluate_srnn_poses(sess, srnn_model,
                                                        srnn_iter,
                                                        srnn_gts, undo_norm_fn)
            log_data = dict()
            which_actions = ['walking', 'eating', 'discussion', 'smoking']

            print("{:<10}".format(""), end="")
            for ms in [80, 160, 320, 400]:
                print("  {0:4d}  ".format(ms), end="")

            if predictions_euler[which_actions[0]][0].shape[0] > 12:
                for ms in [560, 1000]:
                    print("  {0:4d}  ".format(ms), end="")
            print()
            
            test_str = " {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}"
            long_test_str = " \t{:.3f} \t{:.3f}"
            
            for action in which_actions:
                # get the mean over all samples for that action
                assert len(predictions_euler[action]) == 8
                euler_mean = np.mean(np.stack(predictions_euler[action]), axis=0)
                s = "{:<10}:".format(action)

                # get the metrics at the time-steps:
                s += test_str.format(euler_mean[1],
                                     euler_mean[3],
                                     euler_mean[7],
                                     euler_mean[9])
                if euler_mean.shape[0] > 12:
                    s += long_test_str.format(euler_mean[13],
                                              euler_mean[24])
                print(s)

                log_data[action[0] + "80"] = euler_mean[1]
                log_data[action[0] + "160"] = euler_mean[3]
                log_data[action[0] + "320"] = euler_mean[7]
                log_data[action[0] + "400"] = euler_mean[9]
                if euler_mean.shape[0] > 12:
                    log_data[action[0] + "560"] = euler_mean[13]
                    log_data[action[0] + "1000"] = euler_mean[24]

            if GLOGGER_AVAILABLE:
                log_data["Step"] = checkpoint_step
                glogger.update_or_append_row(log_data, "h36m")

        print("\nDone!")


def main(argv):
    train()


if __name__ == "__main__":
    tf.app.run()
