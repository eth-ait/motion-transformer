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

from visualization.fk import H36MForwardKinematics
from visualization.fk import SMPLForwardKinematics
from visualization.fk import H36M_MAJOR_JOINTS

from metrics.motion_metrics import MetricsEngine
from common.conversions import rotmat2euler, aa2rotmat


tf.app.flags.DEFINE_integer("seed", 1234, "Seed value.")
tf.app.flags.DEFINE_string("experiment_id", None, "Unique experiment id to restore an existing model.")
tf.app.flags.DEFINE_string("data_dir", None,
                           "Path to data. If not passed, then AMASS_DATA environment variable is used.")
tf.app.flags.DEFINE_string("save_dir", None,
                           "Path to experiments. If not passed, then AMASS_EXPERIMENTS environment variable is used.")
tf.app.flags.DEFINE_string("from_config", None,
                           "Path to an existing config.json to start a new experiment.")
tf.app.flags.DEFINE_integer("print_frequency", 100, "Print/log every this many training steps.")
tf.app.flags.DEFINE_integer("test_frequency", 1000, "Runs validation every this many training steps.")
# If from_config is used, the rest will be ignored.
# Data
tf.app.flags.DEFINE_enum("data_type", "rotmat", ["rotmat", "aa", "quat"],
                         "Which data representation: rotmat (rotation matrix), aa (angle axis), quat (quaternion).")
tf.app.flags.DEFINE_boolean("use_h36m", False, "Use H36M for training and validation.")
tf.app.flags.DEFINE_boolean("no_normalization", False, "If set, do not use zero-mean unit-variance normalization.")
tf.app.flags.DEFINE_integer("source_seq_len", 120, "Number of frames to feed into the encoder.")
tf.app.flags.DEFINE_integer("target_seq_len", 24, "Number of frames that the decoder has to predict.")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
# Training loop.
tf.app.flags.DEFINE_integer("num_epochs", 500, "Training epochs.")
tf.app.flags.DEFINE_boolean("exhaustive_validation", False, "Use entire validation samples (takes much longer).")
tf.app.flags.DEFINE_integer("early_stopping_tolerance", 20, "# of waiting steps until the validation loss improves.")
# Optimization.
tf.app.flags.DEFINE_float("learning_rate", .001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_rate", 0.98, "Learning rate multiplier. See tf.exponential_decay.")
tf.app.flags.DEFINE_integer("learning_rate_decay_steps", 1000, "Decay steps. See tf.exponential_decay.")
tf.app.flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd"], "Optimization function type.")
tf.app.flags.DEFINE_float("grad_clip_norm", 1.0, "Clip gradients to this norm. If 0, gradient clipping is not applied.")
# Model
tf.app.flags.DEFINE_enum("model_type", "rnn", ["rnn", "seq2seq", "zero_velocity"], "Which model to use.")
tf.app.flags.DEFINE_float("input_dropout_rate", 0.1, "Dropout rate on the inputs.")
tf.app.flags.DEFINE_integer("input_hidden_layers", 1, "# of hidden layers directly on the inputs.")
tf.app.flags.DEFINE_integer("input_hidden_size", 256, "Size of hidden layers directly on the inputs.")
tf.app.flags.DEFINE_enum("cell_type", "lstm", ["lstm", "gru"], "RNN cell type: gru or lstm.")
tf.app.flags.DEFINE_integer("cell_size", 1024, "RNN cell size.")
tf.app.flags.DEFINE_integer("cell_layers", 1, "Number of cells in the RNN model.")
tf.app.flags.DEFINE_boolean("residual_velocity", True, "Add a residual connection that effectively models velocities.")
tf.app.flags.DEFINE_enum("loss_type", "joint_sum", ["joint_sum", "all_mean"], "Joint-wise or vanilla mean loss.")
tf.app.flags.DEFINE_enum("joint_prediction_layer", "spl", ["spl", "spl_sparse", "plain"],
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


def get_model_cls(model_type):
    if model_type == C.MODEL_ZERO_VEL:
        return ZeroVelocityBaseline
    elif model_type == C.MODEL_RNN:
        return RNN
    elif model_type == C.MODEL_SEQ2SEQ:
        return Seq2SeqModel
    else:
        raise Exception("Unknown model type.")
    

def create_model(session):
    # Set experiment directory.
    save_dir = args.save_dir if args.save_dir else os.environ["AMASS_EXPERIMENTS"]
    if args.use_h36m:
        save_dir = os.path.join(save_dir, '../', 'experiments_h36m')
    
    # Load an existing config or initialize one based on the command-line arguments.
    if args.experiment_id is not None:
        experiment_dir = glob.glob(os.path.join(save_dir, args.experiment_id + "-*"), recursive=False)[0]
        config = json.load(open(os.path.join(experiment_dir, "config.json"), "r"))
        model_cls = get_model_cls(config["model_type"])
    else:
        # Initialize config and experiment name.
        if args.from_config is not None:
            from_config = json.load(open(args.from_config, "r"))
            model_cls = get_model_cls(from_config["model_type"])
        else:
            from_config = None
            model_cls = get_model_cls(args.model_type)
        config, experiment_name = model_cls.get_model_config(args, from_config)
        experiment_dir = os.path.normpath(os.path.join(save_dir, experiment_name))
        os.mkdir(experiment_dir)

    tf.random.set_random_seed(config["seed"])
    
    # Set data paths.
    data_dir = args.data_dir if args.data_dir else os.environ["AMASS_DATA"]
    if config["use_h36m"]:
        data_dir = os.path.join(data_dir, '../../h3.6m/tfrecords/')

    train_data_path = os.path.join(data_dir, config["data_type"], "training", "amass-?????-of-?????")
    test_data_path = os.path.join(data_dir, config["data_type"], "test", "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    
    # Exhaustive validation uses all motion windows extracted from the sequences. Since it takes much longer, it is
    # advised to use the default validation procedure. It basically extracts a window randomly or from the center of a
    # motion sequence. The latter one is deterministic and reproducible.
    if config.get("exhaustive_validation", False):
        valid_data_path = os.path.join(data_dir, config["data_type"], "validation", "amass-?????-of-?????")
    else:
        valid_data_path = os.path.join(data_dir, config["data_type"], "validation_dynamic", "amass-?????-of-?????")

    # Data splits.
    # Each sample in training data is a full motion clip. We extract windows of seed+target length randomly.
    window_length = config["source_seq_len"] + config["target_seq_len"]
    with tf.name_scope("training_data"):
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"],
                                           shuffle=True,
                                           extract_windows_of=window_length,
                                           window_type=C.DATA_WINDOW_RANDOM,
                                           num_parallel_calls=4,
                                           normalize=not config["no_normalization"])
        train_pl = train_data.get_tf_samples()

    if config.get("exhaustive_validation", False):
        window_length = 0
        assert window_length <= 180, "TFRecords are hardcoded with length of 180."
    
    with tf.name_scope("validation_data"):
        valid_data = TFRecordMotionDataset(data_path=valid_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"]*2,
                                           shuffle=False,
                                           extract_windows_of=window_length,
                                           window_type=C.DATA_WINDOW_CENTER,
                                           num_parallel_calls=4,
                                           normalize=not config["no_normalization"])
        valid_pl = valid_data.get_tf_samples()

    with tf.name_scope("test_data"):
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=256,  # to speedup inference
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          window_type=C.DATA_WINDOW_BEGINNING,
                                          num_parallel_calls=4,
                                          normalize=not config["no_normalization"])
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
            extract_windows_of = 60
            srnn_path = os.path.join(data_dir, config["data_type"], srnn_dir, "amass-?????-of-?????")
            srnn_data = SRNNTFRecordMotionDataset(data_path=srnn_path,
                                                  meta_data_path=meta_data_path,
                                                  batch_size=config["batch_size"],
                                                  shuffle=False,
                                                  extract_windows_of=extract_windows_of,
                                                  extract_random_windows=False,
                                                  num_parallel_calls=4,
                                                  normalize=not config["no_normalization"])
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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
    
    # Initialize a new model or load a pre-trained one.
    if args.experiment_id is None:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return models, data, saver, global_step, experiment_dir, config

    # Load a pre-trained model.
    load_latest_checkpoint(session, saver, experiment_dir)
    return models, data, saver, global_step, experiment_dir, config


def train():
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Create the model
        models, data, saver, global_step, experiment_dir, config = create_model(sess)
        
        # If it is h36m data, iterate once over entire dataset to load all ground-truth samples.
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
        pck_thresholds = C.METRIC_PCK_THRESHS  # thresholds for pck, in meters
        if config["use_h36m"]:
            fk_engine = H36MForwardKinematics()
            tls = C.METRIC_TARGET_LENGTHS_H36M_25FPS
            target_lengths = [x for x in tls if x <= train_model.target_seq_len]
        else:
            target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= train_model.target_seq_len]
            fk_engine = SMPLForwardKinematics()
        
        metrics_engine = MetricsEngine(fk_engine,
                                       target_lengths,
                                       pck_threshs=pck_thresholds,
                                       rep=C.QUATERNION if train_model.use_quat else C.ANGLE_AXIS if train_model.use_aa else C.ROT_MATRIX,
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
        improvement_ratio = 0.01  # defines the significant amount of improvement wrt the previous evaluation.
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

        def _evaluate_srnn_poses(_eval_model, _srnn_iter, _gt_euler):
            # compute the euler angle metric on the SRNN poses
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

                    # convert to euler angles to calculate the error.
                    # NOTE: these ground truth euler angles come from Martinez et al., so we shouldn't use quat2euler
                    # as this uses a different convention
                    if train_model.use_quat:
                        rot = quaternion.as_rotation_matrix(quaternion.from_float_array(np.reshape(p, [batch_size, seq_length, -1, 4])))
                        p_euler = rotmat2euler(rot)
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

        while not stop_signal:
            # Training.
            for i in range(args.test_frequency):
                try:
                    start_time = time.perf_counter()
                    step += 1

                    step_loss, summary, _ = train_model.step(sess)
                    train_writer.add_summary(summary, step)
                    train_loss += step_loss

                    time_counter += (time.perf_counter() - start_time)
                    if step % args.print_frequency == 0:
                        train_loss_avg = train_loss / args.print_frequency
                        time_elapsed = time_counter/args.print_frequency
                        train_loss, time_counter = 0., 0.
                        print("Train [{:04d}] \t Loss: {:.3f} \t time/batch: {:.3f}".format(step,
                                                                                            train_loss_avg,
                                                                                            time_elapsed))
                except tf.errors.OutOfRangeError:
                    sess.run(train_iter.initializer)
                    epoch += 1
                    if epoch >= config["num_epochs"]:
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

            if config["use_h36m"]:
                # do early stopping based on euler angle loss
                predictions_euler, _ = _evaluate_srnn_poses(srnn_model, srnn_iter, srnn_gts)
                selected_actions_mean_error = []

                for action in ['walking', 'eating', 'discussion', 'smoking']:
                    selected_actions_mean_error.append(np.stack(predictions_euler[action]))

                valid_loss = np.mean(np.concatenate(selected_actions_mean_error, axis=0))
                print("Euler angle valid loss: {}".format(valid_loss))
            
            # Check if the improvement is good enough. If not, we wait for some evaluation
            # turns (i.e., early_stopping_tolerance).
            if (best_valid_loss - valid_loss) > np.abs(best_valid_loss*improvement_ratio):
                num_steps_wo_improvement = 0
            else:
                num_steps_wo_improvement += 1
            if num_steps_wo_improvement == config.get("early_stopping_tolerance", 20):
                stop_signal = True

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Saving the model to {}".format(experiment_dir))
                saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=step-1)

        print("End of Training.")
        load_latest_checkpoint(sess, saver, experiment_dir)

        if not config["use_h36m"]:
            print("Evaluating validation set...")
            valid_metrics, valid_time, _ = evaluate_model(valid_model, valid_iter, metrics_engine)
            print("Valid [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                      metrics_engine.get_summary_string(valid_metrics),
                                                                      valid_time))
            
            print("Evaluating test set...")
            test_metrics, test_time, _ = evaluate_model(test_model, test_iter, metrics_engine)
            print("Test [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                     metrics_engine.get_summary_string_all(
                                                                         test_metrics,
                                                                         target_lengths,
                                                                         pck_thresholds),
                                                                     test_time))
        else:
            predictions_euler, _ = _evaluate_srnn_poses(srnn_model, srnn_iter, srnn_gts)
            which_actions = ['walking', 'eating', 'discussion', 'smoking']

            print("{:<10}".format(""), end="")
            for ms in [80, 160, 320, 400]:
                print("  {0:4d}  ".format(ms), end="")
            print()
            for action in which_actions:
                # get the mean over all samples for that action
                assert len(predictions_euler[action]) == 8
                euler_mean = np.mean(np.stack(predictions_euler[action]), axis=0)
                s = "{:<10}:".format(action)
        
                # get the metrics at the time-steps:
                at_idxs = [1, 3, 7, 9]
                s += " {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}".format(euler_mean[at_idxs[0]],
                                                                 euler_mean[at_idxs[1]],
                                                                 euler_mean[at_idxs[2]],
                                                                 euler_mean[at_idxs[3]])
                print(s)
        print("\nDone!")


def main(argv):
    train()


if __name__ == "__main__":
    tf.app.run()
