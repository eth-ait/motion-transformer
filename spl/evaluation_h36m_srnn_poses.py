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
import os
import glob
import json
import time
import argparse

import quaternion
import numpy as np
import tensorflow as tf

from spl.data.srnn_tf import SRNNTFRecordMotionDataset
from spl.model.zero_velocity import ZeroVelocityBaseline
from spl.model.rnn import RNN
from spl.model.seq2seq import Seq2SeqModel
from spl.model.transformer_h36m import Transformer2d
from spl.model.vanilla import Transformer1d

from common.constants import Constants as C
from visualization.fk import H36M_MAJOR_JOINTS
from common.conversions import rotmat2euler, aa2rotmat

import matplotlib.pyplot as plt

plt.switch_backend('agg')


sample_keys_h36m = [
        "h36/0/S9_walkingd",
        "h36/0/S7_discussi",
        "h36/0/S9_smoki",
        "h36/0/S6_walkingd",
        "h36/0/S11_sitti",
        "h36/0/S11_walkingtogeth"
        ]

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


def load_latest_checkpoint(session, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        raise (ValueError, "Checkpoint {0} does not seem to exist".format(ckpt.model_checkpoint_path))


def get_model_cls(model_type, is_h36m=False):
    if model_type == C.MODEL_ZERO_VEL:
        return ZeroVelocityBaseline
    elif model_type == C.MODEL_RNN:
        return RNN
    elif model_type == C.MODEL_SEQ2SEQ:
        return Seq2SeqModel
    elif model_type == C.MODEL_TRANS2D:
        return Transformer2d
    elif model_type == "transformer1d":
        return Transformer1d
    else:
        raise Exception("Unknown model type.")


def create_and_restore_model(session, experiment_dir, data_dir, config):
    model_cls = get_model_cls(config["model_type"], config["use_h36m"])
    print("Using model " + str(model_cls))

    data_dir = os.path.join(data_dir, '../h3.6m/tfrecords/')
    srnn_dir = "srnn_poses_25fps"
    srnn_path = os.path.join(data_dir, config["data_type"], srnn_dir, "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    print("Loading H3.6M (SRNN poses) test data from " + srnn_path)
    
    # Create model and data for SRNN evaluation
    with tf.name_scope("srnn_data"):
        srnn_data = SRNNTFRecordMotionDataset(data_path=srnn_path,
                                              meta_data_path=meta_data_path,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              seed_len=config["source_seq_len"],
                                              target_len=config["target_seq_len"],
                                              num_parallel_calls=4,
                                              normalize=not config["no_normalization"],
                                              normalization_dim=config.get("normalization_dim", "channel"),
                                              use_std_norm=config.get("use_std_norm", False),)
        srnn_pl = srnn_data.get_tf_samples()

    with tf.name_scope("SRNN"):
        srnn_model = model_cls(
            config=config,
            data_pl=srnn_pl,
            mode=C.SAMPLE,
            reuse=False,
            dtype=tf.float32)
        srnn_model.build_graph()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    # Restore model parameters.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
    load_latest_checkpoint(session, saver, experiment_dir)
    return srnn_model, srnn_data


def rotmat_to_euler_padded(samples):
    batch_size, seq_length, _ = samples.shape
    p_euler = rotmat2euler(np.reshape(samples, [batch_size, seq_length, -1, 3, 3]))
    p_euler_padded = np.zeros([batch_size, seq_length, 32, 3])
    p_euler_padded[:, :, H36M_MAJOR_JOINTS] = p_euler
    p_euler_padded = np.reshape(p_euler_padded, [batch_size, seq_length, -1])
    p_euler_padded[:, :, 0:3] = 0
    return p_euler_padded


def _evaluate_srnn_poses(session, _eval_model, _srnn_iter, _srnn_pl,
                         undo_normalization_fn):
    # Get GT Euler
    try:
        sess.run(_srnn_iter.initializer)
        _gt_euler = dict()
        while True:
            srnn_batch = sess.run(_srnn_pl)
            # Store each test sample and corresponding predictions
            # with the unique sample IDs.
            for k in range(srnn_batch["euler_targets"].shape[0]):
                euler_targ = srnn_batch["euler_targets"][k]  # (window_size, 96)
                _gt_euler[
                    srnn_batch[C.BATCH_ID][k].decode("utf-8")] = euler_targ
    except tf.errors.OutOfRangeError:
        pass
    
    using_attention_model = False
    if isinstance(_eval_model, Transformer2d):
        print("Using Attention Model.")
        using_attention_model = True
    
    # compute the euler angle metric on the SRNN poses
    _start_time = time.perf_counter()
    session.run(_srnn_iter.initializer)
    # {action -> list of mean euler angles per frame}
    _euler_angle_metrics = dict()
    _eval_result_euler = dict()
    _eval_result = dict()
    n_batches = 0
    try:
        while True:
            # get the predictions and ground truth values
            res = _eval_model.sampled_step(session,
                                           prediction_steps=_args.seq_length_out)
            if using_attention_model:
                prediction, targets, seed_sequence, data_id, attention = res
            else:
                prediction, targets, seed_sequence, data_id = res
            
            # Unnormalize predictions if there normalization applied.
            p = undo_normalization_fn(
                {"poses": prediction}, "poses")["poses"]
            t = undo_normalization_fn(
                {"poses": targets}, "poses")["poses"]
            s = undo_normalization_fn(
                {"poses": seed_sequence}, "poses")["poses"]
            
            batch_size, seq_length = p.shape[0], p.shape[1]
            
            # Convert to euler angles to calculate the error.
            # NOTE: these ground truth euler angles come from Martinez et al.,
            # so we shouldn't use quat2euler as this uses a different convention
            if _eval_model.use_quat:
                rot = quaternion.as_rotation_matrix(quaternion.from_float_array(np.reshape(p, [batch_size, seq_length, -1, 4])))
                p_euler = rotmat2euler(rot)
            elif _eval_model.use_aa:
                p_euler = rotmat2euler(aa2rotmat(np.reshape(p, [batch_size, seq_length, -1, 3])))
            else:
                p_euler_padded = rotmat_to_euler_padded(p)
                t_euler_padded = rotmat_to_euler_padded(t)
                s_euler_padded = rotmat_to_euler_padded(s)
            
            idx_to_use = np.where(np.reshape(t_euler_padded, [-1, 96]).std(0) > 1e-4)[0]
            idx_to_ignore = np.where(np.reshape(t_euler_padded, [-1, 96]).std(0) < 1e-4)[0]
            
            p_euler_padded[:, :, idx_to_ignore] = 0
            t_euler_padded[:, :, idx_to_ignore] = 0
            s_euler_padded[:, :, idx_to_ignore] = 0
            for k in range(batch_size):
                _d_id = data_id[k].decode("utf-8")
                
                # Store results.
                _eval_result_euler[_d_id] = (
                p_euler_padded[k], t_euler_padded[k], s_euler_padded[k])
                _eval_result[_d_id] = (p[k], t[k], s[k])
                
                _action = _d_id.split('/')[-1]
                _targ = _gt_euler[_d_id][-_eval_model.target_seq_len:]  # (seq_length, 96)
                _pred = p_euler_padded[k][:_eval_model.target_seq_len]  # (seq_length, 96)
                
                # compute euler loss like Martinez does it,
                # but we don't have global translation
                gt_i = np.copy(_targ)
                gt_i[:, 0:3] = 0.0
                _pred[:, 0:3] = 0.0
                
                # compute the error only on the joints that we use for training
                idx_to_use = np.where(np.std(gt_i, 0) > 1e-4)[0]
                
                euc_error = np.power(gt_i[:, idx_to_use] - _pred[:, idx_to_use],
                                     2)
                euc_error = np.sum(euc_error, axis=1)
                euc_error = np.sqrt(euc_error)  # (seq_length, )
                if _action not in _euler_angle_metrics:
                    _euler_angle_metrics[_action] = [euc_error]
                else:
                    _euler_angle_metrics[_action].append(euc_error)
            
            n_batches += 1
            if n_batches%10 == 0:
                print("Evaluated {} samples...".format(
                    n_batches*prediction.shape[0]))
    
    except tf.errors.OutOfRangeError:
        pass
    
    print("Elapsed time: ", time.perf_counter() - _start_time)
    return _euler_angle_metrics, _eval_result_euler, _eval_result


def evaluate(session, test_model, test_data, args, eval_dir):
    _srnn_iter = test_data.get_iterator()
    _srnn_pl = test_data.get_tf_samples()

    print("Evaluating H3.6M test set (SRNN poses)...")
    undo_norm_fn = test_data.unnormalization_func
    predictions_euler, _, _ = _evaluate_srnn_poses(session,
                                                   test_model,
                                                   _srnn_iter,
                                                   _srnn_pl,
                                                   undo_norm_fn)
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

    if args.glog_entry and GLOGGER_AVAILABLE:
        exp_id = os.path.split(eval_dir)[-1].split("-")[0]
        glogger_workbook = os.environ["GLOGGER_WORKBOOK_AMASS"]
        gdrive_key = os.environ["GDRIVE_API_KEY"]
        model_name = '-'.join(os.path.split(eval_dir)[-1].split('-')[1:])
        static_values = dict()
        # exp_id = exp_id + "-W" + str(args.seq_length_in)
        static_values["Model ID"] = exp_id
        static_values["Model Name"] = model_name
    
        credentials = tf.gfile.Open(gdrive_key, "r")
        glogger = GoogleSheetLogger(
            credentials,
            glogger_workbook,
            sheet_names=["h36m"],
            model_identifier=exp_id,
            static_values=static_values)
        
        # log_data["Step"] = checkpoint_step
        glogger.update_or_append_row(log_data, "h36m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, default=None, type=str,
                        help="Experiment ID (experiment timestamp) or "
                             "comma-separated list of ids.")
    parser.add_argument('--eval_dir', required=False, default=None, type=str,
                        help="Main visualization directory. First, a folder "
                             "with the experiment name is created inside. "
                             "If not passed, then save_dir is used.")
    parser.add_argument('--save_dir', required=False, default=None, type=str,
                        help="Path to experiments. If not passed, "
                             "then AMASS_EXPERIMENTS environment variable is "
                             "used.")
    parser.add_argument('--data_dir', required=False, default=None, type=str,
                        help="Path to data. If not passed, "
                             "then AMASS_DATA environment variable is used.")

    parser.add_argument('--seq_length_in', required=False, type=int,
                        help="Seed sequence length")
    parser.add_argument('--seq_length_out', required=False, type=int,
                        help="Target sequence length")
    parser.add_argument('--batch_size', required=False, default=64, type=int,
                        help="Batch size")

    parser.add_argument('--glog_entry', required=False,
                        action="store_true",
                        help="Create a Google sheet entry if available.")

    _args = parser.parse_args()
    if ',' in _args.model_id:
        model_ids = _args.model_id.split(',')
    else:
        model_ids = [_args.model_id]

    # Set experiment directory.
    _save_dir = _args.save_dir if _args.save_dir else os.environ["AMASS_EXPERIMENTS"]
    # Set data paths.
    _data_dir = _args.data_dir if _args.data_dir else os.environ["AMASS_DATA"]

    # Run evaluation for each model id.
    for model_id in model_ids:
        try:
            _experiment_dir = glob.glob(os.path.join(_save_dir, model_id + "-*"), recursive=False)[0]
        except IndexError:
            print("Model " + str(model_id) + " is not found in " + str(_save_dir))
            continue

        try:
            tf.reset_default_graph()
            _config = json.load(open(os.path.abspath(os.path.join(_experiment_dir, 'config.json')), 'r'))
            _config["experiment_dir"] = _experiment_dir

            if _args.seq_length_out is not None and _config["target_seq_len"] != _args.seq_length_out:
                print("!!! Prediction length for training and sampling is different !!!")
                _config["target_seq_len"] = _args.seq_length_out

            if _args.seq_length_in is not None and _config["source_seq_len"] != _args.seq_length_in:
                print("!!! Seed sequence length for training and sampling is different !!!")
                _config["source_seq_len"] = _args.seq_length_in

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                exp_name = os.path.split(_experiment_dir)[-1]
                _eval_dir = _experiment_dir if _args.eval_dir is None else os.path.join(_args.eval_dir, exp_name)
                if not os.path.exists(_eval_dir):
                    os.mkdir(_eval_dir)
                _test_model, _test_data = create_and_restore_model(sess, _experiment_dir, _data_dir, _config)
                print("Evaluating Model " + str(model_id))
                evaluate(sess, _test_model, _test_data, _args, _eval_dir)
                
        except Exception as e:
            print("Something went wrong when evaluating model {}".format(model_id))
            raise Exception(e)
