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
import argparse
import time
import quaternion

import numpy as np
import tensorflow as tf

from spl.data.srnn_tf import SRNNTFRecordMotionDataset
from spl.data.amass_tf import TFRecordMotionDataset
from spl.model.zero_velocity import ZeroVelocityBaseline
from spl.model.rnn import RNN
from spl.model.seq2seq import Seq2SeqModel
from spl.model.transformer import Transformer2d
from spl.model.vanilla import Transformer1d

import visualization.fk as fk
from common.constants import Constants as C
from visualization.fk import H36MForwardKinematics
from common.conversions import get_closest_rotmat, sparse_to_full, is_valid_rotmat
from common.conversions import rotmat2euler, aa2rotmat

from visualization.render import animate_matplotlib

from metrics.distribution_metrics import power_spectrum
from metrics.distribution_metrics import ps_entropy
from metrics.distribution_metrics import ps_kld

import matplotlib.pyplot as plt
plt.switch_backend('agg')
_prop_cycle = plt.rcParams['axes.prop_cycle']
_colors = _prop_cycle.by_key()['color']

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


def get_model_cls(model_type):
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


def create_and_restore_model(session, experiment_dir, data_dir, config, mode):
    model_cls = get_model_cls(config["model_type"])

    srnn_dir = "srnn_poses_25fps"
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    srnn_path = os.path.join(data_dir, config["data_type"], srnn_dir, "amass-?????-of-?????")
    
    with tf.name_scope("srnn_data"):
        srnn_data = SRNNTFRecordMotionDataset(data_path=srnn_path,
                                              meta_data_path=meta_data_path,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              seed_len=config["source_seq_len"],
                                              target_len=config["target_seq_len"],
                                              num_parallel_calls=2,
                                              normalize=not config["no_normalization"])
        
        srnn_pl = srnn_data.get_tf_samples()
    print("Loading test data from " + srnn_path)
    
    # Create dataset.
    train_data_path = os.path.join(data_dir, config["data_type"], "training", "amass-?????-of-?????")
    with tf.name_scope("training_data"):
        window_length = config["source_seq_len"] // 2
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"],
                                           shuffle=False,
                                           extract_windows_of=window_length,
                                           window_type=C.DATA_WINDOW_RANDOM,
                                           num_parallel_calls=2,
                                           normalize=False)
    # Create model.
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
    return srnn_model, srnn_data, train_data


def rotmat_to_euler_padded(samples):
    batch_size, seq_length, _ = samples.shape
    p_euler = rotmat2euler(np.reshape(samples, [batch_size, seq_length, -1, 3, 3]))
    p_euler_padded = np.zeros([batch_size, seq_length, 32, 3])
    p_euler_padded[:, :, fk.H36M_MAJOR_JOINTS] = p_euler
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
                _gt_euler[srnn_batch[C.BATCH_ID][k].decode("utf-8")] = euler_targ
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
            res = _eval_model.sampled_step(session, prediction_steps=_args.seq_length_out)
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
                rot = quaternion.as_rotation_matrix(quaternion.from_float_array(
                    np.reshape(p, [batch_size, seq_length, -1, 4])))
                p_euler = rotmat2euler(rot)
            elif _eval_model.use_aa:
                p_euler = rotmat2euler(
                    aa2rotmat(np.reshape(p, [batch_size, seq_length, -1, 3])))
            else:
                # p_euler = rotmat2euler(np.reshape(p, [batch_size, seq_length, -1, 3, 3]))
                # t_euler = rotmat2euler(np.reshape(t, [batch_size, t.shape[1], -1, 3, 3]))
                # s_euler = rotmat2euler(np.reshape(s, [batch_size, s.shape[1], -1, 3, 3]))
                p_euler_padded = rotmat_to_euler_padded(p)
                t_euler_padded = rotmat_to_euler_padded(t)
                s_euler_padded = rotmat_to_euler_padded(s)

            # p_euler_padded = np.zeros([batch_size, seq_length, 32, 3])
            # p_euler_padded[:, :, fk.H36M_MAJOR_JOINTS] = p_euler
            # p_euler_padded = np.reshape(p_euler_padded, [batch_size, seq_length, -1])
            #
            # t_euler_padded = np.zeros([batch_size, t.shape[1], 32, 3])
            # t_euler_padded[:, :, fk.H36M_MAJOR_JOINTS] = t_euler
            # t_euler_padded = np.reshape(t_euler_padded, [batch_size, t.shape[1], -1])
            #
            # s_euler_padded = np.zeros([batch_size, s.shape[1], 32, 3])
            # s_euler_padded[:, :, fk.H36M_MAJOR_JOINTS] = s_euler
            # s_euler_padded = np.reshape(s_euler_padded, [batch_size, s.shape[1], -1])

            idx_to_use = np.where(np.reshape(t_euler_padded, [-1, 96]).std(0) > 1e-4)[0]
            idx_to_ignore = np.where(np.reshape(t_euler_padded, [-1, 96]).std(0) < 1e-4)[0]

            p_euler_padded[:, :, idx_to_ignore] = 0
            t_euler_padded[:, :, idx_to_ignore] = 0
            s_euler_padded[:, :, idx_to_ignore] = 0
            for k in range(batch_size):
                _d_id = data_id[k].decode("utf-8")

                # Store results.
                _eval_result_euler[_d_id] = (p_euler_padded[k], t_euler_padded[k], s_euler_padded[k])
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

                euc_error = np.power(gt_i[:, idx_to_use] - _pred[:, idx_to_use], 2)
                euc_error = np.sum(euc_error, axis=1)
                euc_error = np.sqrt(euc_error)  # (seq_length, )
                if _action not in _euler_angle_metrics:
                    _euler_angle_metrics[_action] = [euc_error]
                else:
                    _euler_angle_metrics[_action].append(euc_error)

            n_batches += 1
            if n_batches % 10 == 0:
                print("Evaluated {} samples...".format(
                    n_batches*prediction.shape[0]))
            
    except tf.errors.OutOfRangeError:
        pass
    
    print("Elapsed time: ", time.perf_counter() - _start_time)
    return _euler_angle_metrics, _eval_result_euler, _eval_result


def to_3d_pos(angles, fk_engine, dof=9, force_valid_rot=True, is_sparse=True):
    n_joints = angles.shape[2] // dof
    assert n_joints*dof == angles.shape[-1], "unexpected number of joints"

    # enforce valid rotations
    if force_valid_rot:
        angle_val = np.reshape(angles, [-1, n_joints, 3, 3])
        angles = get_closest_rotmat(angle_val)
        angles = np.reshape(angles, [-1, n_joints*dof])

    # check that the rotations are valid
    are_valid = is_valid_rotmat(np.reshape(angles, [-1, n_joints, 3, 3]))
    assert are_valid, 'Rotation matrices are not valid'

    # add potentially missing joints
    if is_sparse:
        angles = sparse_to_full(angles, fk_engine.major_joints,
                                fk_engine.n_joints, rep="rotmat")

    # make sure we don't consider the root orientation
    assert angles.shape[-1] == fk_engine.n_joints*dof
    angles[:, 0:9] = np.eye(3, 3).flatten()
    pos = fk_engine.from_rotmat(angles)  # (-1, full_n_joints, 3)
    pos = pos[..., [0, 2, 1]]
    return pos


def load_data_samples(session, dataset, n_samples=1):
    all_samples = []
    data_pl = dataset.get_tf_samples()
    session.run(dataset.iterator.initializer)
    i = 0
    while i < n_samples:
        try:
            batch = session.run(data_pl)
            np_batch = batch["inputs"]
            all_samples.append(np_batch)
            i += np_batch.shape[0]
        except tf.errors.OutOfRangeError:
            session.run(dataset.iterator.initializer)
    return np.vstack(all_samples)


def log_euler_loss(euler_losses):
    log_data = dict()
    which_actions = ['walking', 'eating', 'discussion', 'smoking']
    
    print("{:<10}".format(""), end="")
    for ms in [80, 160, 320, 400]:
        print("  {0:4d}  ".format(ms), end="")
    
    if euler_losses[which_actions[0]][0].shape[0] > 12:
        for ms in [560, 1000]:
            print("  {0:4d}  ".format(ms), end="")
    print()
    
    test_str = " {:.3f} \t{:.3f} \t{:.3f} \t{:.3f}"
    long_test_str = " \t{:.3f} \t{:.3f}"
    
    for action in which_actions:
        # get the mean over all samples for that action
        assert len(euler_losses[action]) == 8
        euler_mean = np.mean(np.stack(euler_losses[action]), axis=0)
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


def evaluate(session, test_model, test_data, args, eval_dir, train_data=None, mode="periodic"):
    _srnn_iter = test_data.get_iterator()
    _srnn_pl = test_data.get_tf_samples()

    # Create metrics engine including summaries
    fk_engine = H36MForwardKinematics()

    # Load training data.
    train_samples = load_data_samples(session, train_data, n_samples=20000)
    train_samples = train_samples.astype(np.float32)
    np.save(os.path.join(eval_dir, "h36m_train_rotmat"), train_samples)
    train_samples_euler = rotmat_to_euler_padded(train_samples)
    train_samples_euler= train_samples_euler.astype(np.float32)
    np.save(os.path.join(eval_dir, "h36m_train_euler"), train_samples_euler)
    
    print("Evaluating test set...")
    undo_norm_fn = test_data.unnormalize_zero_mean_unit_variance_channel
    euler_loss, eval_result_euler, eval_result = _evaluate_srnn_poses(session,
                                                                      test_model,
                                                                      _srnn_iter,
                                                                      _srnn_pl,
                                                                      undo_norm_fn)

    log_euler_loss(euler_loss)
    np.save(os.path.join(eval_dir, "srnn_test_preds_euler"), eval_result_euler)
    np.save(os.path.join(eval_dir, "srnn_test_preds_rotmat"), eval_result)

    calculate_dist_metrics(eval_dir, train_samples, eval_result, rep="rotmat", fk_engine=fk_engine)
    

def split_into_chunks(tensor, split_len, chunk_id=None):
    """
    Args:
        tensor: (batch_size, seq_len, ...)
        split_len: chunk len
        chunk_id: which chunk to return (0-based). None means all.
    Returns:
    """
    seq_len_ = tensor.shape[1]
    if split_len >= seq_len_:
        return tensor
    
    chunks = []
    for i in range(0, seq_len_ - split_len + 1, split_len):
        chunks.append(tensor[:, i:i + split_len])
    
    if chunk_id is not None:
        return chunks[chunk_id]
    else:
        return np.vstack(chunks)
    

def calculate_dist_metrics(eval_dir, train_samples, eval_samples, eval_seq_len=25, rep="rotmat", fk_engine=None, actions=None):
    if rep == "rotmat" and fk_engine is None:
        raise Exception("fk_engine is required for 3d pos conversion.")
    
    n_joints = 32
    gt_train = []
    n_train_samples = train_samples.shape[0]
    print("# of training samples ", n_train_samples)
    
    if rep is "rotmat":
        for i in range(n_train_samples):
            pos = to_3d_pos(train_samples[i:i + 1], fk_engine)
            gt_train.append(np.expand_dims(pos, axis=0))
        all_gt_train = np.concatenate(gt_train, axis=0)
        
        train_pos_path = os.path.join(eval_dir, "h36m_train_pos.npy")
        if not os.path.exists(train_pos_path):
            np.save(os.path.join(eval_dir, "h36m_train_pos"), np.reshape(all_gt_train, [n_train_samples, -1, n_joints*3]))
    else:
        all_gt_train = np.reshape(train_samples, [n_train_samples, -1, n_joints, 3])
    
    # Create chunks of length eval_seq_len.
    all_gt_train = split_into_chunks(all_gt_train, eval_seq_len)
    
    # # Sanity check
    # animate_matplotlib([all_gt_train[50]],
    #                    colors=[_colors[0]], titles=[""], fig_title="",
    #                    parents=fk_engine.parents, out_dir=eval_dir,
    #                    to_video=True,
    #                    keep_frames=True, fname="train_h36m")
    predictions = []
    targets = []
    gt_seeds = []
    pos_dict = dict()
    if rep is "rotmat":
        print("Converting rotations into positions...")
        for key_, sample in eval_samples.items():
            pred, target, seed = sample
            predictions.append(np.expand_dims(to_3d_pos(np.expand_dims(pred, axis=0), fk_engine, dof=9), axis=0))
            targets.append(np.expand_dims(to_3d_pos(np.expand_dims(target, axis=0), fk_engine, dof=9), axis=0))
            gt_seeds.append(np.expand_dims(to_3d_pos(np.expand_dims(seed[0:25], axis=0), fk_engine, dof=9), axis=0))
            gt_seeds.append(np.expand_dims(to_3d_pos(np.expand_dims(seed[25:50], axis=0), fk_engine, dof=9), axis=0))

            pos_dict[key_] = (np.reshape(predictions[-1], [-1, n_joints*3]), np.reshape(targets[-1], [-1, n_joints*3]), np.reshape(gt_seeds[-1], [-1, n_joints*3]))

        eval_pos_path = os.path.join(eval_dir, "srnn_test_preds_pos.npy")
        if not os.path.exists(eval_pos_path):
            np.save(os.path.join(eval_dir, "srnn_test_preds_pos"), pos_dict)
    else:
        for key_, sample in eval_samples.items():
            pred, target, seed = sample
            if actions is None or key_.split("/")[1] in actions:
                predictions.append(np.reshape(pred, [1, -1, n_joints, 3]))
                targets.append(np.reshape(target, [1, -1, n_joints, 3]))
                gt_seeds.append(np.reshape(seed, [1, -1, n_joints, 3]))

    all_pred = np.concatenate(predictions, axis=0)
    all_pred = split_into_chunks(all_pred, eval_seq_len, chunk_id=0)

    all_gt_target = np.concatenate(targets, axis=0)
    all_gt_target = split_into_chunks(all_gt_target, eval_seq_len, chunk_id=0)

    all_gt_seed = np.concatenate(gt_seeds, axis=0)
    all_gt_seed = split_into_chunks(all_gt_seed, eval_seq_len)
    all_gt_test = np.vstack([all_gt_seed, all_gt_target])  # Using all test chunks.
    
    # Sanity check.
    print("Train shape: ", str(all_gt_train.shape))
    print("Test shape: ", str(all_gt_test.shape))
    print("Prediction shape: ", str(all_pred.shape))
    print("Train 0 entries: ", np.where(np.reshape(all_gt_train, [-1, 96]).std(0) == 0)[0].shape[0])
    print("Seed 0 entries: ", np.where(np.reshape(all_gt_seed, [-1, 96]).std(0) == 0)[0].shape[0])
    print("Prediction 0 entries: ", np.where(np.reshape(all_pred, [-1, 96]).std(0) == 0)[0].shape[0])
    print("Target 0 entries: ", np.where(np.reshape(all_gt_target, [-1, 96]).std(0) == 0)[0].shape[0])

    all_gt_train = np.transpose(all_gt_train, (0, 2, 1, 3))
    all_pred = np.transpose(all_pred, (0, 2, 1, 3))
    all_gt_target = np.transpose(all_gt_target, (0, 2, 1, 3))
    all_gt_test = np.transpose(all_gt_test, (0, 2, 1, 3))
    
    ps_gt_test = power_spectrum(all_gt_test)
    ps_gt_train = power_spectrum(all_gt_train)
    ps_gt_target = power_spectrum(all_gt_target)
    
    results = dict()
    ent_gt_train = ps_entropy(ps_gt_train)
    results["entropy_gt_train"] = ent_gt_train.mean()
    
    ent_gt_test = ps_entropy(ps_gt_test)
    results["entropy_gt_test"] = ent_gt_test.mean()
    
    kld_train_test = ps_kld(ps_gt_train, ps_gt_test)
    kld_test_train = ps_kld(ps_gt_test, ps_gt_train)
    results["kld_train_test"] = kld_train_test.mean()
    results["kld_test_train"] = kld_test_train.mean()
    
    results["entropy_prediction"] = list()
    results["kld_train_prediction"] = list()
    results["kld_prediction_train"] = list()
    results["kld_test_prediction"] = list()
    results["kld_prediction_test"] = list()
    results["kld_prediction_target"] = list()
    results["kld_target_prediction"] = list()
    results["kld_test_target"] = list()
    results["kld_target_test"] = list()
    
    pred_len = all_pred.shape[2]
    for sec, frame in enumerate(range(0, pred_len - eval_seq_len + 1, eval_seq_len)):
        ps_pred = power_spectrum(all_pred[:, :, frame:frame + eval_seq_len])
        
        ent_pred = ps_entropy(ps_pred)
        results["entropy_prediction"].append(ent_pred.mean())
        
        kld_pred_train = ps_kld(ps_pred, ps_gt_train)
        results["kld_prediction_train"].append(kld_pred_train.mean())
        
        kld_train_pred = ps_kld(ps_gt_train, ps_pred)
        results["kld_train_prediction"].append(kld_train_pred.mean())
        
        kld_pred_test = ps_kld(ps_pred, ps_gt_test)
        results["kld_prediction_test"].append(kld_pred_test.mean())
        
        kld_test_pred = ps_kld(ps_gt_test, ps_pred)
        results["kld_test_prediction"].append(kld_test_pred.mean())

        kld_pred_target = ps_kld(ps_pred, ps_gt_target)
        results["kld_prediction_target"].append(kld_pred_target.mean())

        kld_target_pred = ps_kld(ps_gt_target, ps_pred)
        results["kld_target_prediction"].append(kld_target_pred.mean())

        kld_test_target = ps_kld(ps_gt_test, ps_gt_target)
        results["kld_test_target"].append(kld_test_target.mean())
    
        kld_target_test = ps_kld(ps_gt_target, ps_gt_test)
        results["kld_target_test"].append(kld_target_test.mean())
    
    # np.save(os.path.join(eval_dir, "dist_metrics_" + mode), results)
    log_metrics(_args, _eval_dir, results, mode)
    

def log_metrics(args, eval_dir, results, mode, sheet_name=None):
    sheet_name = sheet_name or "dist_metrics_h36m"
    glog_entry = dict()
    
    print("GT Train Entropy: ", results["entropy_gt_train"])
    glog_entry["entropy_gt_train"] = results["entropy_gt_train"]

    print("GT Test Entropy: ", results["entropy_gt_test"])
    glog_entry["entropy_gt_test"] = results["entropy_gt_test"]

    print("GT Train -> GT Test KLD: ", results["kld_train_test"])
    glog_entry["kld_train_test"] = results["kld_train_test"]
    print("GT Test -> GT Train KLD: ", results["kld_test_train"])
    glog_entry["kld_test_train"] = results["kld_test_train"]
    glog_entry["kld_avg_train_test"] = (results["kld_test_train"] + results["kld_train_test"])/2
    
    n_entries = len(results["entropy_prediction"])
    for sec in range(n_entries):
        i = str(sec + 1)
        print("[{}] Prediction Entropy: {}".format(sec + 1, results["entropy_prediction"][sec]))
        glog_entry[i + "_entropy_pred"] = results["entropy_prediction"][sec]
    
        print("[{}] KLD Prediction -> GT Train: {}".format(sec + 1, results["kld_prediction_train"][sec]))
        print("[{}] KLD GT Train -> Prediction: {}".format(sec + 1, results["kld_train_prediction"][sec]))
        glog_entry[i + "_avg_kld_pred_train"] = (results["kld_prediction_train"][sec] + results["kld_train_prediction"][sec]) / 2
    
        print("[{}] KLD Prediction -> GT Test: {}".format(sec + 1, results["kld_prediction_test"][sec]))
        print("[{}] KLD GT Test -> Prediction: {}".format(sec + 1, results["kld_test_prediction"][sec]))
        glog_entry[i + "_avg_kld_pred_test"] = (results["kld_prediction_test"][sec] + results["kld_test_prediction"][sec])/2
        
        print("[{}] KLD Prediction -> GT Target: {}".format(sec + 1, results["kld_prediction_target"][sec]))
        print("[{}] KLD GT Target -> Prediction: {}".format(sec + 1, results["kld_target_prediction"][sec]))
        glog_entry[i + "_avg_kld_pred_target"] = (results["kld_prediction_target"][sec] + results["kld_target_prediction"][sec])/2
        
        print("[{}] KLD GT Test -> GT Target: {}".format(sec + 1, results["kld_test_target"][sec]))
        print("[{}] KLD GT Target -> GT Test: {}".format(sec + 1, results["kld_target_test"][sec]))
        glog_entry[i + "_avg_kld_target_test"] = (results["kld_test_target"][sec] + results["kld_target_test"][sec])/2
        
        print()
    
    if args.glog_entry and GLOGGER_AVAILABLE:
        exp_id = os.path.split(eval_dir)[-1].split("-")[0] + "-" + mode
        glogger_workbook = os.environ["GLOGGER_WORKBOOK_AMASS"]
        gdrive_key = os.environ["GDRIVE_API_KEY"]
        model_name = '-'.join(os.path.split(eval_dir)[-1].split('-')[1:])
        static_values = dict()
        static_values["Model ID"] = exp_id
        static_values["Model Name"] = model_name

        credentials = tf.gfile.Open(gdrive_key, "r")
        glogger = GoogleSheetLogger(
            credentials,
            glogger_workbook,
            sheet_names=[sheet_name],
            model_identifier=exp_id,
            static_values=static_values)

        glogger.update_or_append_row(glog_entry, sheet_name)


if __name__ == '__main__':
    # If you would like to quantitatively evaluate a model, then
    # --dynamic_test_split shouldn't be passed. In this case, the model will be
    # evaluated on 180 frame windows extracted from the entire test split.
    # You can still visualize samples. However, number of predicted frames
    # will be less than or equal to 60. If you intend to evaluate/visualize
    # longer predictions, then you should pass --dynamic_test_split which
    # enables using original full-length test sequences. Hence,
    # --seq_length_out can be much longer than 60.

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

    parser.add_argument('--visualize', required=False, action="store_true",
                        help="Visualize ground-truth and predictions "
                             "side-by-side by using human skeleton.")
    parser.add_argument('--no_skel', required=False, action="store_true",
                        help="Dont show skeleton in offline visualization.")
    parser.add_argument('--no_mesh', required=False, action="store_true",
                        help="Dont show mesh in offline visualization")
    parser.add_argument('--to_video', required=False, action="store_true",
                        help="Save the model predictions to mp4 videos in the "
                             "experiments folder.")
    parser.add_argument('--dynamic_test_split', required=False,
                        action="store_true",
                        help="Test samples are extracted on-the-fly.")
    parser.add_argument('--glog_entry', required=False,
                        action="store_true",
                        help="Create a Google sheet entry if available.")
    parser.add_argument('--new_experiment_id', required=False, default=None,
                        type=str, help="Not used. only for leonhard.")

    _args = parser.parse_args()
    if ',' in _args.model_id:
        model_ids = _args.model_id.split(',')
    else:
        model_ids = [_args.model_id]

    # Set experiment directory.
    _save_dir = _args.save_dir if _args.save_dir else os.environ["AMASS_EXPERIMENTS"]
    # Set data paths.
    _data_dir = _args.data_dir if _args.data_dir else os.environ["AMASS_DATA"]
    _data_dir = os.path.join(_data_dir, '../h3.6m/tfrecords/')

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
                # _config["target_seq_len"] = _args.seq_length_out

            if _args.seq_length_in is not None and _config["source_seq_len"] != _args.seq_length_in:
                print("!!! Seed sequence length for training and sampling is different !!!")
                # _config["source_seq_len"] = _args.seq_length_in

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                exp_name = os.path.split(_experiment_dir)[-1]
                _eval_dir = _experiment_dir if _args.eval_dir is None else os.path.join(_args.eval_dir, exp_name)

                eval_len = 25
                which_actions = ['walking', 'eating', 'discussion', 'smoking']
                # which_actions = None
                mode = "euler_ours_20k-{}_frames{}".format(eval_len, "-actions" if which_actions is not None else "")
                saved_metrics_p = os.path.join(_eval_dir, "dist_metrics_{}.npy".format(mode))
                saved_samples = os.path.join(_eval_dir, "h36m_train_euler.npy")
                # saved_samples = os.path.join(_eval_dir, "dct_h36m_train_euler.npy")
                if os.path.exists(saved_metrics_p):
                    dist_metrics = np.load(saved_metrics_p).tolist()
                    log_metrics(_args, _eval_dir, dist_metrics, mode)
                elif os.path.exists(saved_samples):
                    _train_samples = np.load(saved_samples)
                    _eval_samples = np.load(os.path.join(_eval_dir, "srnn_test_preds_euler.npy")).tolist()
                    fk_engine = H36MForwardKinematics()
                    calculate_dist_metrics(_eval_dir, _train_samples, _eval_samples, rep="pos", fk_engine=fk_engine, eval_seq_len=eval_len, actions=which_actions)
                else:
                    if not os.path.exists(_eval_dir):
                        os.mkdir(_eval_dir)
                    _test_model, _test_data, _train_data = create_and_restore_model(sess, _experiment_dir, _data_dir, _config, mode)
                    print("Evaluating Model " + str(model_id))
                    evaluate(sess, _test_model, _test_data, _args, _eval_dir, _train_data, mode)
                
        except Exception as e:
            print("Something went wrong when evaluating model {}".format(model_id))
            raise Exception(e)