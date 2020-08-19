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

import numpy as np
import tensorflow as tf

from spl.data.amass_tf import TFRecordMotionDataset
from spl.model.zero_velocity import ZeroVelocityBaseline
from spl.model.rnn import RNN
from spl.model.seq2seq import Seq2SeqModel
from spl.model.transformer import Transformer2d
from spl.model.vanilla import Transformer1d

from common.constants import Constants as C
from visualization.render import Visualizer
from visualization.fk import H36MForwardKinematics
from visualization.fk import SMPLForwardKinematics
from common.conversions import get_closest_rotmat, sparse_to_full, is_valid_rotmat

from metrics.distribution_metrics import power_spectrum
from metrics.distribution_metrics import ps_entropy
from metrics.distribution_metrics import ps_kld
from metrics.distribution_metrics import compute_npss

import seaborn as sn
import pandas as pd
from metrics.motion_metrics import MetricsEngine
import matplotlib.pyplot as plt

plt.switch_backend('agg')


AMASS_SIZE = 135
AMASS_N_JOINTS = 15
AMASS_SEED_LEN = 120


sample_keys_amass = [
    # Additional samples
    "Eyes/0/Eyes/kaiwawalk_SB2_03_SB2_sneak_SB2_kaiwa_dynamics",
    "Eyes/0/Eyes/kaiwaturn_SB2_02_SB2_look_SB_around_SB2_kaiwa_dynamics",
    "Eyes/0/Eyes/hamashowalk_SB2_06_SB2_catwalk_SB2_hamasho_dynamics",
    "Eyes/0/Eyes/ichigepose_SB2_20_SB2_zombee_SB2_ichige_dynamics",
    "Eyes/0/Eyes/kudowalk_SB2_07_SB2_moonwalk_SB2_kudo_dynamics",
    "BioMotion/0/BioMotion/rub0390000_treadmill_norm_dynamics",
    "BioMotion/0/BioMotion/rub0680000_treadmill_norm_dynamics",
    "BioMotion/0/BioMotion/rub0420028_scamper_dynamics",
    "Eyes/0/Eyes/hamashogesture_etc_SB2_20_SB2_swing_SB_chair_SB2_hamasho_dynamics",
    "HDM05/0/HDM05/bdHDM_bd_03_SB2_02_02_120_dynamics",
    "Eyes/0/Eyes/shionojump_SB2_10_SB2_rope_SB_long_SB2_shiono_dynamics",
    # Shorter than 1200 steps.
    "CMU/0/CMU/136_136_18",
    "CMU/0/CMU/143_143_23",  # punching
    "BioMotion/0/BioMotion/rub0700002_treadmill_slow_dynamics",
    "BioMotion/0/BioMotion/rub0220001_treadmill_fast_dynamics",
    # Longer than 1200
    "BioMotion/0/BioMotion/rub0640003_treadmill_jog_dynamics",
    "BioMotion/0/BioMotion/rub1110002_treadmill_slow_dynamics",
    "BioMotion/0/BioMotion/rub1030000_treadmill_norm_dynamics",
    "BioMotion/0/BioMotion/rub0800029_scamper_dynamics",
    "BioMotion/0/BioMotion/rub0830021_catching_and_throwing_dynamics",
    "Eyes/0/Eyes/kaiwajump_SB2_06_SB2_rope_SB_normal_SB_run_SB_fast_SB2_kaiwa_dynamics",
    "Eyes/0/Eyes/yokoyamathrow_toss_SB2_01_SB2_over_SB2_yokoyama_dynamics",
    
    "BioMotion/0/BioMotion/rub0830021_catching_and_throwing_dynamics",
    "CMU/0/CMU/143_143_23",
    "Eyes/0/Eyes/yokoyamathrow_toss_SB2_01_SB2_over_SB2_yokoyama_dynamics",
    "BioMotion/0/BioMotion/rub0640003_treadmill_jog_dynamics",
    "BioMotion/0/BioMotion/rub0410003_treadmill_jog_dynamics",
    "BioMotion/0/BioMotion/rub0290000_treadmill_norm_dynamics",
    "BioMotion/0/BioMotion/rub0830029_jumping2_dynamics",
    "BioMotion/0/BioMotion/rub0150028_circle_walk_dynamics",
    "BioMotion/0/BioMotion/rub0050003_treadmill_jog_dynamics",
    "Eyes/0/Eyes/hamadajump_SB2_12_SB2_boxer_SB_step_SB2_hamada_dynamics",
    "Eyes/0/Eyes/azumithrow_toss_SB2_06_SB2_both_SB_hands_SB_under_SB_light_SB2_azumi_dynamics",
    "ACCAD/0/ACCAD/Female1Running_c3dC4_SB__SB2__SB_Run_SB_to_SB_walk1_dynamics",
    "ACCAD/0/ACCAD/Male1Running_c3dRun_SB_C27_SB__SB2__SB_crouch_SB_to_SB_run_dynamics",
    "Transition/0/Transition/mazen_c3djumpingjacks_turntwist180",
    "Transition/0/Transition/mazen_c3dJOOF_runbackwards",
    
    "BioMotion/0/BioMotion/rub0290000_treadmill_norm_dynamics",
    "ACCAD/0/ACCAD/Male1Running_c3dRun_SB_C27_SB__SB2__SB_crouch_SB_to_SB_run_dynamics",
    "HDM05/0/HDM05/bdHDM_bd_03_SB2_02_02_120_dynamics",
    "BioMotion/0/BioMotion/rub0640003_treadmill_jog_dynamics",
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


def create_training_data(data_dir, config, seq_len):
    train_data_path = os.path.join(data_dir, config["data_type"], "training",
                                   "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training",
                                  "stats.npz")
    # Create dataset.
    with tf.name_scope("training_data"):
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=config["batch_size"],
                                           shuffle=False,
                                           extract_windows_of=seq_len,
                                           window_type=C.DATA_WINDOW_RANDOM,
                                           num_parallel_calls=2,
                                           normalize=False)
        return train_data


def create_and_restore_model(session, experiment_dir, data_dir, config, dynamic_test_split, mode):
    model_cls = get_model_cls(config["model_type"])

    if dynamic_test_split:  # For visualization
        data_split = "test_dynamic"
        if mode == "periodic":
            filter_sample_keys = sample_keys_amass
        else:
            filter_sample_keys = None
        beginning_index = 0
        window_type = C.DATA_WINDOW_CENTER
    else:  # For quantitative evaluation.
        data_split = "test"
        filter_sample_keys = None
        default_seed_len = 120
        if config["use_h36m"]:
            default_seed_len = 50
        beginning_index = default_seed_len - config["source_seq_len"]
        window_type = C.DATA_WINDOW_BEGINNING

    train_data_path = os.path.join(data_dir, config["data_type"], "training", "amass-?????-of-?????")
    test_data_path = os.path.join(data_dir, config["data_type"], data_split, "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training", "stats.npz")
    print("Loading test data from " + test_data_path)
    
    # Create dataset.
    # with tf.name_scope("training_data"):
    #     window_length = config["source_seq_len"] // 2
    #     train_data = TFRecordMotionDataset(data_path=train_data_path,
    #                                        meta_data_path=meta_data_path,
    #                                        batch_size=config["batch_size"],
    #                                        shuffle=False,
    #                                        extract_windows_of=window_length,
    #                                        window_type=C.DATA_WINDOW_RANDOM,
    #                                        num_parallel_calls=2,
    #                                        normalize=False)
    train_data = None
    
    with tf.name_scope("test_data"):
        window_length = config["source_seq_len"] + config["target_seq_len"]
        test_data = TFRecordMotionDataset(data_path=test_data_path,
                                          meta_data_path=meta_data_path,
                                          batch_size=8,
                                          shuffle=False,
                                          extract_windows_of=window_length,
                                          window_type=window_type,
                                          num_parallel_calls=2,
                                          normalize=not config["no_normalization"],
                                          normalization_dim=config.get("normalization_dim", "channel"),
                                          beginning_index=beginning_index,
                                          filter_by_key=filter_sample_keys,
                                          apply_length_filter=False)
        test_pl = test_data.get_tf_samples()

    # Create model.
    with tf.name_scope(C.TEST):
        test_model = model_cls(
            config=config,
            data_pl=test_pl,
            mode=C.SAMPLE,
            reuse=False)
        test_model.build_graph()
        test_model.summary_routines()

    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))

    # Restore model parameters.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)
    load_latest_checkpoint(session, saver, experiment_dir)
    return test_model, test_data, train_data


def evaluate_model(session, _eval_model, _eval_iter, _metrics_engine,
                   undo_normalization_fn, _return_results=False):
    # make a full pass on the validation or test dataset and compute the metrics
    n_batches = 0
    _eval_result = dict()
    _metrics_engine.reset()
    _attention_weights = dict()
    session.run(_eval_iter.initializer)

    using_attention_model = False
    if isinstance(_eval_model, Transformer2d):
        print("Using Attention Model.")
        using_attention_model = True
    
    try:
        while True:
            # Get the predictions and ground truth values
            res = _eval_model.sampled_step(session)
            if using_attention_model:
                prediction, targets, seed_sequence, data_id, attention = res
            else:
                prediction, targets, seed_sequence, data_id = res
            # Unnormalize predictions if there normalization applied.
            p = undo_normalization_fn(
                {"poses": prediction}, "poses")
            t = undo_normalization_fn(
                {"poses": targets}, "poses")
            s = undo_normalization_fn(
                {"poses": seed_sequence}, "poses")
            # Store each test sample and corresponding predictions with
            # the unique sample IDs.
            for k in range(prediction.shape[0]):
                _eval_result[data_id[k].decode("utf-8")] = (
                    p["poses"][k],
                    t["poses"][k],
                    s["poses"][k])
            n_batches += 1
            if n_batches % 10 == 0:
                print("Evaluated {} samples...".format(n_batches*prediction.shape[0]))
            
    except tf.errors.OutOfRangeError:
        pass
    print("Evaluated on " + str(n_batches) + " batches.")
        # finalize the computation of the metrics
    # final_metrics = _metrics_engine.get_final_metrics()
    return None, _eval_result, _attention_weights


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


def load_data_samples(session, data_dir, config, seq_len=120, n_samples=1):
    # Create dataset
    assert config["data_type"] == "rotmat"
    train_data_path = os.path.join(data_dir, config["data_type"], "training",
                                   "amass-?????-of-?????")
    meta_data_path = os.path.join(data_dir, config["data_type"], "training",
                                  "stats.npz")
    with tf.name_scope("training_data"):
        dataset = TFRecordMotionDataset(data_path=train_data_path,
                                        meta_data_path=meta_data_path,
                                        batch_size=config["batch_size"],
                                        shuffle=False,
                                        extract_windows_of=seq_len,
                                        window_type=C.DATA_WINDOW_RANDOM,
                                        num_parallel_calls=2,
                                        normalize=False)
    data_pl = dataset.get_tf_samples()
    session.run(dataset.iterator.initializer)
    
    all_samples = []
    i = 0
    while i < n_samples:
        try:
            batch = session.run(data_pl)
            np_batch = batch["inputs"]
            all_samples.append(np_batch)
            i += np_batch.shape[0]
        except tf.errors.OutOfRangeError:
            session.run(dataset.iterator.initializer)
    print("# samples: ", i)

    all_samples = np.vstack(all_samples)
    all_samples = np.reshape(all_samples, [all_samples.shape[0], all_samples.shape[1], -1, 9])
    return all_samples


def evaluate(session, test_model, test_data, args, eval_dir, mode="all_test"):
    test_iter = test_data.get_iterator()

    # Create metrics engine including summaries
    pck_thresholds = C.METRIC_PCK_THRESHS
    fk_engine = SMPLForwardKinematics()
    target_lengths = [x for x in C.METRIC_TARGET_LENGTHS_AMASS if x <= test_model.target_seq_len]
    
    representation = C.QUATERNION if test_model.use_quat else C.ANGLE_AXIS if test_model.use_aa else C.ROT_MATRIX
    metrics_engine = MetricsEngine(fk_engine,
                                   target_lengths,
                                   force_valid_rot=True,
                                   pck_threshs=pck_thresholds,
                                   rep=representation)
    # create the necessary summary placeholders and ops
    metrics_engine.create_summaries()
    # reset computation of metrics
    metrics_engine.reset()
    
    print("Evaluating test set...")
    undo_norm_fn = test_data.unnormalize_zero_mean_unit_variance_channel
    test_metrics, eval_result, attention_weights = evaluate_model(session,
                                                                  test_model,
                                                                  test_iter,
                                                                  metrics_engine,
                                                                  undo_norm_fn,
                                                                  _return_results=True)
    np.save(os.path.join(eval_dir, "eval_samples_preds_" + mode), eval_result)
    

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


def calculate_dist_metrics(eval_samples, sample_keys, train_samples=None,
                           to_pos=True, eval_seq_len=60):
    print("Computing PS KLD and PS Entropy metrics...")
    n_joints = AMASS_N_JOINTS
    results = dict()
    
    if train_samples is not None:
        n_train_samples = train_samples.shape[0]
        print("# of training samples ", n_train_samples)
        if to_pos:
            fk_engine = SMPLForwardKinematics()
            all_gt_train = to_3d_pos(np.reshape(train_samples, [n_train_samples, train_samples.shape[1], 135]), fk_engine)
            all_gt_train = np.reshape(all_gt_train, [n_train_samples, -1, all_gt_train.shape[1], all_gt_train.shape[2]])
        else:
            all_gt_train = train_samples
    
        # Create chunks of length eval_seq_len.
        all_gt_train = split_into_chunks(all_gt_train, eval_seq_len)
        all_gt_train = np.transpose(all_gt_train, (0, 2, 1, 3))
        ps_gt_train = power_spectrum(all_gt_train)
        ent_gt_train = ps_entropy(ps_gt_train)
        results["entropy_gt_train"] = ent_gt_train.mean()
    
    all_pred, all_gt_target, all_gt_seed = convert_eval_samples(eval_samples, sample_keys, to_pos, clip_to_min_len=False)
    
    # all_pred = split_into_chunks(all_pred, eval_seq_len, chunk_id=0)
    # all_gt_target = split_into_chunks(all_gt_target, eval_seq_len, chunk_id=0)
    all_gt_seed = split_into_chunks(all_gt_seed, eval_seq_len)
    # Split existing targets into chunks of eval_seq_len (i.e., 1 second) to
    # increase the number of motion samples in the test distribution.
    all_gt_target_eval_len = split_into_chunks(all_gt_target, eval_seq_len, chunk_id=0)
    all_gt_test = np.vstack([all_gt_seed, all_gt_target_eval_len])  # Using all test chunks.
    
    all_pred = np.transpose(all_pred, (0, 2, 1, 3))
    all_gt_target = np.transpose(all_gt_target, (0, 2, 1, 3))
    all_gt_test = np.transpose(all_gt_test, (0, 2, 1, 3))
    
    ps_gt_test = power_spectrum(all_gt_test)
    
    ent_gt_test = ps_entropy(ps_gt_test)
    results["entropy_gt_test"] = ent_gt_test.mean()
    
    if train_samples is not None:
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
        # Compare 1 second chunk of the predictions with 1 second real data.
        ps_pred = power_spectrum(all_pred[:, :, frame:frame + eval_seq_len])
        
        ent_pred = ps_entropy(ps_pred)
        results["entropy_prediction"].append(ent_pred.mean())
        
        kld_pred_test = ps_kld(ps_pred, ps_gt_test)
        results["kld_prediction_test"].append(kld_pred_test.mean())
        
        kld_test_pred = ps_kld(ps_gt_test, ps_pred)
        results["kld_test_prediction"].append(kld_test_pred.mean())

        if train_samples is not None:
            kld_pred_train = ps_kld(ps_pred, ps_gt_train)
            results["kld_prediction_train"].append(kld_pred_train.mean())
    
            kld_train_pred = ps_kld(ps_gt_train, ps_pred)
            results["kld_train_prediction"].append(kld_train_pred.mean())

        # If ground-truth targets are available, also make a direct comparison.
        if all_gt_target.shape[2] >= frame + eval_seq_len:
            ps_gt_target = power_spectrum(all_gt_target[:, :, frame:frame + eval_seq_len])
            
            kld_pred_target = ps_kld(ps_pred, ps_gt_target)
            results["kld_prediction_target"].append(kld_pred_target.mean())
            
            kld_target_pred = ps_kld(ps_gt_target, ps_pred)
            results["kld_target_prediction"].append(kld_target_pred.mean())
            
            kld_test_target = ps_kld(ps_gt_test, ps_gt_target)
            results["kld_test_target"].append(kld_test_target.mean())
            
            kld_target_test = ps_kld(ps_gt_target, ps_gt_test)
            results["kld_target_test"].append(kld_target_test.mean())
    return results


def convert_eval_samples(eval_samples, sample_keys, to_pos, clip_to_min_len=True):
    """
    
    Args:
        eval_samples: dict of (prediction, target, seed)
        sample_keys:
        to_pos:

    Returns: (batch_size, seq_len, n_joints, feature_size)

    """
    all_predictions = []
    all_targets = []
    all_seeds = []
    for key_ in sample_keys:
        pred, target, seed = eval_samples[key_]
        
        all_predictions.append(np.expand_dims(pred, 0))
        all_targets.append(np.expand_dims(target, 0))
        all_seeds.append(np.expand_dims(seed, 0))

    # Check if there are shorter targets.
    target_lens = np.array([tt.shape[1] for tt in all_targets])
    max_len = (target_lens.min()//60)*60

    if clip_to_min_len:
        all_predictions = np.vstack(all_predictions)[:, :max_len]
    else:
        all_predictions = np.vstack(all_predictions)
      
    all_targets = [sample[:, :max_len] for sample in all_targets]
    all_targets = np.vstack(all_targets)
    all_seeds = np.vstack(all_seeds)

    assert all_predictions.shape[-1] == AMASS_SIZE
    assert all_targets.shape[-1] == AMASS_SIZE
    assert all_seeds.shape[-1] == AMASS_SIZE
    assert all_targets.shape[0] == all_predictions.shape[0]
    assert all_targets.shape[-1] == all_predictions.shape[-1]
    joint_size = 9
    batch_size = all_predictions.shape[0]
    seed_len = all_seeds.shape[1]
    pred_len = all_predictions.shape[1]
    
    if to_pos:
        fk_engine = SMPLForwardKinematics()
        all_predictions = to_3d_pos(all_predictions, fk_engine, dof=9, force_valid_rot=True, is_sparse=True)
        all_targets = to_3d_pos(all_targets, fk_engine, dof=9, force_valid_rot=True, is_sparse=True)
        all_seeds = to_3d_pos(all_seeds, fk_engine, dof=9, force_valid_rot=True, is_sparse=True)
        joint_size = 3
        # animate_matplotlib([pos_[0][:240], pos_tar[0][:240]], colors=[_colors[0], _colors[1]], titles=["pred", "tar"], fig_title="", parents=fk_engine.parents, out_dir="./", to_video=True, keep_frames=True, fname="amass")

    all_predictions = np.reshape(all_predictions, [batch_size, pred_len, -1, joint_size])
    all_targets = np.reshape(all_targets, [batch_size, max_len, -1, joint_size])
    all_seeds = np.reshape(all_seeds, [batch_size, seed_len, -1, joint_size])
      
    return all_predictions, all_targets, all_seeds
      

def calculate_npss_metrics(eval_samples, sample_keys, to_pos=True):
    print("Computing NPSS metric...")
    results = dict()
    all_predictions, all_targets, _ = convert_eval_samples(eval_samples, sample_keys, to_pos)

    if to_pos:
        all_predictions = np.reshape(all_predictions, [all_predictions.shape[0], all_predictions.shape[1], 72])
        all_targets = np.reshape(all_targets, [all_predictions.shape[0], all_predictions.shape[1], 72])
    else:
        all_predictions = np.reshape(all_predictions, [all_predictions.shape[0], all_predictions.shape[1], AMASS_SIZE])
        all_targets = np.reshape(all_targets, [all_predictions.shape[0], all_predictions.shape[1], AMASS_SIZE])
    
    fps = 60    
    time_indices = [400, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for ms in time_indices:
        sec = (ms / 1000.0)
        idx = int(sec * fps)
        if all_targets.shape[1] >= idx:
            results["npss_" + str(int(sec))] = compute_npss(all_targets[:, :idx], all_predictions[:, :idx])
          
    return results
      

def log_metrics(args, results, exp_id, model_name, sheet_name=None):
    sheet_name = sheet_name or "dist_metrics_amass"
    glog_entry = dict()
    
    # Log dist-based metrics.
    if "entropy_gt_train" in results:
        print("GT Train Entropy: ", results["entropy_gt_train"])
        glog_entry["entropy_gt_train"] = results["entropy_gt_train"]
    
        print("GT Train -> GT Test KLD: ", results["kld_train_test"])
        glog_entry["kld_train_test"] = results["kld_train_test"]
        print("GT Test -> GT Train KLD: ", results["kld_test_train"])
        glog_entry["kld_test_train"] = results["kld_test_train"]
        glog_entry["kld_avg_train_test"] = (results["kld_test_train"] + results["kld_train_test"])/2
    
    if "entropy_prediction" in results:
        print("GT Test Entropy: ", results["entropy_gt_test"])
        glog_entry["entropy_gt_test"] = results["entropy_gt_test"]
        
        n_entries = len(results["entropy_prediction"])
        for sec in range(n_entries):
            i = str(sec + 1)
            print("[{}] Prediction Entropy: {}".format(sec + 1, results["entropy_prediction"][sec]))
            glog_entry[i + "_entropy_pred"] = results["entropy_prediction"][sec]
        
            print("[{}] KLD Prediction -> GT Test: {}".format(sec + 1, results["kld_prediction_test"][sec]))
            print("[{}] KLD GT Test -> Prediction: {}".format(sec + 1, results["kld_test_prediction"][sec]))
            glog_entry[i + "_avg_kld_pred_test"] = (results["kld_prediction_test"][sec] + results["kld_test_prediction"][sec])/2
            
            if len(results["kld_prediction_target"]) > sec:
                print("[{}] KLD Prediction -> GT Target: {}".format(sec + 1, results["kld_prediction_target"][sec]))
                print("[{}] KLD GT Target -> Prediction: {}".format(sec + 1, results["kld_target_prediction"][sec]))
                glog_entry[i + "_avg_kld_pred_target"] = (results["kld_prediction_target"][sec] + results["kld_target_prediction"][sec])/2
                
                print("[{}] KLD GT Test -> GT Target: {}".format(sec + 1, results["kld_test_target"][sec]))
                print("[{}] KLD GT Target -> GT Test: {}".format(sec + 1, results["kld_target_test"][sec]))
                glog_entry[i + "_avg_kld_target_test"] = (results["kld_test_target"][sec] + results["kld_target_test"][sec])/2

            # Check if training results available.
            if "entropy_gt_train" in results:
                print("[{}] KLD Prediction -> GT Train: {}".format(sec + 1, results["kld_prediction_train"][sec]))
                print("[{}] KLD GT Train -> Prediction: {}".format(sec + 1, results["kld_train_prediction"][sec]))
                glog_entry[i + "_avg_kld_pred_train"] = (results["kld_prediction_train"][sec] + results["kld_train_prediction"][sec]) / 2
            print()
    
    # Log npss results.
    for key_ in sorted(results.keys()):
        if key_.startswith("npss_"):
            glog_entry[key_] = results[key_]
            print(key_, ": ", results[key_])
        
    if args.glog_entry and GLOGGER_AVAILABLE:
        glogger_workbook = os.environ["GLOGGER_WORKBOOK_AMASS"]
        gdrive_key = os.environ["GDRIVE_API_KEY"]
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

                eval_seq_len = 60
                mode = "periodic"  # "periodic" or "all_test"
                saved_metrics_p = os.path.join(_eval_dir, "dist_metrics_{}.npy".format(mode))
                saved_predictions = os.path.join(_eval_dir, "eval_samples_preds_{}.npy".format(mode))
                saved_training = os.path.join(_eval_dir, "amass_train_rotmat_{}.npy".format(eval_seq_len))

                exp_id = os.path.split(_eval_dir)[-1].split("-")[0] + "-" + mode
                model_name = '-'.join(os.path.split(_eval_dir)[-1].split('-')[1:])
                
                if not os.path.exists(saved_predictions):  # and not os.path.exists(saved_training):
                    if not os.path.exists(_eval_dir):
                        os.mkdir(_eval_dir)
                    _test_model, _test_data, _train_data = create_and_restore_model(sess, _experiment_dir, _data_dir, _config, _args.dynamic_test_split, mode)
                    print("Evaluating Model " + str(model_id))
                    evaluate(sess, _test_model, _test_data, _args, _eval_dir, mode)

                if not os.path.exists(saved_training):  # Load training data for dist. metrics.
                    training_samples = load_data_samples(sess, _data_dir, _config, n_samples=20000, seq_len=eval_seq_len)
                    np.save(os.path.join(_eval_dir, "amass_train_rotmat_" + str(eval_seq_len)), training_samples)

                if os.path.exists(saved_predictions):  # NPSS.
                    _eval_samples = np.load(saved_predictions).tolist()

                    if mode == "periodic":
                        _sample_keys = list(sample_keys_amass)
                    elif mode == "all_test":
                        _sample_keys = list(_eval_samples.keys())
                    else:
                        raise Exception("Unknown mode.")

                    npss_results = calculate_npss_metrics(_eval_samples, _sample_keys)
                    log_metrics(_args, npss_results, exp_id, model_name)

                if os.path.exists(saved_predictions):  # ps kld

                    _eval_samples = np.load(saved_predictions).tolist()

                    _train_samples = None
                    if os.path.exists(saved_training):
                        _train_samples = np.load(saved_training)

                    if mode == "periodic":
                        _sample_keys = list(sample_keys_amass)
                    elif mode == "all_test":
                        _sample_keys = list(_eval_samples.keys())
                    else:
                        raise Exception("Unknown mode.")

                    dist_results = calculate_dist_metrics(_eval_samples, _sample_keys, train_samples=_train_samples, to_pos=True, eval_seq_len=eval_seq_len)
                    np.save(os.path.join(_eval_dir, "dist_metrics_" + mode), dist_results)
                    log_metrics(_args, dist_results, exp_id, model_name)
                
                # if os.path.exists(saved_metrics_p):
                #     dist_metrics = np.load(saved_metrics_p).tolist()
                #     log_metrics(_args, _eval_dir, dist_metrics, mode)
                
                
        except Exception as e:
            print("Something went wrong when evaluating model {}".format(model_id))
            raise Exception(e)
