"""
Process the H36M data that Martinez uses.
"""

import numpy as np
import os
import tensorflow as tf
import cv2

from data_utils import readCSVasFloat
from amass_prepare import create_tfrecord_writers
from amass_prepare import write_tfexample
from amass_prepare import split_into_windows
from amass_prepare import close_tfrecord_writers


RNG = np.random.RandomState(42)


def to_tfexample(poses, file_id, db_name, one_hot):
    features = dict()
    features['file_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode('utf-8')]))
    features['db_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db_name.encode('utf-8')]))
    features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=poses.shape))
    features['poses'] = tf.train.Feature(float_list=tf.train.FloatList(value=poses.flatten()))
    features['one_hot'] = tf.train.Feature(float_list=tf.train.FloatList(value=one_hot))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def process_split(poses, one_hots, file_ids, output_path, n_shards, compute_stats, create_windows=None):
    print("storing into {} computing stats {}".format(output_path, "YES" if compute_stats else "NO"))

    if compute_stats:
        assert create_windows is None, "computing the statistics should only be done when not extracting windows"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, 'amass'), n_shards)
    if create_windows is not None:
        if not os.path.exists(output_path + "_dynamic"):
            os.makedirs(output_path + "_dynamic")
        tfrecord_writers_dyn = create_tfrecord_writers(os.path.join(output_path + "_dynamic", "amass"), n_shards)

    # compute normalization stats online
    n_all, mean_all, var_all, m2_all = 0.0, 0.0, 0.0, 0.0
    n_channel, mean_channel, var_channel, m2_channel = 0.0, 0.0, 0.0, 0.0
    min_all, max_all = np.inf, -np.inf
    min_seq_len, max_seq_len = np.inf, -np.inf

    # keep track of some stats to print in the end
    meta_stats_per_db = dict()

    for idx in range(len(poses)):
        pose = poses[idx]  # shape (seq_length, 135)
        assert len(pose) > 0, 'file is empty'

        db_name = "h36"
        if db_name not in meta_stats_per_db:
            meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

        if create_windows is not None:
            if pose.shape[0] < create_windows[0]:
                continue

            # first save it without splitting into windows
            tfexample = to_tfexample(pose, "{}/{}".format(0, file_ids[idx]), db_name, one_hots[idx])
            write_tfexample(tfrecord_writers_dyn, tfexample)

            # then split into windows and save later
            pose_w = split_into_windows(pose, create_windows[0], create_windows[1])
            assert pose_w.shape[1] == create_windows[0]

        else:
            pose_w = pose[np.newaxis, ...]

        for w in range(pose_w.shape[0]):
            poses_window = pose_w[w]
            tfexample = to_tfexample(poses_window, "{}/{}".format(w, file_ids[idx]), db_name, one_hots[idx])
            write_tfexample(tfrecord_writers, tfexample)

            meta_stats_per_db[db_name]['n_samples'] += 1
            meta_stats_per_db[db_name]['n_frames'] += poses_window.shape[0]

            # update normalization stats
            if compute_stats:
                seq_len, feature_size = poses_window.shape

                # Global mean&variance
                n_all += seq_len * feature_size
                delta_all = poses_window - mean_all
                mean_all = mean_all + delta_all.sum() / n_all
                m2_all = m2_all + (delta_all * (poses_window - mean_all)).sum()

                # Channel-wise mean&variance
                n_channel += seq_len
                delta_channel = poses_window - mean_channel
                mean_channel = mean_channel + delta_channel.sum(axis=0) / n_channel
                m2_channel = m2_channel + (delta_channel * (poses_window - mean_channel)).sum(axis=0)

                # Global min&max values.
                min_all = np.min(poses_window) if np.min(poses_window) < min_all else min_all
                max_all = np.max(poses_window) if np.max(poses_window) > max_all else max_all

                # Min&max sequence length.
                min_seq_len = seq_len if seq_len < min_seq_len else min_seq_len
                max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len

    close_tfrecord_writers(tfrecord_writers)
    if create_windows is not None:
        close_tfrecord_writers(tfrecord_writers_dyn)

    # print meta stats
    print()
    tot_samples = 0
    tot_frames = 0
    for db in meta_stats_per_db.keys():
        tot_frames += meta_stats_per_db[db]['n_frames']
        tot_samples += meta_stats_per_db[db]['n_samples']
        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db, meta_stats_per_db[db]['n_samples'],
                                                                  meta_stats_per_db[db]['n_frames']))

    print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))

    # finalize and save stats
    if compute_stats:
        var_all = m2_all / (n_all - 1)
        var_channel = m2_channel / (n_channel - 1)

        stats = {'mean_all': mean_all, 'mean_channel': mean_channel, 'var_all': var_all,
                 'var_channel': var_channel, 'min_all': min_all, 'max_all': max_all,
                 'min_seq_len': min_seq_len, 'max_seq_len': max_seq_len, 'num_samples': tot_samples}

        stats_file = os.path.join(output_path, 'stats.npz')
        print('saving statistics to {} ...'.format(stats_file))
        np.savez(stats_file, stats=stats)

    return meta_stats_per_db


def load_data(path_to_dataset, subjects, actions, one_hot, as_rotmat=False):
    """
    Borrowed and adapted from Martinez et al.

    Args
      path_to_dataset: string. directory where the data resides
      subjects: list of numbers. The subjects to load
      actions: list of string. The actions to load
      one_hot: Whether to add a one-hot encoding to the data
      as_rotmat: Whether to convert the data to rotation matrices
    Returns
      trainData: dictionary with k:v
        k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
      completeData: nxd matrix with all the data. Used to normlization stats
    """
    nactions = len(actions)

    poses = []
    one_hots = []
    file_ids = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):

            action = actions[action_idx]

            for subact in [1, 2]:  # subactions

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                action_sequence = readCSVasFloat(filename)

                n_samples, dof = action_sequence.shape
                n_joints = dof // 3

                if as_rotmat:
                    expmap = np.reshape(action_sequence, [n_samples*n_joints, 3])
                    # first three values are positions, so technically it's meaningless to convert them,
                    # but we do it anyway because later we discard this values anywho
                    rotmats = np.zeros([n_samples*n_joints, 3, 3])
                    for i in range(rotmats.shape[0]):
                        rotmats[i] = cv2.Rodrigues(expmap[i])[0]
                    rotmats = np.reshape(rotmats, [n_samples, n_joints*3*3])
                    action_sequence = rotmats

                if one_hot:
                    one = np.zeros([nactions], dtype=np.float)
                    one[action_idx] = 1.0
                    one_hots.append(one)

                poses.append(action_sequence)
                file_ids.append("S{}_{}_{}".format(subj, action, subact))

    return poses, one_hots, file_ids


if __name__ == '__main__':
    h36m_folder = "../data/h3.6m/dataset/"
    output_folder = "C:/Users/manuel/projects/motion-modelling/data/h3.6m/tfrecords/"
    n_shards = 1  # need to save the data in shards, it's too big otherwise
    train_subjects = [1, 6, 7, 8, 9, 11]  # for h3.6m this is fixed
    test_subjects = [5]  # for h3.6m this is fixed, use test subject as validation
    as_quat = False  # converts the data to quaternions
    as_aa = False  # converts tha data to angle_axis
    test_window_size = 160
    test_window_stride = 100

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    assert not (as_quat and as_aa), 'must choose between quat or aa'

    as_rotmat = not (as_quat or as_aa)
    assert as_aa or as_rotmat, "only implemented for angle-axis and rotmat"

    train_data, train_one_hot, train_ids = load_data(h36m_folder, train_subjects, actions,
                                                     one_hot=True, as_rotmat=as_rotmat)
    test_data, test_one_hot, test_ids = load_data(h36m_folder, train_subjects, actions,
                                                  one_hot=True, as_rotmat=as_rotmat)

    rep = "quat" if as_quat else "rotmat"
    rep = "aa" if as_aa else "rotmat"
    tr_stats = process_split(train_data, train_one_hot, train_ids, os.path.join(output_folder, rep, "training"),
                             n_shards, compute_stats=True, create_windows=None)

    print("process validation data ...")
    va_stats = process_split(test_data, test_one_hot, test_ids, os.path.join(output_folder, rep, "validation"),
                             n_shards, compute_stats=False, create_windows=(test_window_size, test_window_stride))

    print("process test data ...")
    te_stats = process_split(test_data, test_one_hot, test_ids, os.path.join(output_folder, rep, "test"),
                             n_shards, compute_stats=False, create_windows=(test_window_size, test_window_stride))

    print("Meta stats for all splits combined")
    total_stats = tr_stats
    for db in tr_stats.keys():
        for k in tr_stats[db].keys():
            total_stats[db][k] += va_stats[db][k] if db in va_stats else 0
            total_stats[db][k] += te_stats[db][k] if db in te_stats else 0

    tot_samples = 0
    tot_frames = 0
    for db in total_stats.keys():
        tot_frames += total_stats[db]['n_frames']
        tot_samples += total_stats[db]['n_samples']
        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db, total_stats[db]['n_samples'],
                                                                  total_stats[db]['n_frames']))

    print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))
