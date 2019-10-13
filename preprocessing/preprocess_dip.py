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
import argparse
import numpy as np
import os
import pickle as pkl
import tensorflow as tf
import quaternion
import cv2


RNG = np.random.RandomState(42)


def create_tfrecord_writers(output_file, n_shards):
    writers = []
    for i in range(n_shards):
        writers.append(tf.python_io.TFRecordWriter("{}-{:0>5d}-of-{:0>5d}".format(output_file, i, n_shards)))
    return writers


def close_tfrecord_writers(writers):
    for w in writers:
        w.close()


def write_tfexample(writers, tf_example):
    random_writer_idx = RNG.randint(0, len(writers))
    writers[random_writer_idx].write(tf_example.SerializeToString())


def to_tfexample(poses, file_id, db_name):
    features = dict()
    features['file_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode('utf-8')]))
    features['db_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db_name.encode('utf-8')]))
    features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=poses.shape))
    features['poses'] = tf.train.Feature(float_list=tf.train.FloatList(value=poses.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def split_into_windows(poses, window_size, stride):
    """Split (seq_length, dof) array into arrays of shape (window_size, dof) with the given stride."""
    n_windows = (poses.shape[0] - window_size) // stride + 1
    windows = poses[stride * np.arange(n_windows)[:, None] + np.arange(window_size)]
    return windows


def correct_antipodal_quaternions(quat):
    """
    Removes discontinuities coming from antipodal representation of quaternions. At time step t it checks which
    representation, q or -q, is closer to time step t-1 and chooses the closest one.
    Args:
        quat: numpy array of shape (N, K, 4) where N is the number of frames and K the number of joints. K is optional,
          i.e. can be 0.

    Returns: numpy array of shape (N, K, 4) with fixed antipodal representation
    """
    assert len(quat.shape) == 3 or len(quat.shape) == 2
    assert quat.shape[-1] == 4

    if len(quat.shape) == 2:
        quat_r = quat[:, np.newaxis].copy()
    else:
        quat_r = quat.copy()

    def dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

    # Naive implementation looping over all time steps sequentially.
    # For a faster implementation check the QuaterNet paper.
    quat_corrected = np.zeros_like(quat_r)
    quat_corrected[0] = quat_r[0]
    for t in range(1, quat.shape[0]):
        diff_to_plus = dist(quat_r[t], quat_corrected[t - 1])
        diff_to_neg = dist(-quat_r[t], quat_corrected[t - 1])

        # diffs are vectors
        qc = quat_r[t]
        swap_idx = np.where(diff_to_neg < diff_to_plus)
        qc[swap_idx] = -quat_r[t, swap_idx]
        quat_corrected[t] = qc
    quat_corrected = np.squeeze(quat_corrected)
    return quat_corrected


def rotmat2quat(rotmats):
    """
    Convert rotation matrices to quaternions. It ensures that there's no switch to the antipodal representation
    within this sequence of rotations.
    Args:
        oris: np array of shape (seq_length, n_joints*9).

    Returns: np array of shape (seq_length, n_joints*4)
    """
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    ori = np.reshape(rotmats, [seq_length, -1, 3, 3])
    ori_q = quaternion.as_float_array(quaternion.from_rotation_matrix(ori))
    ori_qc = correct_antipodal_quaternions(ori_q)
    ori_qc = np.reshape(ori_qc, [seq_length, -1])
    return ori_qc


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis format.
    Args:
        oris: np array of shape (seq_length, n_joints*9).

    Returns: np array of shape (seq_length, n_joints*3)
    """
    seq_length = rotmats.shape[0]
    assert rotmats.shape[1] % 9 == 0
    n_joints = rotmats.shape[1] // 9
    ori = np.reshape(rotmats, [seq_length*n_joints, 3, 3])
    aas = np.zeros([seq_length*n_joints, 3])
    for i in range(ori.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(ori[i])[0])
    return np.reshape(aas, [seq_length, n_joints*3])


def process_split(all_fnames, output_path, n_shards, compute_stats, rep, create_windows=None):
    """
    Process data into tfrecords.
    Args:
        all_fnames: List of filenames that should be processed.
        output_path: Where to store the tfrecord files.
        n_shards: How many tfrecord files to create.
        compute_stats: Whether to compute and store normalization statistics.
        rep: If the output data should be rotation matrices, quaternions or angle-axis.
        create_windows: Tuple (size, stride) of windows that should be extracted from each sequence or None otherwise.
          If given, it will also store a version where not windows were extracted, stored under a folder with suffix
          '*_dynamic'. This is helpful for validation and test splits, as they can become quite big if windows are
          extracted.

    Returns:
        Some meta statistics (how many sequences processed etc.).
    """
    assert rep in ["aa", "rotmat", "quat"]
    print("storing into {} computing stats {}".format(output_path, "YES" if compute_stats else "NO"))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, 'amass'), n_shards)
    tfrecord_writers_dyn = None
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

    for idx in range(len(all_fnames)):
        root_dir, f, file_id = all_fnames[idx]
        with open(os.path.join(root_dir, f), 'rb') as f_handle:
            print('\r [{:0>5d} / {:0>5d}] processing file {}'.format(idx + 1, len(all_fnames), f), end='')
            data = pkl.load(f_handle, encoding='latin1')
            poses = np.array(data['poses'])  # shape (seq_length, 135)
            assert len(poses) > 0, 'file is empty'

            if rep == "quat":
                # convert to quaternions
                poses = rotmat2quat(poses)
            elif rep == "aa":
                poses = rotmat2aa(poses)
            else:
                pass

            db_name = file_id.split('/')[0]
            if db_name not in meta_stats_per_db:
                meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

            if create_windows is not None:
                if poses.shape[0] < create_windows[0]:
                    continue

                # first save it without splitting into windows
                tfexample = to_tfexample(poses, "{}/{}".format(0, file_id), db_name)
                write_tfexample(tfrecord_writers_dyn, tfexample)

                # then split into windows and save later
                poses_w = split_into_windows(poses, create_windows[0], create_windows[1])
                assert poses_w.shape[1] == create_windows[0]

            else:
                poses_w = poses[np.newaxis, ...]

            for w in range(poses_w.shape[0]):
                poses_window = poses_w[w]
                tfexample = to_tfexample(poses_window, "{}/{}".format(w, file_id), db_name)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Location of the downloaded and unpacked zip file.")
    parser.add_argument("--output_dir", required=True, help="Where to store the tfrecords.")
    parser.add_argument("--split_dir", default="./", help="Where the text files defining the data splits are stored.")
    parser.add_argument("--n_shards", type=int, default=20, help="How many tfrecord files to create per split.")
    parser.add_argument("--as_quat", action="store_true", help="Whether to convert data to quaternions.")
    parser.add_argument("--as_aa", action="store_true", help="Whether to convert data to angle-axis.")
    parser.add_argument("--window_size", type=int, default=180, help="Window size for test and val, in frames.")
    parser.add_argument("--window_stride", type=int, default=120, help="Window stride for test and val, in frames.")

    args = parser.parse_args()

    assert not (args.as_quat and args.as_aa), 'must choose between quaternion or angle-axis representation'

    # Load training, validation and test split.
    def _read_fnames(from_):
        with open(from_, 'r') as fh:
            lines = fh.readlines()
            return [line.strip() for line in lines]

    train_fnames = _read_fnames(os.path.join(args.split_dir, 'training_fnames.txt'))
    valid_fnames = _read_fnames(os.path.join(args.split_dir, 'validation_fnames.txt'))
    test_fnames = _read_fnames(os.path.join(args.split_dir, 'test_fnames.txt'))

    print("Pre-determined splits: {} train, {} valid, {} test.".format(len(train_fnames),
                                                                       len(valid_fnames),
                                                                       len(test_fnames)))

    # Load all available filenames from the source directory.
    train_fnames_avail = []
    test_fnames_avail = []
    valid_fnames_avail = []
    for root_dir, dir_names, file_names in os.walk(args.input_dir):
        dir_names.sort()
        for f in sorted(file_names):
            if f.endswith('.pkl'):
                # Extract name of the database.
                db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
                db_name = '_'.join(db_name.split('_')[1:]) if "AMASS" in db_name else db_name.split('_')[0]
                file_id = "{}/{}".format(db_name, f)

                if file_id in train_fnames:
                    train_fnames_avail.append((root_dir, f, file_id))
                elif file_id in valid_fnames:
                    valid_fnames_avail.append((root_dir, f, file_id))
                elif file_id in test_fnames:
                    test_fnames_avail.append((root_dir, f, file_id))
                else:
                    # This file was rejected by us because its total sequence length is smaller than 180 (3 seconds)
                    pass

    tot_files = len(train_fnames_avail) + len(test_fnames_avail) + len(valid_fnames_avail)
    print("found {} training files {:.2f} %".format(len(train_fnames_avail), len(train_fnames_avail) / tot_files * 100.0))
    print("found {} validation files {:.2f} %".format(len(valid_fnames_avail), len(valid_fnames_avail) / tot_files * 100.0))
    print("found {} test files {:.2f} %".format(len(test_fnames_avail), len(test_fnames_avail) / tot_files * 100.0))

    # print("process training data ...")
    rep = "quat" if args.as_quat else "aa" if args.as_aa else "rotmat"
    tr_stats = process_split(train_fnames_avail, os.path.join(args.output_dir, rep, "training"),
                             args.n_shards, compute_stats=True, rep=rep,
                             create_windows=None)

    print("process validation data ...")
    va_stats = process_split(valid_fnames_avail, os.path.join(args.output_dir, rep, "validation"),
                             args.n_shards, compute_stats=False, rep=rep,
                             create_windows=(args.window_size, args.window_stride))

    print("process test data ...")
    te_stats = process_split(test_fnames_avail, os.path.join(args.output_dir, rep, "test"),
                             args.n_shards, compute_stats=False, rep=rep,
                             create_windows=(args.window_size, args.window_stride))

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
