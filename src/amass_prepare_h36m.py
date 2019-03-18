import numpy as np
import os
import pickle as pkl
import cv2

from amass_prepare import create_tfrecord_writers
from amass_prepare import close_tfrecord_writers
from amass_prepare import write_tfexample
from amass_prepare import to_tfexample
from amass_prepare import split_into_windows
from amass_prepare import rotmat2quat


RNG = np.random.RandomState(42)


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis.
    Args:
        oris: np array of shape (seq_length, n_joints*9).

    Returns: np array of shape (seq_length, n_joints*3)
    """
    n_joints = rotmats.shape[-1]//9
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, [-1, n_joints*3])


def process_split(all_fnames, output_path, n_shards, compute_stats, rep, create_windows=None):
    assert rep in ["rotmat", "quat", "aa"]
    print("storing into {} computing stats {}".format(output_path, "YES" if compute_stats else "NO"))

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

    for idx in range(len(all_fnames)):
        root_dir, f = all_fnames[idx]
        with open(os.path.join(root_dir, f), 'rb') as f_handle:
            print('\r [{:0>5d} / {:0>5d}] processing file {}'.format(idx+1, len(all_fnames), f), end='')
            data = pkl.load(f_handle, encoding='latin1')
            poses = np.array(data['poses'])  # shape (seq_length, 135)
            assert len(poses) > 0, 'file is empty'

            if rep == "quat":
                # convert to quaternions
                poses = rotmat2quat(poses)
            elif rep == "aa":
                # convert to angle-axis
                poses = rotmat2aa(poses)

            db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
            if "AMASS" in db_name:
                db_name = '_'.join(db_name.split('_')[1:])
            else:
                db_name = db_name.split('_')[0]

            if db_name not in meta_stats_per_db:
                meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

            if create_windows is not None:
                if poses.shape[0] < create_windows[0]:
                    continue

                # first save it without splitting into windows
                tfexample = to_tfexample(poses, "{}/{}".format(0, f), db_name)
                write_tfexample(tfrecord_writers_dyn, tfexample)

                # then split into windows and save later
                poses_w = split_into_windows(poses, create_windows[0], create_windows[1])
                assert poses_w.shape[1] == create_windows[0]

            else:
                poses_w = poses[np.newaxis, ...]

            for w in range(poses_w.shape[0]):
                poses_window = poses_w[w]
                tfexample = to_tfexample(poses_window, "{}/{}".format(w, f), db_name)
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


if __name__ == '__main__':
    amass_h36m_folder = "C:/Users/manuel/projects/imu/data/new_batch_v9_unnorm/H36_60FPS_for_TC_No_Blend"
    output_folder = "C:/Users/manuel/projects/motion-modelling/data/amass/per_db/h36m"
    n_shards = 5
    train_subjects = [1, 6, 7, 8, 9, 11]  # for h3.6m this is fixed
    test_subjects = [5]  # for h3.6m this is fixed, use test subject as validation
    as_quat = False  # converts the data to quaternions
    as_aa = True  # converts tha data to angle_axis
    test_window_size = 180  # 3 seconds
    test_window_stride = 120  # 2 seconds

    assert not (as_quat and as_aa), 'must choose between quat or aa'

    train_fnames = []
    valid_fnames = []
    test_fnames = []
    for root_dir, dir_names, file_names in os.walk(amass_h36m_folder):
        dir_names.sort()
        for f in sorted(file_names):
            if f.endswith('.pkl'):
                # extract subject ID
                subject_id = int(f.split('_')[0][1:])
                if subject_id in train_subjects:
                    train_fnames.append((root_dir, f))
                else:
                    valid_fnames.append((root_dir, f))
                    test_fnames.append((root_dir, f))

    # make sure the splits are distinct
    training_set = set(train_fnames)
    test_set = set(test_fnames)
    assert len(training_set.intersection(test_set)) == 0

    tot_files = len(train_fnames) + len(valid_fnames) + len(test_fnames)
    print("found {} training files {:.2f} %".format(len(train_fnames), len(train_fnames) / tot_files * 100.0))
    print("found {} validation files {:.2f} %".format(len(valid_fnames), len(valid_fnames) / tot_files * 100.0))
    print("found {} test files {:.2f} %".format(len(test_fnames), len(test_fnames) / tot_files * 100.0))

    print("process training data ...")
    rep = "quat" if as_quat else "aa" if as_aa else "rotmat"
    tr_stats = process_split(train_fnames, os.path.join(output_folder, rep, "training"), n_shards, compute_stats=True,
                             rep=rep, create_windows=None)

    print("process validation data ...")
    va_stats = process_split(valid_fnames, os.path.join(output_folder, rep, "validation"), n_shards, compute_stats=False,
                             rep=rep, create_windows=(test_window_size, test_window_stride))

    print("process test data ...")
    te_stats = process_split(test_fnames, os.path.join(output_folder, rep, "test"), n_shards, compute_stats=False,
                             rep=rep, create_windows=(test_window_size, test_window_stride))

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
