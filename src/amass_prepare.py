import numpy as np
import os
import pickle as pkl
import tensorflow as tf

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


def process_split(all_fnames, output_path, n_shards, compute_stats):
    print("storing into {} computing stats {}".format(output_path, "YES" if compute_stats else "NO"))

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, 'amass'), n_shards)

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
            print('\rprocessing file {}'.format(f), end='')
            data = pkl.load(f_handle, encoding='latin1')
            poses = np.array(data['poses'])  # shape (seq_length, 135)
            assert len(poses) > 0, 'file is empty'

            db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
            if "AMASS" in db_name:
                db_name = '_'.join(db_name.split('_')[1:])
            else:
                db_name = db_name.split('_')[0]

            if db_name not in meta_stats_per_db:
                meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

            tfexample = to_tfexample(poses, f, db_name)
            write_tfexample(tfrecord_writers, tfexample)

            meta_stats_per_db[db_name]['n_samples'] += 1
            meta_stats_per_db[db_name]['n_frames'] += poses.shape[0]

            # update normalization stats
            if compute_stats:
                seq_len, feature_size = poses.shape

                # Global mean&variance
                n_all += seq_len * feature_size
                delta_all = poses - mean_all
                mean_all = mean_all + delta_all.sum() / n_all
                m2_all = m2_all + (delta_all * (poses - mean_all)).sum()

                # Channel-wise mean&variance
                n_channel += seq_len
                delta_channel = poses - mean_channel
                mean_channel = mean_channel + delta_channel.sum(axis=0) / n_channel
                m2_channel = m2_channel + (delta_channel * (poses - mean_channel)).sum(axis=0)

                # Global min&max values.
                min_all = np.min(poses) if np.min(poses) < min_all else min_all
                max_all = np.max(poses) if np.max(poses) > max_all else max_all

                # Min&max sequence length.
                min_seq_len = seq_len if seq_len < min_seq_len else min_seq_len
                max_seq_len = seq_len if seq_len > max_seq_len else max_seq_len

    # finalize and save stats
    if compute_stats:
        var_all = m2_all / (n_all - 1)
        var_channel = m2_channel / (n_channel - 1)

        stats = {'mean_all': mean_all, 'mean_channel': mean_channel, 'var_all': var_all,
                 'var_channel': var_channel, 'min_all': min_all, 'max_all': max_all,
                 'min_seq_len': min_seq_len, 'max_seq_len': max_seq_len, 'num_samples': len(all_fnames)}

        stats_file = os.path.join(output_path, 'stats.npz')
        print('saving statistics to {} ...'.format(stats_file))
        np.savez(stats_file, stats=stats)

    close_tfrecord_writers(tfrecord_writers)

    # print meta stats
    tot_samples = len(all_fnames)
    tot_frames = 0
    for db in meta_stats_per_db.keys():
        tot_frames += meta_stats_per_db[db]['n_frames']
        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db, meta_stats_per_db[db]['n_samples'],
                                                                  meta_stats_per_db[db]['n_frames']))

    print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))

    return meta_stats_per_db


if __name__ == '__main__':
    amass_folder = "C:/Users/manuel/projects/imu/data/new_batch_v9_unnorm"
    output_folder = "C:/Users/manuel/projects/motion-modelling/data/amass/tfrecords"
    n_shards = 20  # need to save the data in shards, it's too big otherwise
    valid_split = 0.05  # percentage of files we want to save for validation
    test_split = 0.05  # percentage of files we want to save for test

    # gather all file names to create the training/val/test splits
    # this assumes the files are not empty
    # we're sorthing so that the order is not dependent on the OS
    train_fnames = []
    valid_fnames = []
    test_fnames = []
    for root_dir, dir_names, file_names in os.walk(amass_folder):
        dir_names.sort()
        for f in sorted(file_names):
            if f.endswith('.pkl'):
                # special case for S5 in H36M => should always be in the test set
                db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
                if db_name.lower().startswith("h36_") and f.lower().startswith("s5_"):
                    test_fnames.append((root_dir, f))
                    continue

                # otherwise assign to validation or test by chance
                is_not_train = RNG.binomial(1, valid_split + test_split)
                if is_not_train:
                    is_valid = RNG.binomial(1, valid_split / (valid_split + test_split))
                    if is_valid:
                        valid_fnames.append((root_dir, f))
                    else:
                        test_fnames.append((root_dir, f))
                else:
                    train_fnames.append((root_dir, f))

    # make sure the splits are distinct
    training_set = set(train_fnames)
    validation_set = set(valid_fnames)
    test_set = set(test_fnames)
    assert len(training_set.intersection(validation_set)) == 0
    assert len(training_set.intersection(test_set)) == 0
    assert len(validation_set.intersection(test_set)) == 0

    tot_files = len(train_fnames) + len(valid_fnames) + len(test_fnames)
    print("found {} training files {:.2f} %".format(len(train_fnames), len(train_fnames) / tot_files * 100.0))
    print("found {} validation files {:.2f} %".format(len(valid_fnames), len(valid_fnames) / tot_files * 100.0))
    print("found {} test files {:.2f} %".format(len(test_fnames), len(test_fnames) / tot_files * 100.0))

    print("process training data ...")
    tr_stats = process_split(train_fnames, os.path.join(output_folder, "training"), n_shards, compute_stats=True)

    print("process validation data ...")
    va_stats = process_split(valid_fnames, os.path.join(output_folder, "validation"), n_shards, compute_stats=False)

    print("process test data ...")
    te_stats = process_split(test_fnames, os.path.join(output_folder, "test"), n_shards, compute_stats=False)

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
