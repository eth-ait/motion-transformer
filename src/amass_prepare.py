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


if __name__ == '__main__':
    amass_folder = "C:/Users/manuel/projects/imu/data/new_batch_v9_unnorm"
    output_folder = "C:/Users/manuel/projects/motion-modelling/data/amass"
    n_shards = 20  # need to save the data in shards, it's too big otherwise

    # gather all file names first so that we can create the shards
    all_fnames = []
    for root_dir, dir_names, file_names in os.walk(amass_folder):
        for f in file_names:
            if f.endswith('.pkl'):
                all_fnames.append((root_dir, f))

    # find out size of each shard
    print("found {} files ...".format(len(all_fnames)))
    n_samples_per_shard = [len(all_fnames) // n_shards] * n_shards
    n_samples_per_shard[-1] += len(all_fnames) % n_shards  # make last one a bit bigger
    assert np.sum(n_samples_per_shard) == len(all_fnames), 'not using all the samples'

    # also save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_folder, 'tfrecords', 'amass'), n_shards)

    # compute normalization stats online
    n_all, mean_all, var_all, m2_all = 0.0, 0.0, 0.0, 0.0
    n_channel, mean_channel, var_channel, m2_channel = 0.0, 0.0, 0.0, 0.0
    min_all, max_all = np.inf, -np.inf
    min_seq_len, max_seq_len = np.inf, -np.inf

    # keep track of some stats to print in the end
    stats_per_db = dict()
    idx = 0
    tot_samples = 0

    for i, n_samples in enumerate(n_samples_per_shard):

        all_samples = dict()  # db_name -> list of samples

        for n in range(n_samples):
            root_dir, f = all_fnames[idx]
            with open(os.path.join(root_dir, f), 'rb') as f_handle:
                print('\rprocessing file {}'.format(f), end='')
                data = pkl.load(f_handle, encoding='latin1')
                poses = np.array(data['poses'])  # shape (seq_length, 135)

                # some files are empty
                if len(poses) > 0:
                    db_name = os.path.split(os.path.dirname(os.path.join(root_dir, f)))[1]
                    if "AMASS" in db_name:
                        db_name = '_'.join(db_name.split('_')[1:])
                    else:
                        db_name = db_name.split('_')[0]

                    if db_name not in all_samples:
                        all_samples[db_name] = {'poses': [], 'file_ids': []}
                    if db_name not in stats_per_db:
                        stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

                    all_samples[db_name]['poses'].append(poses)
                    all_samples[db_name]['file_ids'].append(f)

                    tfexample = to_tfexample(poses, f, db_name)
                    write_tfexample(tfrecord_writers, tfexample)

                    stats_per_db[db_name]['n_samples'] += 1
                    stats_per_db[db_name]['n_frames'] += poses.shape[0]
                    tot_samples += 1

                    # update normalization stats
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

                idx += 1

        # save to data directory with numpy
        out_file = os.path.join(output_folder, 'amass-shard{:0>2d}'.format(i))
        print('\nsaving to {}...'.format(out_file))
        np.savez(out_file, data=all_samples)

    # finalize stats
    var_all = m2_all / (n_all - 1)
    var_channel = m2_channel / (n_channel - 1)

    stats = {'mean_all': mean_all, 'mean_channel': mean_channel, 'var_all': var_all,
             'var_channel': var_channel, 'min_all': min_all, 'max_all': max_all,
             'min_seq_len': min_seq_len, 'max_seq_len': max_seq_len, 'num_samples': idx}

    # save stats
    stats_file = os.path.join(output_folder, 'stats.npz')
    print('saving statistics to {} ...'.format(stats_file))
    np.savez(stats_file, stats=stats)

    close_tfrecord_writers(tfrecord_writers)

    # print stats
    tot_frames = 0
    for db in stats_per_db.keys():
        tot_frames += stats_per_db[db]['n_frames']

        print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format(db, stats_per_db[db]['n_samples'],
                                                                  stats_per_db[db]['n_frames']))

    print('{:>20} -> {:>4d} sequences, {:>12d} frames'.format('Total', tot_samples, tot_frames))
