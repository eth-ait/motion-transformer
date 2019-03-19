"""
Store SRNN poses into tfrecords for evaluation.
"""

import numpy as np
import os
import tensorflow as tf

from amass_prepare import create_tfrecord_writers
from amass_prepare import write_tfexample
from amass_prepare import close_tfrecord_writers


RNG = np.random.RandomState(42)


def to_tfexample(euler_angles, file_id, db_name):
    features = dict()
    features['file_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode('utf-8')]))
    features['db_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db_name.encode('utf-8')]))
    features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=euler_angles.shape))
    features['euler_angles'] = tf.train.Feature(float_list=tf.train.FloatList(value=euler_angles.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def process_split(euler_angles, action_labels, output_path, n_shards):
    print("storing into {}".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, 'amass'), n_shards)

    # keep track of some stats to print in the end
    meta_stats_per_db = dict()

    for idx in range(len(euler_angles)):
        eul = euler_angles[idx]
        action = action_labels[idx]
        file_id = "{}/{}".format(idx, action)

        db_name = "h36"
        if db_name not in meta_stats_per_db:
            meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

        tfexample = to_tfexample(eul, file_id, db_name)
        write_tfexample(tfrecord_writers, tfexample)

        meta_stats_per_db[db_name]['n_samples'] += 1
        meta_stats_per_db[db_name]['n_frames'] += eul.shape[0]

    close_tfrecord_writers(tfrecord_writers)

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

    return meta_stats_per_db


if __name__ == '__main__':
    srnn_poses = "../data/h3.6m/h36m_srnn_test_samples_50fps.npz"
    output_folder = "C:/Users/manuel/projects/motion-modelling/data/h3.6m/tfrecords/"
    n_shards = 1  # need to save the data in shards, it's too big otherwise
    as_quat = False  # converts the data to quaternions
    as_aa = False  # converts tha data to angle_axis

    assert not (as_quat and as_aa), 'must choose between quat or aa'

    data = dict(np.load(srnn_poses))

    # NOTE: samples here are euler angles!
    samples = data['samples']  # list of np arrays of shape (batch_size, 150, n_joints*3)
    action_labels = data['labels']

    output_path = os.path.join(output_folder, "euler_srnn_poses")
    _ = process_split(samples, action_labels, output_path, n_shards)
