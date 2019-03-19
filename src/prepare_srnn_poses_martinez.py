"""
Store SRNN poses into tfrecords for evaluation.
"""

import numpy as np
import os
import tensorflow as tf
import quaternion

from amass_prepare import create_tfrecord_writers
from amass_prepare import write_tfexample
from amass_prepare import close_tfrecord_writers


RNG = np.random.RandomState(42)


def to_tfexample(pose, euler_target, file_id, db_name):
    features = dict()
    features['file_id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_id.encode('utf-8')]))
    features['db_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[db_name.encode('utf-8')]))
    features['euler_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=euler_target.shape))
    features['euler_targets'] = tf.train.Feature(float_list=tf.train.FloatList(value=euler_target.flatten()))
    features['pose_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=pose.shape))
    features['poses'] = tf.train.Feature(float_list=tf.train.FloatList(value=pose.flatten()))
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def process_poses(data, rep, output_path, n_shards):
    assert rep in ["aa", "rotmat", "quat"]
    print("storing into {}".format(output_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save data as tfrecords
    tfrecord_writers = create_tfrecord_writers(os.path.join(output_path, 'amass'), n_shards)

    # NOTE: samples here are euler angles!
    poses_q = data['quaternion_samples']  # list of np arrays of shape (batch_size, 150, n_joints*4)
    euler_targets = data['euler_samples']  # list of np arrays of shape (batch_size, 150, n_joints*3)
    action_labels = data['labels']
    n_samples = len(poses_q)
    assert n_samples == len(euler_targets) == len(action_labels)

    # keep track of some stats to print in the end
    meta_stats_per_db = dict()

    for idx in range(n_samples):
        pose = poses_q[idx]
        seq_len = pose.shape[0]
        pose_r = quaternion.from_float_array(np.reshape(pose, [seq_len, -1, 4]))
        if rep == "aa":
            pose_aa = quaternion.as_rotation_vector(pose_r)
            pose = np.reshape(pose_aa, [seq_len, -1])
        elif rep == "rotmat":
            pose_rot = quaternion.as_rotation_matrix(pose_r)
            pose = np.reshape(pose_rot, [seq_len, -1])
        else:
            pass  # data is already in quaternion format

        eul = euler_targets[idx]
        action = action_labels[idx]
        file_id = "{}/{}".format(idx, action)

        db_name = "h36"
        if db_name not in meta_stats_per_db:
            meta_stats_per_db[db_name] = {'n_samples': 0, 'n_frames': 0}

        tfexample = to_tfexample(pose, eul, file_id, db_name)
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
    as_aa = True  # converts tha data to angle_axis

    assert not (as_quat and as_aa), 'must choose between quat or aa'

    data = dict(np.load(srnn_poses))

    rep = "quat" if as_quat else "aa" if as_aa else "rotmat"
    output_path = os.path.join(output_folder, rep, "srnn_poses")
    _ = process_poses(data, rep, output_path, n_shards)
