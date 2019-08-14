import os
import tensorflow as tf
import functools

tf.enable_eager_execution()


def get_fnames_in_tfrecords(tfrecords_path):

    def _parse_tf_example(proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "db_name": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
        }

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])

        # Remove chunk_id from chunk_id/file_id
        file_id = tf.strings.split([parsed_features["file_id"]], "/")
        file_id = tf.sparse.to_dense(file_id, default_value='')[0, -1]
        parsed_features["sample_id"] = tf.strings.join([parsed_features["db_name"], file_id], separator="/")
        return parsed_features

    def _to_model_inputs(tf_sample_dict):
        model_sample = dict()
        model_sample["sample_id"] = tf_sample_dict["sample_id"]
        return model_sample

    tf_data = tf.data.TFRecordDataset.list_files(tfrecords_path)
    tf_data = tf_data.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1)
    tf_data = tf_data.map(functools.partial(_parse_tf_example), 4)
    tf_data = tf_data.map(functools.partial(_to_model_inputs), 4)

    iterator = tf_data.make_one_shot_iterator()
    all_filenames = set()
    for s in iterator:
        filename = s["sample_id"].numpy().decode("utf-8")
        all_filenames.add(filename)

    return all_filenames


if __name__ == '__main__':
    base_path = "C:/users/manuel/projects/motion-modelling/data/from_dip"

    test_path = os.path.join(base_path, "aa/test_dynamic/amass-?????-of-?????")
    test_out = os.path.join(base_path, "test_fnames.txt")

    valid_path = os.path.join(base_path, "aa/validation_dynamic/amass-?????-of-?????")
    valid_out =  os.path.join(base_path, "validation_fnames.txt")

    train_path = os.path.join(base_path, "aa/training/amass-?????-of-?????")
    train_out =  os.path.join(base_path, "training_fnames.txt")

    def _write_fnames(to, fnames):
        with open(to, 'w') as fh:
            for fname in sorted(list(fnames)):
                fh.write(fname + "\n")

    _write_fnames(test_out, get_fnames_in_tfrecords(test_path))
    _write_fnames(valid_out, get_fnames_in_tfrecords(valid_path))
    _write_fnames(train_out, get_fnames_in_tfrecords(train_path))
