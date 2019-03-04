import tensorflow as tf
import numpy as np
import os
import functools


class Dataset(object):
    """
    A base wrapper class around tf.data.Dataset API. Depending on the dataset requirements, it applies data
    transformations.
    """

    def __init__(self, data_path, meta_data_path, batch_size):
        self.tf_dataset = None
        self.data_path = data_path
        self.batch_size = batch_size

        # Load statistics and other data summary stored in the meta-data file.
        self.meta_data = self.load_meta_data(meta_data_path)
        self.data_summary()

        self.mean_all = self.meta_data['mean_all']
        self.var_all = self.meta_data['var_all']
        self.mean_channel = self.meta_data['mean_channel']
        self.var_channel = self.meta_data['var_channel']

        self.tf_data_transformations()
        self.tf_data_normalization()
        self.tf_data_to_model()

    def load_meta_data(self, meta_data_path):
        """
        Loads meta-data file given the path. It is assumed to be in numpy.
        Args:
            meta_data_path:
        Returns:
            Meta-data dictionary or False if it is not found.
        """
        if not meta_data_path or not os.path.exists(meta_data_path):
            print("Meta-data not found.")
            return False
        else:
            return dict(np.load(meta_data_path))

    def tf_data_transformations(self):
        raise NotImplementedError('Subclass must override sample method')

    def tf_data_normalization(self):
        raise NotImplementedError('Subclass must override sample method')

    def tf_data_to_model(self):
        raise NotImplementedError('Subclass must override sample method')

    def create_meta_data(self):
        raise NotImplementedError('Subclass must override sample method')

    def data_summary(self):
        raise NotImplementedError('Subclass must override sample method')

    def normalize_zero_mean_unit_variance_all(self, sample_dict, key):
        sample_dict[key] = (sample_dict[key] - self.mean_all) / self.var_all
        return sample_dict

    def normalize_zero_mean_unit_variance_channel(self, sample_dict, key):
        sample_dict[key] = (sample_dict[key] - self.mean_channel) / self.var_channel
        return sample_dict

    def __iter__(self):
        return self.tf_dataset.__iter__()


class TFRecordMotionDataset(Dataset):
    """
    Dataset class for AMASS dataset stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size):
        super(TFRecordMotionDataset, self).__init__(data_path, meta_data_path, batch_size)

    def load_meta_data(self, meta_data_path):
        """
        Loads meta-data file given the path. It is assumed to be in numpy.
        Args:
            meta_data_path:
        Returns:
            Meta-data dictionary or False if it is not found.
        """
        if not meta_data_path or not os.path.exists(meta_data_path):
            print("Meta-data not found.")
            return False
        else:
            return np.load(meta_data_path)['stats'].tolist()

    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        self.tf_dataset = tf.data.TFRecordDataset.list_files(self.data_path, seed=42, shuffle=True)
        self.tf_dataset = self.tf_dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=1)
        self.tf_dataset = self.tf_dataset.map(functools.partial(self.__parse_tfexample_fn), num_parallel_calls=4)

    def tf_data_normalization(self):
        # Applies normalization.
        self.tf_dataset = self.tf_dataset.prefetch(self.batch_size * 2)
        self.tf_dataset = self.tf_dataset.map(functools.partial(self.normalize_zero_mean_unit_variance_channel,
                                                                key="poses"))

    def tf_data_to_model(self):
        # Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
        self.tf_dataset = self.tf_dataset.map(functools.partial(self.__to_model_batch))
        self.tf_dataset = self.tf_dataset.padded_batch(self.batch_size, padded_shapes=self.tf_dataset.output_shapes)

    def create_meta_data(self):
        """We assume meta data always exists."""
        raise RuntimeError("We do not create here.")

    def data_summary(self):
        pass

    def __to_model_batch(self, tf_sample_dict):
        """
        Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
        the required fields by the models.
        Args:
            tf_sample_dict:
        Returns:
        """
        model_sample = dict()
        model_sample["batch_seq_len"] = tf_sample_dict["shape"][0]
        model_sample["batch_input"] = tf_sample_dict["poses"]
        model_sample["batch_target"] = tf_sample_dict["poses"]
        model_sample["file_id"] = tf_sample_dict["file_id"]
        return model_sample

    def __parse_tfexample_fn(self, proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "db_name": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
            }

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])
        return parsed_features


if __name__ == '__main__':

    def log_stats(stats, tag="Online"):
        print("[{2}] mean: {0}, std: {1}".format(stats["mean_all"], stats["var_all"], tag))
        print("[{2}] mean channel: {0}, std channel: {1}".format(stats["mean_channel"], stats["var_channel"], tag))
        print("[{2}] min value: {0}, max value: {1}".format(stats["min_all"], stats["max_all"], tag))
        print("[{2}] min length: {0}, max length: {1}".format(stats["min_seq_len"], stats["max_seq_len"], tag))
        print("============")

    # some tests
    tfrecord_pattern = "../data/amass/tfrecords/amass-?????-of-?????"
    dataset = TFRecordMotionDataset(tfrecord_pattern,
                                    "../data/amass/stats.npz",
                                    32)

    stats = dataset.meta_data
    # log_stats(stats, "OnlineTFRecord")

    train_iterator = dataset.tf_dataset.make_one_shot_iterator()
    next_batch = train_iterator.get_next()

    counter = 0
    with tf.Session() as sess:
        try:
            while True:
                b = sess.run(next_batch)
                print(b["file_id"])
                counter += len(b["file_id"])
        except tf.errors.OutOfRangeError:
            pass

    print("found {} samples".format(counter))
