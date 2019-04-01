import tensorflow as tf
import numpy as np
import os
import functools

from fk import H36M_MAJOR_JOINTS
from constants import Constants as C


class Dataset(object):
    """
    A base wrapper class around tf.data.Dataset API. Depending on the dataset requirements, it applies data
    transformations.
    """

    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        self.tf_data = None
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle

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

        if tf.executing_eagerly():
            self.iterator = self.tf_data.make_one_shot_iterator()
            self.tf_samples = None
        else:
            self.iterator = self.tf_data.make_initializable_iterator()
            self.tf_samples = self.iterator.get_next()

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

    def unnormalize_zero_mean_unit_variance_all(self, sample_dict, key):
        sample_dict[key] = sample_dict[key] * self.var_all + self.mean_all
        return sample_dict

    def unnormalize_zero_mean_unit_variance_channel(self, sample_dict, key):
        sample_dict[key] = sample_dict[key] * self.var_channel + self.mean_channel
        return sample_dict

    def get_iterator(self):
        return self.iterator

    def get_tf_samples(self):
        return self.tf_samples


class TFRecordMotionDataset(Dataset):
    """
    Dataset class for AMASS dataset stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        # Extract a window randomly. If the sequence is shorter, ignore it.
        self.extract_windows_of = kwargs.get("extract_windows_of", 0)
        # Whether to extract windows randomly or from the beginning of the sequence.
        self.extract_random_windows = kwargs.get("extract_random_windows", True)
        self.length_threshold = kwargs.get("length_threshold", self.extract_windows_of)
        self.num_parallel_calls = kwargs.get("num_parallel_calls", 16)
        self.normalize = kwargs.get("normalize", True)

        super(TFRecordMotionDataset, self).__init__(data_path, meta_data_path, batch_size, shuffle, **kwargs)

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
        tf_data_opt = tf.data.Options()
        tf_data_opt.experimental_autotune = True

        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
        self.tf_data = self.tf_data.map(functools.partial(self.__parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)
        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(self.batch_size*10)

        if self.extract_windows_of > 0:
            self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))
            if self.extract_random_windows:
                self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_randomly),
                                                num_parallel_calls=self.num_parallel_calls)
            else:
                self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_from_beginning),
                                                num_parallel_calls=self.num_parallel_calls)

    def tf_data_normalization(self):
        # Applies normalization.
        if self.normalize:
            self.tf_data = self.tf_data.map(
                functools.partial(self.normalize_zero_mean_unit_variance_channel, key="poses"),
                num_parallel_calls=self.num_parallel_calls)
        else:  # Some models require the feature size.
            self.tf_data = self.tf_data.map(functools.partial(self.__pp_set_feature_size),
                                            num_parallel_calls=self.num_parallel_calls)

    def unnormalize_zero_mean_unit_variance_all(self, sample_dict, key):
        if self.normalize:
            return super(TFRecordMotionDataset, self).unnormalize_zero_mean_unit_variance_all(sample_dict, key)
        else:
            return sample_dict

    def unnormalize_zero_mean_unit_variance_channel(self, sample_dict, key):
        if self.normalize:
            return super(TFRecordMotionDataset, self).unnormalize_zero_mean_unit_variance_channel(sample_dict, key)
        else:
            return sample_dict

    def tf_data_to_model(self):
        # Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
        self.tf_data = self.tf_data.map(functools.partial(self.__to_model_inputs), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=self.tf_data.output_shapes)
        self.tf_data = self.tf_data.prefetch(2)
        self.tf_data = self.tf_data.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))

    def create_meta_data(self):
        """We assume meta data always exists."""
        raise RuntimeError("We do not create here.")

    def data_summary(self):
        pass

    def __pp_set_feature_size(self, sample):
        seq_len = sample["poses"].get_shape().as_list()[0]
        sample["poses"].set_shape([seq_len, self.mean_channel.shape[0]])
        return sample

    def __pp_filter(self, sample):
        return tf.shape(sample["poses"])[0] >= self.length_threshold

    def __pp_get_windows_randomly(self, sample):
        start = tf.random_uniform((1, 1), minval=0, maxval=tf.shape(sample["poses"])[0]-self.extract_windows_of+1, dtype=tf.int32)[0][0]
        end = tf.minimum(start+self.extract_windows_of, tf.shape(sample["poses"])[0])
        sample["poses"] = sample["poses"][start:end, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def __pp_get_windows_from_beginning(self, sample):
        sample["poses"] = sample["poses"][0:self.extract_windows_of, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def __to_model_inputs(self, tf_sample_dict):
        """
        Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
        the required fields by the models.
        Args:
            tf_sample_dict:
        Returns:
        """
        model_sample = dict()
        model_sample[C.BATCH_SEQ_LEN] = tf_sample_dict["shape"][0]
        model_sample[C.BATCH_INPUT] = tf_sample_dict["poses"]
        model_sample[C.BATCH_TARGET] = tf_sample_dict["poses"]
        model_sample[C.BATCH_ID] = tf_sample_dict["sample_id"]
        return model_sample

    def __parse_single_tfexample_fn(self, proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "db_name": tf.FixedLenFeature([], dtype=tf.string),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
        }

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])

        # Remove ".pkl" extension.
        file_id = tf.strings.substr(parsed_features["file_id"], 0, tf.strings.length(parsed_features["file_id"]) - 4)
        parsed_features["sample_id"] = tf.strings.join([parsed_features["db_name"], file_id], separator="/")

        return parsed_features


class SRNNTFRecordMotionDataset(TFRecordMotionDataset):
    """
    Dataset class for AMASS dataset stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        super(SRNNTFRecordMotionDataset, self).__init__(data_path, meta_data_path, batch_size, shuffle, **kwargs)

    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        tf_data_opt = tf.data.Options()
        tf_data_opt.experimental_autotune = True

        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
        self.tf_data = self.tf_data.map(functools.partial(self.__parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)

        if self.extract_windows_of > 0:
            self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_from_beginning),
                                            num_parallel_calls=self.num_parallel_calls)

    def tf_data_to_model(self):
        # Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
        self.tf_data = self.tf_data.map(functools.partial(self.__to_model_inputs), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=self.tf_data.output_shapes)
        self.tf_data = self.tf_data.prefetch(2)

    def __pp_get_windows_from_beginning(self, sample):
        sample["poses"] = sample["poses"][0:self.extract_windows_of, :]
        sample["shape"] = tf.shape(sample["poses"])
        sample["euler_targets"] = sample["euler_targets"][0:self.extract_windows_of, :]
        sample["euler_shape"] = tf.shape(sample["euler_targets"])
        return sample

    def __to_model_inputs(self, tf_sample_dict):
        """
        Transforms a TFRecord sample into a more general sample representation where we use global keys to represent
        the required fields by the models.
        Args:
            tf_sample_dict:
        Returns:
        """
        model_sample = dict()
        model_sample[C.BATCH_SEQ_LEN] = tf_sample_dict["shape"][0]
        model_sample[C.BATCH_INPUT] = tf_sample_dict["poses"]
        model_sample[C.BATCH_TARGET] = tf_sample_dict["poses"]
        model_sample[C.BATCH_ID] = tf_sample_dict["sample_id"]
        model_sample["euler_targets"] = tf_sample_dict["euler_targets"]
        return model_sample

    def __parse_single_tfexample_fn(self, proto):
        feature_to_type = {
            "file_id": tf.FixedLenFeature([], dtype=tf.string),
            "db_name": tf.FixedLenFeature([], dtype=tf.string),
            "pose_shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "poses": tf.VarLenFeature(dtype=tf.float32),
            "euler_shape": tf.FixedLenFeature([2], dtype=tf.int64),
            "euler_targets": tf.VarLenFeature(dtype=tf.float32)}

        parsed_features = tf.parse_single_example(proto, feature_to_type)
        parsed_features["shape"] = parsed_features["pose_shape"]
        parsed_features["poses"] = tf.reshape(tf.sparse.to_dense(parsed_features["poses"]), parsed_features["shape"])
        parsed_features["euler_targets"] = tf.reshape(tf.sparse.to_dense(parsed_features["euler_targets"]), parsed_features["euler_shape"])
        parsed_features["sample_id"] = parsed_features["file_id"]
        return parsed_features


if __name__ == '__main__':

    def log_stats(stats, tag="Online"):
        print("[{2}] mean: {0}, std: {1}".format(stats["mean_all"], stats["var_all"], tag))
        print("[{2}] mean channel: {0}, std channel: {1}".format(stats["mean_channel"], stats["var_channel"], tag))
        print("[{2}] min value: {0}, max value: {1}".format(stats["min_all"], stats["max_all"], tag))
        print("[{2}] min length: {0}, max length: {1}".format(stats["min_seq_len"], stats["max_seq_len"], tag))
        print("============")

    """
    # some tests in eager mode.
    tf.enable_eager_execution()
    tfrecord_pattern = "../data/h3.6m/tfrecords/quat/srnn_poses/amass-?????-of-?????"
    dataset = SRNNTFRecordMotionDataset(data_path=tfrecord_pattern,
                                        meta_data_path="../data/h3.6m/tfrecords/quat/training/stats.npz",
                                        batch_size=32,
                                        shuffle=False,
                                        extract_windows_of=120,
                                        extract_random_windows=False)
    stats = dataset.meta_data
    train_iterator = dataset.get_iterator()
    sample = train_iterator.get_next()
    import time
    start_time = time.perf_counter()
    i = 0
    for batch in train_iterator:
        i += 1
        print(i, batch[C.BATCH_INPUT].shape)
    print("Elapsed time {:.3f}".format(time.perf_counter() - start_time))
    """

    # some tests in eager mode.
    tf.enable_eager_execution()

    tfrecord_pattern = "../data/amass/tfrecords/quat/test/amass-?????-of-?????"
    dataset = TFRecordMotionDataset(data_path=tfrecord_pattern,
                                    meta_data_path="../data/amass/tfrecords/quat/training/stats.npz",
                                    batch_size=32,
                                    shuffle=False,
                                    extract_windows_of=0,
                                    extract_random_windows=False)
    train_iterator = dataset.get_iterator()
    labels = []
    try:
        for batch in train_iterator:
            # batch = next(train_iterator)
            batch_ids = [s.numpy().decode("utf-8") for s in batch[C.BATCH_ID]]
            labels.extend(batch_ids)
    except tf.errors.OutOfRangeError:
        print("Done")

    print(len(labels))
    for l in labels:
        if l.startswith("ACCAD"):
            print(l)

    """
    train_iterator = dataset.get_iterator()
    sample = train_iterator.get_next()
    import time
    start_time = time.perf_counter()
    i = 0
    for batch in train_iterator:
        i += 1
        print(i, batch[C.BATCH_INPUT].shape)
    print("Elapsed time {:.3f}".format(time.perf_counter()-start_time))
    """