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
import tensorflow as tf
import numpy as np
import os
import functools

from common.constants import Constants as C
from spl.data.base_dataset import Dataset


class TFRecordMotionDataset(Dataset):
    """
    Dataset class for AMASS dataset stored as TFRecord files.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        print("Loading motion data from {}".format(os.path.abspath(data_path)))
        # Extract a window randomly. If the sequence is shorter, ignore it.
        self.extract_windows_of = kwargs.get("extract_windows_of", 0)
        # Whether to extract windows randomly, from the beginning or the middle of the sequence.
        self.window_type = kwargs.get("window_type", True)
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
        # tf_data_opt.experimental_autotune = True

        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
        self.tf_data = self.tf_data.map(functools.partial(self.__parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)
        if self.shuffle:
            self.tf_data = self.tf_data.shuffle(self.batch_size*10)

        if self.extract_windows_of > 0:
            self.tf_data = self.tf_data.filter(functools.partial(self.__pp_filter))
            if self.window_type == C.DATA_WINDOW_BEGINNING:
                self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_beginning),
                                                num_parallel_calls=self.num_parallel_calls)
            elif self.window_type == C.DATA_WINDOW_CENTER:
                self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_middle),
                                                num_parallel_calls=self.num_parallel_calls)
            elif self.window_type == C.DATA_WINDOW_RANDOM:
                self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_random),
                                                num_parallel_calls=self.num_parallel_calls)
            else:
                raise Exception("Unknown window type.")

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
        if tf.test.is_gpu_available():
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

    def __pp_get_windows_random(self, sample):
        start = tf.random_uniform((1, 1), minval=0, maxval=tf.shape(sample["poses"])[0]-self.extract_windows_of+1, dtype=tf.int32)[0][0]
        end = tf.minimum(start+self.extract_windows_of, tf.shape(sample["poses"])[0])
        sample["poses"] = sample["poses"][start:end, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def __pp_get_windows_beginning(self, sample):
        # Extract a window from the beginning of the sequence.
        sample["poses"] = sample["poses"][0:self.extract_windows_of, :]
        sample["shape"] = tf.shape(sample["poses"])
        return sample

    def __pp_get_windows_middle(self, sample):
        # Window is located at the center of the sequence.
        seq_len = tf.shape(sample["poses"])[0]
        start = tf.maximum((seq_len//2) - (self.extract_windows_of//2), 0)
        end = start + self.extract_windows_of
        sample["poses"] = sample["poses"][start:end, :]
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
