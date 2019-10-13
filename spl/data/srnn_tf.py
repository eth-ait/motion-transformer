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
import functools

from common.constants import Constants as C
from spl.data.amass_tf import TFRecordMotionDataset


class SRNNTFRecordMotionDataset(TFRecordMotionDataset):
    """
    Dataset class for the test sequences on H3.6M defined by Jain et al. in the S-RNN paper.
    """
    def __init__(self, data_path, meta_data_path, batch_size, shuffle, **kwargs):
        super(SRNNTFRecordMotionDataset, self).__init__(data_path, meta_data_path, batch_size, shuffle, **kwargs)

    def tf_data_transformations(self):
        """
        Loads the raw data and apply preprocessing.
        This method is also used in calculation of the dataset statistics (i.e., meta-data file).
        """
        tf_data_opt = tf.data.Options()

        self.tf_data = tf.data.TFRecordDataset.list_files(self.data_path, seed=1234, shuffle=self.shuffle)
        self.tf_data = self.tf_data.with_options(tf_data_opt)
        self.tf_data = self.tf_data.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=self.num_parallel_calls, block_length=1, sloppy=self.shuffle))
        self.tf_data = self.tf_data.map(functools.partial(self.__parse_single_tfexample_fn), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.prefetch(self.batch_size*10)

        if self.extract_windows_of > 0:
            self.tf_data = self.tf_data.map(functools.partial(self.__pp_get_windows_middle),
                                            num_parallel_calls=self.num_parallel_calls)

    def tf_data_to_model(self):
        # Converts the data into the format that a model expects. Creates input, target, sequence_length, etc.
        self.tf_data = self.tf_data.map(functools.partial(self.__to_model_inputs), num_parallel_calls=self.num_parallel_calls)
        self.tf_data = self.tf_data.padded_batch(self.batch_size, padded_shapes=self.tf_data.output_shapes)
        self.tf_data = self.tf_data.prefetch(2)

    def __pp_get_windows_beginning(self, sample):
        sample["poses"] = sample["poses"][0:self.extract_windows_of, :]
        sample["shape"] = tf.shape(sample["poses"])
        sample["euler_targets"] = sample["euler_targets"][0:self.extract_windows_of, :]
        sample["euler_shape"] = tf.shape(sample["euler_targets"])
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