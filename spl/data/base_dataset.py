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

    @classmethod
    def load_meta_data(cls, meta_data_path):
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
