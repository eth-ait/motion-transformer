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


This script shows how to read and convert our data in tfrecord format into numpy.

If you wish, you can store the same data in numpy for your own purpose.
Similarly, you can use our TFRecordMotionDataset class to read, preprocess, normalize the data and then create batches
to train a pytorch model.
"""
import os
import numpy as np
import tensorflow as tf
from spl.data.amass_tf import TFRecordMotionDataset
tf.enable_eager_execution()

# Here we load full-length test dataset in rotation matrix format. You can set the path manually.
DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "test_dynamic", "amass-?????-of-?????")
META_DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "training", "stats.npz")

# Create dataset object.
tf_data = TFRecordMotionDataset(data_path=DATA_PATH,
                                meta_data_path=META_DATA_PATH,
                                batch_size=1,
                                shuffle=False,
                                extract_windows_of=0,
                                window_type=None,
                                num_parallel_calls=4,
                                normalize=False)
data_iter_ = tf_data.get_iterator()

# Run the iterator in TF eager mode.
# Each batch consists of a dictionary with keys "seq_len", "inputs", "targets" and "id". Note that both "inputs" and
# "targets" are the same as we do the shifting in the model code.

np_data = []
np_ids = []
np_seq_len = []
try:
    for batch in data_iter_:
        np_data.append(batch["inputs"].numpy()[0])
        np_ids.append(batch["id"].numpy()[0].decode("utf-8"))
        np_seq_len.append(batch["seq_len"].numpy()[0])
except tf.errors.OutOfRangeError:
    np_ids = np.concatenate(np_ids)
    np_seq_len = np.concatenate(np_seq_len)

assert len(np_ids) == len(np_data)
assert np_seq_len[0] == len(np_data[0])
