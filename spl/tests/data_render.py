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
import tensorflow as tf
from spl.data.amass_tf import TFRecordMotionDataset
from visualization.fk import SMPLForwardKinematics
from visualization.render import Visualizer
tf.enable_eager_execution()

# Here we visualize a window of 180 frames from full-length test dataset in rotation matrix format.
DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "test_dynamic", "amass-?????-of-?????")
META_DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "training", "stats.npz")
SAVE_DIR = "spl/tests/tmp_samples/"

# Create dataset object.
tf_data = TFRecordMotionDataset(data_path=DATA_PATH,
                                meta_data_path=META_DATA_PATH,
                                batch_size=1,
                                shuffle=False,
                                extract_windows_of=180,
                                window_type="from_beginning",
                                num_parallel_calls=4,
                                normalize=False)
data_iter_ = tf_data.get_iterator()
batch = data_iter_.get_next()

fk_engine = SMPLForwardKinematics()
visualizer = Visualizer(interactive=False, fk_engine=fk_engine, rep="rotmat", output_dir=SAVE_DIR, skeleton=True,
                        dense=True, to_video=True)

pose_seq = batch["inputs"].numpy()
pose_name = batch["id"].numpy()[0].decode("utf-8")

# Visualize skeleton.
try:
    visualizer.create_clip_smpl(pose_seq, pose_name)
except:
    visualizer.create_clip_skeleton(pose_seq, pose_name)
