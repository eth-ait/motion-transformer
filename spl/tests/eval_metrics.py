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


This script shows how to use metrics.

We calculate the metrics between the last 60 frames of test samples and the repeated frames (i.e. zero-velocity).
"""

import os
import numpy as np
import tensorflow as tf
from spl.data.amass_tf import TFRecordMotionDataset
from metrics.motion_metrics import MetricsEngine
from visualization.fk import SMPLForwardKinematics

tf.enable_eager_execution()


DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "test", "amass-?????-of-?????")
META_DATA_PATH = os.path.join(os.environ["AMASS_DATA"], "rotmat", "training", "stats.npz")

# Create dataset object.
tf_data = TFRecordMotionDataset(data_path=DATA_PATH,
                                meta_data_path=META_DATA_PATH,
                                batch_size=64,
                                shuffle=False,
                                extract_windows_of=0,
                                window_type=None,
                                num_parallel_calls=4,
                                normalize=False)
data_iter_ = tf_data.get_iterator()

target_lengths = [3, 6, 12, 18, 24]  # Calculate error for this number of frames
pck_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]  # PCK
fk_engine = SMPLForwardKinematics()
metrics_engine = MetricsEngine(fk_engine,
                               target_lengths,
                               force_valid_rot=True,
                               pck_threshs=pck_thresholds,
                               rep="rotmat")
metrics_engine.reset()
try:
    for batch in data_iter_:
        pose_sequence = batch["inputs"].numpy()
        last_seed_pose = pose_sequence[:, 119:120]
        static_pose = np.tile(last_seed_pose, (1, 60, 1))
        target_pose = pose_sequence[:, 120:]

        metrics_engine.compute_and_aggregate(static_pose, target_pose)
except tf.errors.OutOfRangeError:
    pass
finally:
    # finalize the computation of the metrics
    final_metrics = metrics_engine.get_final_metrics()

print(metrics_engine.get_summary_string_all(final_metrics, target_lengths, pck_thresholds))
