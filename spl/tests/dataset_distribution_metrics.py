import os
import numpy as np
import tensorflow as tf

from metrics.distribution_metrics import power_spectrum
from metrics.distribution_metrics import ps_entropy
from metrics.distribution_metrics import ps_kld
from spl.data.amass_tf import TFRecordMotionDataset
from common.constants import Constants as C


tf.enable_eager_execution()


# Here we load full-length test dataset in rotation matrix format. You can set
# the path manually.
DATA_DIR = os.environ["AMASS_DATA"]
# DATA_DIR = '<path-to-data>'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "quat", "training",
                               "amass-?????-of-?????")
VALID_DATA_PATH = os.path.join(DATA_DIR, "quat", "validation",
                               "amass-?????-of-?????")
TEST_DATA_PATH = os.path.join(DATA_DIR, "quat", "test",
                              "amass-?????-of-?????")
META_DATA_PATH = os.path.join(DATA_DIR, "quat", "training", "stats.npz")

# N_JOINTS = 21
N_JOINTS = 15
FEATURE_SIZE = 4
BATCH_SIZE = 32
WINDOW_LEN = 25
N_TOTAL_SAMPLES = 3000
# Create dataset object.
train_data = TFRecordMotionDataset(data_path=TRAIN_DATA_PATH,
                                   meta_data_path=META_DATA_PATH,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   extract_windows_of=WINDOW_LEN,
                                   window_type=C.DATA_WINDOW_RANDOM,
                                   num_parallel_calls=2,
                                   normalize=False)

valid_data = TFRecordMotionDataset(data_path=VALID_DATA_PATH,
                                   meta_data_path=META_DATA_PATH,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   extract_windows_of=WINDOW_LEN,
                                   window_type=C.DATA_WINDOW_RANDOM,
                                   num_parallel_calls=2,
                                   normalize=False)


def collect_samples(dataset):
    all_samples = []
    data_iter_ = dataset.iterator
    
    i = 0
    while i < N_TOTAL_SAMPLES:
        for batch in data_iter_:
            np_batch = tf.reshape(batch["inputs"],
                                  (-1, WINDOW_LEN, N_JOINTS, FEATURE_SIZE))
            np_batch = tf.transpose(np_batch, [0, 2, 1, 3]).numpy()
            all_samples.append(np_batch)
            
            i += np_batch.shape[0]
        
        data_iter_ = dataset.tf_data.make_one_shot_iterator()
    return np.vstack(all_samples)


# Run the iterator in TF eager mode.
# Each batch consists of a dictionary with keys "seq_len", "inputs", "targets"
# and "id". Note that both "inputs" and "targets" are the same as we do the
# shifting in the model code.
train_samples = collect_samples(train_data)
valid_samples = collect_samples(valid_data)

train_ps = power_spectrum(train_samples)
train_ent = ps_entropy(train_ps)
print("Training Entropy: ", train_ent.mean())

valid_ps = power_spectrum(valid_samples)
valid_ent = ps_entropy(valid_ps)
print("Validation Entropy: ", valid_ent.mean())

print("Train -> Valid KLD: ", ps_kld(train_ps, valid_ps).mean())
print("Valid -> Train KLD: ", ps_kld(valid_ps, train_ps).mean())