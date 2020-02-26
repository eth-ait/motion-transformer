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
# DATA_DIR = '/media/eaksan/Warehouse-SSD2/Projects/motion-modelling/data/h3.6m/tfrecords'
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

"""
H3.6M ROTMAT
Training Entropy:  0.6879303034898961
Validation Entropy:  0.6914626436299748
Train -> Valid KLD:  0.00019910731667157326
Valid -> Train KLD:  0.00020651284232743624

H3.6M AA
Training Entropy:  1.030456081442283
Validation Entropy:  1.0256683212345559
Train -> Valid KLD:  0.000935693467787958
Valid -> Train KLD:  0.0009007654765174061

H3.6M QUAT
Training Entropy:  0.7616054139405295
Validation Entropy:  0.759497232470289
Train -> Valid KLD:  0.0006352943544436511
Valid -> Train KLD:  0.0006281378823478356

###########################33

AMASS ROTMAT
Training Entropy:  0.3190181311324122
Validation Entropy:  0.20668640466683785
Train -> Valid KLD:  0.016662914189846546
Valid -> Train KLD:  0.00914850194890248

AMASS AA
Training Entropy:  0.6378890617154102
Validation Entropy:  0.25937128132463455
Train -> Valid KLD:  0.1251075505438437
Valid -> Train KLD:  0.05372268695675013

AMASS QUAT
Training Entropy:  0.37272389837598313
Validation Entropy:  0.19645105700035953
Train -> Valid KLD:  0.03156473215301844
Valid -> Train KLD:  0.018747517068028532
"""