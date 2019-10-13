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
import numpy as np


class Constants(object):
    SEED = 1234
    
    # Model types.
    MODEL_ZERO_VEL = "zero_velocity"
    MODEL_RNN = "rnn"
    MODEL_SEQ2SEQ = "seq2seq"
    
    # Pre-defined colors for plots.
    RGB_COLORS = [np.array((0, 13, 53)), np.array((0, 91, 149)), np.array((171, 19, 19)), np.array((254, 207, 103)),
                  np.array((153, 104, 129)), np.array((255, 165, 120)), np.array((70, 163, 203)),
                  np.array((194, 34, 80)), np.array((63, 140, 115)), np.array((255, 119, 0))]
    # Run modes.
    TRAIN = 'training'
    TEST = 'test'
    VALID = 'validation'
    EVAL = 'evaluation'
    SAMPLE = 'sampling'

    # Data Batch
    BATCH_SEQ_LEN = "seq_len"
    BATCH_INPUT = "inputs"
    BATCH_TARGET = "targets"
    BATCH_ID = "id"

    # Optimization
    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_SGD = "sgd"

    # RNN cells
    GRU = 'gru'
    LSTM = 'lstm'
    BLSTM = 'blstm'
    LayerNormLSTM = 'LayerNormBasicLSTMCell'

    # Activation functions
    RELU = 'relu'
    ELU = 'elu'
    SIGMOID = 'sigmoid'
    SOFTPLUS = 'softplus'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LRELU = 'lrelu'
    CLRELU = 'clrelu'  # Clamped leaky relu.

    # Reduce function types
    R_MEAN_STEP = 'mean_step_loss'  # Take average of average step loss per sample over batch. Uses sequence length.
    R_MEAN_SEQUENCE = 'mean_sequence_loss'  # Take average of sequence loss (summation of all steps) over batch. Uses sequence length.
    R_MEAN = 'mean'  # Take mean of the whole tensor.
    R_SUM = 'sum'  # Take mean of the whole tensor.
    B_MEAN_STEP = 'batch_mean_step_loss'  # Keep the loss per sample. Uses sequence length.
    R_IDENTITY = 'identity'

    # Data representations
    ROT_MATRIX = "rotmat"
    ANGLE_AXIS = "aa"
    POSITIONAL = "pos"
    QUATERNION = "quat"

    LAYER_FC = "fc"
    LAYER_RNN = "rnn"

    LOSS_POSE_ALL_MEAN = "all_mean"
    LOSS_POSE_JOINT_SUM = "joint_sum"

    LOSS_ACTION_CENT = "cross_entropy"
    LOSS_ACTION_L2 = "l2"
    LOSS_ACTION_NONE = "none"

    METRIC_POSITIONAL = "positional"
    METRIC_JOINT_ANGLE = "joint_angle"
    METRIC_PCK = "pck"
    METRIC_EULER_ANGLE = "euler"
    
    # @ 60 Hz, in ms: 50, 100, 200, 300, 400, 600, 800, 1000
    METRIC_TARGET_LENGTHS_AMASS = [3, 6, 12, 18, 24, 36, 48, 60]
    METRIC_TARGET_LENGTHS_H36M = [4, 8, 16, 20, 28, 50]  # @ 50 Hz, in ms: [80, 160, 320, 400, 560, 1000]
    METRIC_TARGET_LENGTHS_H36M_25FPS = [2, 4, 8, 10, 14, 25]  # @ 25 Hz, in ms: [80, 160, 320, 400, 560, 1000]
    METRIC_PCK_THRESHS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    # Types of data windows.
    DATA_WINDOW_BEGINNING = "from_beginning"
    DATA_WINDOW_CENTER = "from_center"
    DATA_WINDOW_RANDOM = "random"

