""" Extensions to TF RNN class by una_dinosaria"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

# The import for LSTMStateTuple changes in TF >= 1.2.0
from pkg_resources import parse_version as pv

if pv(tf.__version__) >= pv('1.2.0'):
    from tensorflow.contrib.rnn import LSTMStateTuple
else:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
del pv


class ResidualWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell, error_signal_size=0, action_len=0, ignore_actions=True):
        """Create a cell with added residual connection.

        Args:
          cell: an RNNCell. The input is added to the output.
          error_signal_size: dimensionality of error feedback that is appended to the input of the cell

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self._error_signal_size = error_signal_size
        self._action_len = action_len
        self._ignore_actions = ignore_actions

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Add the residual connection ignoring the potential error signal at the end of the input
        error_signal_start = inputs.get_shape()[-1].value - self._error_signal_size
        input_pose = inputs[:, :error_signal_start]
        if self._ignore_actions and self._action_len > 0:
            pred_with_actions = tf.concat([output, tf.zeros_like(input_pose[:, -self._action_len:])], axis=-1)
            output = tf.add(pred_with_actions, input_pose)
        else:
            output = tf.add(output, input_pose)
        return output, new_state


class LinearSpaceDecoderWrapper(RNNCell):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell

        print('output_size = {0}'.format(output_size))
        print(' state_size = {0}'.format(self._cell.state_size))

        # Tuple if multi-rnn
        if isinstance(self._cell.state_size, tuple):

            # Fine if GRU...
            insize = self._cell.state_size[-1]

            # LSTMStateTuple if LSTM
            if isinstance(insize, LSTMStateTuple):
                insize = insize.h

        else:
            # Fine if not multi-rnn
            insize = self._cell.state_size

        self.w_out = tf.get_variable("proj_w_out",
                                     [insize, output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
        self.b_out = tf.get_variable("proj_b_out", [output_size],
                                     dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

        self.linear_output_size = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state, scope)

        # Apply the multiplication to everything
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_state


class StructuredOutputWrapper(RNNCell):
    """Given a structure, implements structured outputs."""

    def __init__(self, cell, structure, hidden_size, num_hidden_layers, activation_fn, joint_size, human_size, reuse, is_sparse=False):
        """Create a cell with with a linear encoder in space.

        Args:
          cell: an RNNCell. The input is passed through a linear layer.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self.prediction_activation = None
        self.activation_fn = activation_fn
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.reuse = reuse
        self.structure = structure
        self.joint_size = joint_size
        self.human_size = human_size
        self.is_sparse = is_sparse

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.human_size

    def build_predictions(self, inputs, output_size, name):
        """
        Builds dense output layers given the inputs. First, creates a number of hidden layers if set in the config and
        then makes the prediction without applying an activation function.
        Args:
            inputs (tf.Tensor):
            output_size (int):
            name (str):
        Returns:
            (tf.Tensor) prediction.
        """
        current_layer = inputs
        for layer_idx in range(self.num_hidden_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer, units=self.hidden_size, activation=self.activation_fn)

        with tf.variable_scope('out_dense_' + name + "_" + str(self.num_hidden_layers), reuse=self.reuse):
            prediction = tf.layers.dense(inputs=current_layer, units=output_size,
                                         activation=self.prediction_activation)
        return prediction

    def __call__(self, inputs, state, scope=None):
        """Use a linear layer and pass the output to the cell."""

        # Run the rnn as usual
        prediction_context, new_state = self._cell(inputs, state, scope)

        prediction = []

        def traverse_parents(tree, source_list, output_list, parent_id):
            if parent_id >= 0:
                output_list.append(source_list[parent_id])
                traverse_parents(tree, source_list, output_list, tree[parent_id][0])

        for joint_key in sorted(self.structure.keys()):
            parent_joint_idx, joint_idx, joint_name = self.structure[joint_key]
            joint_inputs = [prediction_context]
            if self.is_sparse:
                if parent_joint_idx >= 0:
                    joint_inputs.append(prediction[parent_joint_idx])
            else:
                traverse_parents(self.structure, prediction, joint_inputs, parent_joint_idx)
            prediction.append(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.joint_size, joint_name))

        # Apply the multiplication to everything
        pose_prediction = tf.concat(prediction, axis=-1)
        assert pose_prediction.get_shape()[-1] == self.human_size, "Prediction not matching with the skeleton."

        return pose_prediction, new_state


class InputDropoutWrapper(RNNCell):
    """Operator adding residual connections to a given cell."""

    def __init__(self, cell, is_training, dropout_rate=0):
        """Create a cell with added residual connection.

        Args:
          cell: an RNNCell. The input is added to the output.
          error_signal_size: dimensionality of error feedback that is appended to the input of the cell

        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self.dropout_rate = dropout_rate
        self.is_training = is_training
        self.seed = 1234

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""

        inputs_hidden = inputs
        if self.dropout_rate > 0:
            with tf.variable_scope('input_dropout'):
                inputs_hidden = tf.layers.dropout(inputs_hidden,
                                                  rate=self.dropout_rate,
                                                  seed=self.seed,
                                                  training=self.is_training)
        # Run the rnn as usual
        output, new_state = self._cell(inputs_hidden, state, scope)
        return output, new_state
