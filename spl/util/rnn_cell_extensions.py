""" Extensions to TF RNN class by una_dinosaria.

It is taken from https://github.com/una-dinosauria/human-motion-prediction and SPLWrapper is added.
"""

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


class InputEncoderWrapper(RNNCell):
    """Adds dense layer to inputs of the RNN cell."""

    def __init__(self, cell, hidden_size, reuse):
        self._cell = cell
        self._dense = tf.layers.Dense(hidden_size, activation=tf.nn.relu, _reuse=reuse)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        return self._cell(self._dense(inputs), state, scope)


class SPLWrapper(RNNCell):
    """Structured pose prediction by using cell output."""

    def __init__(self, cell, spl, human_size):
        """Create a cell with with a linear encoder in space.

        Args:
          cell: RNNCell instance.
          spl: SPL (structured prediction layer) instance.
        Raises:
          TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

        self._cell = cell
        self._spl = spl
        self.human_size = human_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.human_size

    def __call__(self, inputs, state, scope=None):
        # Run the rnn as usual
        prediction_context, new_state = self._cell(inputs, state, scope)
        # Get SPL predictions.
        pose_prediction = self._spl.build(prediction_context)
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
