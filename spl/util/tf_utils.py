import tensorflow as tf
from common.constants import Constants as C
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def get_activation_fn(activation=C.RELU):
    """
    Return tensorflow activation function given string name.

    Args:
        activation:
    Returns:
        TF activation function.
    """
    # Check if the activation is already callable.
    if callable(activation):
        return activation

    if activation == C.RELU:
        return tf.nn.relu
    elif activation == C.ELU:
        return tf.nn.elu
    elif activation == C.TANH:
        return tf.nn.tanh
    elif activation == C.SIGMOID:
        return tf.nn.sigmoid
    elif activation == C.SOFTPLUS:
        return tf.nn.softplus
    elif activation == C.SOFTMAX:
        return tf.nn.softmax
    elif activation == C.LRELU:
        return lambda x: tf.nn.leaky_relu(x, alpha=1./3.)
    elif activation == C.CLRELU:
        with tf.name_scope('ClampedLeakyRelu'):
            return lambda x: tf.clip_by_value(tf.nn.leaky_relu(x, alpha=1./3.), -3.0, 3.0)
    elif activation is None:
        return None
    else:
        raise Exception("Activation function is not implemented.")
    

def get_reduce_loss_func(op_type="sum_mean", seq_len=None):
    """

    Args:
        op_type: "sum_mean", "mean", "sum".
        seq_len
    Returns:
    """
    def reduce_sum_mean(loss):
        """
        Average batch loss. First calculates per sample loss by summing over the second and third dimensions and then
        takes the average.
        """
        rank = len(loss.get_shape())

        if rank == 3:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2]))
        elif rank == 2:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=[1]))
        else:
            raise Exception("Loss rank must be 2 or 3.")

    def reduce_mean_per_step(loss):
        """
        First calculates average loss per sample (loss per step), and then takes average over samples. Loss per step
        requires sequence length. If all samples have the same sequence length then this is equivalent to `mean`.
        """
        rank = len(loss.get_shape())

        # Calculate loss per step.
        if rank == 3:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1, 2])/tf.cast(seq_len, tf.float32)
        elif rank == 2:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1])/tf.cast(seq_len, tf.float32)
        else:
            raise Exception("Loss rank must be 2 or 3.")
        # Calculate average (per step) sample loss.
        return tf.reduce_mean(step_loss_per_sample)

    def batch_per_step(loss):
        """
        Calculates average loss per sample (loss per step), and keeps the loss for each sample. Loss per step
        requires sequence length. If all samples have the same sequence length then this is equivalent to `mean`.
        """
        rank = len(loss.get_shape())

        # Calculate loss per step.
        if rank == 3:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1, 2])/tf.cast(seq_len, tf.float32)
        elif rank == 2:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1])/tf.cast(seq_len, tf.float32)
        else:
            raise Exception("Loss rank must be 2 or 3.")
        return step_loss_per_sample

    def identity(loss):
        return loss

    if op_type == C.R_MEAN_SEQUENCE:
        return reduce_sum_mean
    elif op_type == C.R_SUM:
        return tf.reduce_sum
    elif op_type == C.R_MEAN:
        return tf.reduce_mean
    elif op_type == C.R_MEAN_STEP:
        return reduce_mean_per_step
    elif op_type == C.B_MEAN_STEP:
        return batch_per_step
    elif op_type == C.R_IDENTITY:
        return identity


class CustomMultiRNNCell(tf.nn.rnn_cell.MultiRNNCell):
    def __init__(self, cells, state_is_tuple=True, intermediate_outputs=True):
        """
        Extends tensorflow MultiRNNCell such that outputs of the intermediate cells can be accessed.
        """
        super(CustomMultiRNNCell, self).__init__(cells, state_is_tuple)
        self._intermediate_outputs = intermediate_outputs

    @property
    def output_size(self):
        if self._intermediate_outputs:
            if self._state_is_tuple:
                return tuple(cell.output_size for cell in self._cells)
            else:
                return sum([cell.output_size for cell in self._cells])
        else:
            return self._cells[-1].output_size

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        new_outputs = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError("Expected state to be a tuple of length %d, but received: %s" % (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)
                new_outputs.append(cur_inp)

        new_states = (tuple(new_states) if self._state_is_tuple else array_ops.concat(new_states, 1))
        if self._intermediate_outputs:
            new_outputs = (tuple(new_outputs) if self._state_is_tuple else array_ops.concat(new_outputs, 1))
            return new_outputs, new_states
        else:
            return cur_inp, new_states


def get_rnn_cell(**kwargs):
    """
    Creates an rnn cell object.

    Args:
        **kwargs: must contain `cell_type`, `size` and `num_layers` key-value pairs. `dropout_keep_prob` is optional.
            `dropout_keep_prob` can be a list of ratios where each cell has different dropout ratio in a stacked
            architecture. If it is a scalar value, then the whole architecture (either a single cell or stacked cell)
            has one DropoutWrapper.

    Returns:
    """
    cell_type = kwargs['cell_type']
    size = kwargs['size']
    num_layers = kwargs['num_layers']
    dropout_keep_prob = kwargs.get('dropout_keep_prob', 1.0)
    intermediate_outputs = kwargs.get('intermediate_outputs', False)

    separate_dropout = False
    if isinstance(dropout_keep_prob, list) and len(dropout_keep_prob) == num_layers:
        separate_dropout = True

    if cell_type == C.LSTM:
        rnn_cell_constructor = tf.contrib.rnn.LSTMCell
    elif cell_type == C.BLSTM:
        rnn_cell_constructor = tf.contrib.rnn.LSTMBlockCell
    elif cell_type.lower() == C.GRU:
        rnn_cell_constructor = tf.contrib.rnn.GRUCell
    elif cell_type.lower() == C.LayerNormLSTM.lower():
        rnn_cell_constructor = tf.contrib.rnn.LayerNormBasicLSTMCell
    else:
        raise Exception("Unsupported RNN Cell.")

    rnn_cells = []
    for i in range(num_layers):
        cell = rnn_cell_constructor(size)
        if separate_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_keep_prob[i],
                                                 output_keep_prob=dropout_keep_prob,
                                                 state_keep_prob=1,
                                                 dtype=tf.float32,
                                                 seed=1)
        rnn_cells.append(cell)

    if num_layers > 1:
        # cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
        cell = CustomMultiRNNCell(cells=rnn_cells, state_is_tuple=True, intermediate_outputs=intermediate_outputs)
    else:
        cell = rnn_cells[0]

    if separate_dropout and dropout_keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob,
                                             state_keep_prob=1,
                                             dtype=tf.float32,
                                             seed=1)
    return cell
