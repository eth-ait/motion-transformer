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


Autoregressive RNN model in its vanilla form or with our structured prediction layer (SPL).
"""

import numpy as np
import tensorflow as tf
import spl.util.tf_utils as model_utils
from spl.model.base_model import BaseModel
from common.constants import Constants as C


class RNN(BaseModel):
    """
    Autoregressive RNN.
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(RNN, self).__init__(config, data_pl, mode, reuse, **kwargs)

        self.loss_all_frames = True  # Calculate training objective both on the seed and target sequences.

        self.cell = None
        self.initial_states = None
        self.rnn_state = None  # Final state of RNN layer.

        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1
        else:
            self.sequence_length = self.target_seq_len
        
        self.prediction_inputs = self.data_inputs[:, :-1, :]
        self.prediction_targets = self.data_inputs[:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            self.tf_batch_size = tf.shape(self.prediction_inputs)[0]

    def create_cell(self):
        return model_utils.get_rnn_cell(cell_type=self.config["cell_type"],
                                        size=self.config["cell_size"],
                                        num_layers=self.config["cell_layers"],
                                        mode=self.mode,
                                        reuse=self.reuse)
    
    def build_input_layer(self, inputs_):
        current_layer = inputs_
        drop_rate = self.config.get("input_dropout_rate", 0)
        if drop_rate > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                current_layer = tf.layers.dropout(current_layer,
                                                  rate=drop_rate,
                                                  seed=self.config["seed"],
                                                  training=self.is_training)
        hidden_layers = self.config.get("input_hidden_layers", 0)
        hidden_size = self.config.get('input_hidden_size', 0)
        for layer_idx in range(hidden_layers):
            with tf.variable_scope("inp_dense_" + str(layer_idx), reuse=self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer,
                                                units=hidden_size,
                                                activation=self.activation_fn)
        return current_layer
    
    def build_network(self):
        self.cell = self.create_cell()
        self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        inputs_hidden = self.build_input_layer(self.prediction_inputs)

        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                            inputs_hidden,
                                                            sequence_length=self.prediction_seq_len,
                                                            initial_state=self.initial_states,
                                                            dtype=tf.float32)
        return self.build_prediction_layer(rnn_outputs)

    def build_loss(self):
        return super(RNN, self).build_loss()

    def step(self, session):
        """
        Run a step of the model feeding the given inputs.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
        """
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, session):
        """
        Generates a synthetic sequence by feeding the prediction at t+1. First, we get the next sample from the dataset.
        Args:
          session: Tensorflow session object.
        Returns:
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets, seed sequence and
          unique sample IDs.
        """
        assert self.is_eval, "Only works in sampling mode."

        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        targets = data_sample[:, self.source_seq_len:]

        # Get the model state by feeding the seed sequence.
        seed_sequence = data_sample[:, :self.source_seq_len]
        predictions = self.sample(session, seed_sequence, prediction_steps=self.target_seq_len)

        return predictions, targets, seed_sequence, data_id

    def sample(self, session, seed_sequence, prediction_steps, **kwargs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1. The first prediction step corresponds to the
        last input step.
        Args:
            session: Tensorflow session object.
            seed_sequence: (batch_size, seq_len, feature_size)
            prediction_steps: number of prediction steps.
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        prediction = prediction[:, -1:]
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)
    
    @classmethod
    def get_model_config(cls, args, from_config=None):
        """Given command-line arguments, creates the configuration dictionary.

        It is later passed to the models and stored in the disk.
        Args:
            args: command-line argument object.
            from_config: use an already existing config dictionary.
        Returns:
            experiment configuration (dict), experiment name (str)
        """
        config, experiment_name = super(RNN, cls).get_model_config(args, from_config)
        
        experiment_name_format = "{}-{}-{}_{}-{}_{}-b{}-in{}_out{}-{}_{}x{}-{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        config["model_type"],
                                                        config["joint_prediction_layer"],
                                                        config["output_hidden_size"],
                                                        "h36m" if config["use_h36m"] else "amass",
                                                        config["data_type"],
                                                        config["batch_size"],
                                                        config["source_seq_len"],
                                                        config["target_seq_len"],
                                                        config["cell_type"],
                                                        config["cell_layers"],
                                                        config["cell_size"],
                                                        config["loss_type"])
        return config, experiment_name
