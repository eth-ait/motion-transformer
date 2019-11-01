"""Sequence-to-sequence model of Martinez et al. (https://arxiv.org/pdf/1705.02445.pdf)
It is taken from https://github.com/una-dinosauria/human-motion-prediction and slightly modified. Code was published
under MIT license:

Copyright (c) 2016 Julieta Martinez, Javier Romero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""
import copy
import numpy as np
import tensorflow as tf
import spl.util.rnn_cell_extensions as rnn_cell_extensions
from spl.model.base_model import BaseModel
from spl.model.spl import SPL
from common.constants import Constants as C


class Seq2SeqModel(BaseModel):
    """Sequence-to-sequence model for human motion prediction"""
    
    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 **kwargs):
        super(Seq2SeqModel, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse, **kwargs)
        self.num_layers = self.config["cell_layers"]
        self.rnn_size = self.config["cell_size"]
        self.input_layer_size = self.config.get("input_hidden_layers", None)
        self.architecture = self.config["architecture"]
        self.autoregressive_input = config["autoregressive_input"]  # sampling_based or supervised
        self.states = None
        
        if self.reuse is False:
            print("Input size is %d" % self.input_size)
            print('rnn_size = {0}'.format(self.rnn_size))
        
        # === Transform the inputs ===
        with tf.name_scope("inputs"):
            self.encoder_inputs = self.data_inputs[:, 0:self.source_seq_len - 1]
            self.decoder_inputs = self.data_inputs[:, self.source_seq_len - 1:-1]
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:self.source_seq_len+self.target_seq_len]
            
            enc_in = tf.transpose(self.encoder_inputs, [1, 0, 2])
            dec_in = tf.transpose(self.decoder_inputs, [1, 0, 2])
            dec_out = tf.transpose(self.decoder_outputs, [1, 0, 2])
            
            enc_in = tf.reshape(enc_in, [-1, self.input_size])
            dec_in = tf.reshape(dec_in, [-1, self.input_size])
            dec_out = tf.reshape(dec_out, [-1, self.input_size])
            
            self.enc_in = tf.split(enc_in, self.source_seq_len - 1, axis=0)
            self.dec_in = tf.split(dec_in, self.target_seq_len, axis=0)
            self.dec_out = tf.split(dec_out, self.target_seq_len, axis=0)
            self.prediction_inputs = self.decoder_inputs
            self.prediction_targets = self.decoder_outputs
    
    def build_network(self):
        # === Create the RNN that will keep the state ===
        if self.config['cell_type'] == C.GRU:
            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
        elif self.config['cell_type'] == C.LSTM:
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        else:
            raise Exception("Cell not found.")

        drop_rate = self.config.get("input_dropout_rate", 0)
        if drop_rate is not None:
            cell = rnn_cell_extensions.InputDropoutWrapper(cell, self.is_training, drop_rate)
        
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_layers)])
        
        with tf.variable_scope("seq2seq", reuse=self.reuse):
            # === Add space decoder ===
            if self.joint_prediction_layer == "plain":
                cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)
            else:
                spl_sparse = True if self.joint_prediction_layer == "spl_sparse" else False
                sp_layer = SPL(hidden_layers=self.config["output_hidden_layers"],
                               hidden_units=self.config["output_hidden_size"],
                               joint_size=self.JOINT_SIZE,
                               sparse=spl_sparse,
                               use_h36m=False,
                               reuse=self.reuse)
                cell = rnn_cell_extensions.SPLWrapper(cell=cell, spl=sp_layer, human_size=self.HUMAN_SIZE)
            
            # Add an input layer the residual connection
            if self.input_layer_size is not None and self.input_layer_size > 0:
                cell = rnn_cell_extensions.InputEncoderWrapper(cell, self.input_layer_size, reuse=self.reuse)
            
            # Finally, wrap everything in a residual layer if we want to model velocities
            if self.residual_velocity:
                cell = rnn_cell_extensions.ResidualWrapper(cell)
            
            if self.is_eval:
                self.autoregressive_input = "sampling_based"
            
            loop_function = None
            if self.autoregressive_input == "sampling_based":
                def loop_function(prev, i):  # function for sampling_based loss
                    return prev
            elif self.autoregressive_input == "supervised":
                pass
            else:
                raise Exception("Unknown input type: " + self.autoregressive_input)
            
            # Build the RNN
            if self.architecture == "basic":
                # Basic RNN does not have a loop function in its API, so copying here.
                with tf.variable_scope("rnn_decoder_cell", reuse=self.reuse):
                    dec_cell = copy.deepcopy(cell)
                
                with tf.variable_scope("basic_rnn_seq2seq"):
                    _, enc_state = tf.contrib.rnn.static_rnn(cell, self.enc_in, dtype=tf.float32)  # Encoder
                    outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(self.dec_in, enc_state, dec_cell,
                                                                                 loop_function=loop_function)  # Decoder
            
            elif self.architecture == "tied":
                outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(self.enc_in, self.dec_in, cell,
                                                                                  loop_function=loop_function)
            else:
                raise Exception("Unknown architecture: " + self.architecture)
    
        return tf.transpose(tf.stack(outputs), (1, 0, 2))  # (N, seq_length, n_joints*dof)
    
    def step(self, session):
        """Run a step of the model feeding the given inputs.

        Args
          session: tensorflow session to use.
          encoder_inputs: list of numpy vectors to feed as encoder inputs.
          decoder_inputs: list of numpy vectors to feed as decoder inputs.
          decoder_outputs: list of numpy vectors that are the expected decoder outputs.
        Returns
          A triple consisting of gradient norm (or None if we did not do backward),
          mean squared error, and the outputs.
        Raises
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
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
        prediction, targets, seed_sequence, data_id = session.run([self.outputs,
                                                                   self.decoder_outputs,
                                                                   self.data_inputs[:, :self.source_seq_len],
                                                                   self.data_ids])
        return prediction, targets, seed_sequence, data_id
    
    def sample(self, session, seed_sequence, prediction_steps, **kwargs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        Args:
            session: Tensorflow session object.
            seed_sequence: (batch_size, seq_len, feature_size)
            prediction_steps: number of prediction steps.
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        
        batch_size, seed_seq_len, feature_size = seed_sequence.shape
        encoder_input = seed_sequence[:, :-1]
        decoder_input = np.concatenate(
            [seed_sequence[:, -1:], np.zeros((batch_size, prediction_steps - 1, feature_size))], axis=1)
        
        prediction = session.run(self.outputs, feed_dict={self.encoder_inputs: encoder_input,
                                                          self.decoder_inputs: decoder_input})
        return prediction

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
        config, experiment_name = super(Seq2SeqModel, cls).get_model_config(args, from_config)
        
        if from_config is None:
            config["architecture"] = args.architecture
            config["autoregressive_input"] = args.autoregressive_input
    
        experiment_name_format = "{}-{}_{}-{}_{}-{}_{}-b{}-in{}_out{}-{}_{}x{}-{}"
        dec_input = ""
        if config["autoregressive_input"] == "sampling_based":
            dec_input = "sampling"
        elif config["input_dropout_rate"] > 0:
            dec_input = "dropout"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        config["model_type"],
                                                        dec_input,
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
