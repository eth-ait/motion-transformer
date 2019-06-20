import numpy as np
import tensorflow as tf

from amass_models import BaseModel
from constants import Constants as C
from tf_model_utils import get_activation_fn
from tf_models import LatentLayer


class Wavenet(BaseModel):
    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(Wavenet, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse, dtype=dtype, **kwargs)
        self.input_layer_config = config.get('input_layer', None)
        self.cnn_layer_config = config.get('cnn_layer')
        self.output_layer_config = config.get('output_layer')
        self.use_gate = self.cnn_layer_config.get('use_gating', False)
        self.use_residual = self.cnn_layer_config.get('use_residual', False)
        self.use_skip = self.cnn_layer_config.get('use_skip', False)

        # If True, at every layer the input sequence is padded with zeros at the beginning such that the output length
        # becomes equal to the input length.
        self.zero_padding = self.cnn_layer_config.get('zero_padding', False)
        self.activation_fn = get_activation_fn(self.cnn_layer_config['activation_fn'])

        # Inputs to the decoder or output layer.
        self.decoder_use_enc_skip = self.config.get('decoder_use_enc_skip', False)
        self.decoder_use_enc_last = self.config.get('decoder_use_enc_last', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.num_encoder_blocks = self.cnn_layer_config.get('num_encoder_layers')
        self.num_decoder_blocks = self.cnn_layer_config.get('num_decoder_layers')

        # List of temporal convolution layers that are used in encoder.
        self.encoder_blocks = []
        self.encoder_blocks_no_res = []
        # List of temporal convolution layers that are used in decoder.
        self.decoder_blocks = []
        self.decoder_blocks_no_res = []

        # Specific to this code:
        self.inputs_hidden = None
        self.receptive_field_width = None
        self.prediction_representation = None
        self.output_width = None

        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :-1, :]
        self.prediction_targets = self.data_inputs[:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

    def build_network(self):
        # We always pad the input sequences such that the output sequence has the same length with input sequence.
        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'],
                                                                  self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.prediction_inputs
        if self.input_dropout_rate is not None:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden,
                                                       rate=self.input_dropout_rate,
                                                       seed=self.config["seed"], training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden,
                                                                                        self.num_encoder_blocks,
                                                                                        self.reuse,
                                                                                        self.cnn_layer_config['filter_size'])

        decoder_inputs = []
        if self.decoder_use_enc_skip:
            skip_connections = [enc_layer for enc_layer in self.encoder_blocks_no_res]
            decoder_inputs.append(self.activation_fn(sum(skip_connections)))
        if self.decoder_use_enc_last:
            decoder_inputs.append(self.encoder_blocks[-1])  # Top-most convolutional layer.
        if self.decoder_use_raw_inputs:
            decoder_inputs.append(self.prediction_inputs)
        assert len(decoder_inputs) != 0, "Decoder input is not defined."

        # Build causal decoder blocks if we have any. Otherwise, we just use a number of 1x1 convolutions in
        # build_output_layer. Note that there are several input options.
        if self.num_decoder_blocks > 0:
            with tf.variable_scope("decoder", reuse=self.reuse):
                decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size",
                                                                self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer,
                                                                                            self.num_decoder_blocks,
                                                                                            self.reuse,
                                                                                            kernel_size=decoder_filter_size)
                self.prediction_representation = self.decoder_blocks[-1]
        else:
            self.prediction_representation = tf.concat(decoder_inputs, axis=-1)

        self.output_width = tf.shape(self.prediction_representation)[1]
        self.build_output_layer()

    def build_temporal_block(self, input_layer, num_layers, reuse, kernel_size=2):
        current_layer = input_layer
        temporal_blocks = []
        temporal_blocks_no_res = []
        for idx in range(num_layers):
            with tf.variable_scope('temporal_block_' + str(idx + 1), reuse=reuse):
                temp_block, temp_wo_res = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                                     num_filters=self.cnn_layer_config['num_filters'],
                                                                     kernel_size=kernel_size,
                                                                     dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                     activation_fn=self.activation_fn,
                                                                     num_extra_conv=0,
                                                                     use_gate=self.use_gate,
                                                                     use_residual=self.use_residual,
                                                                     zero_padding=self.zero_padding)
                temporal_blocks_no_res.append(temp_wo_res)
                temporal_blocks.append(temp_block)
                current_layer = temp_block

        return temporal_blocks, temporal_blocks_no_res

    def build_predictions(self, inputs, output_size, name, share=False):
        """
        Builds the output layers given the inputs. First, creates a number of hidden layers if set in the config and
        then makes the prediction without applying an activation function.

        Args:
            inputs (tf.Tensor):
            output_size (int):
            name (str):
        Returns:
            (tf.Tensor) prediction.
        """
        out_layer_type = self.output_layer_config.get('type', None)
        num_filters = self.output_layer_config.get('size', 0)
        if num_filters < 1:
            num_filters = self.cnn_layer_config['num_filters']
        num_hidden_layers = self.output_layer_config.get('num_layers', 0)

        prediction = dict()
        current_layer = inputs
        if out_layer_type == C.LAYER_CONV1:
            for layer_idx in range(num_hidden_layers):
                with tf.variable_scope('out_conv1d_' + name + "_" + str(layer_idx), reuse=self.reuse):
                    current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                     filters=num_filters, dilation_rate=1,
                                                     activation=self.activation_fn)

            with tf.variable_scope('out_conv1d_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                prediction["mu"] = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                    filters=output_size,
                                                    dilation_rate=1, activation=self.prediction_activation)
            if self.mle_normal:
                with tf.variable_scope('out_conv1d_sigma_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                    sigma = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                             filters=output_size,
                                             dilation_rate=1, activation=tf.nn.softplus)
                    prediction["sigma"] = tf.clip_by_value(sigma, 1e-4, 5.0)

        elif out_layer_type == C.LAYER_TCN:
            kernel_size = self.output_layer_config.get('filter_size', 0)
            if kernel_size < 1:
                kernel_size = self.cnn_layer_config['filter_size']
            for layer_idx in range(num_hidden_layers):
                with tf.variable_scope('out_tcn_' + name + "_" + str(layer_idx), reuse=self.reuse):
                    current_layer, _ = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                                  num_filters=num_filters,
                                                                  kernel_size=kernel_size,
                                                                  dilation=1,
                                                                  activation_fn=self.activation_fn,
                                                                  num_extra_conv=0,
                                                                  use_gate=self.use_gate,
                                                                  use_residual=self.use_residual,
                                                                  zero_padding=True)

            with tf.variable_scope('out_tcn_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                mu, _ = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                   num_filters=output_size,
                                                   kernel_size=kernel_size,
                                                   dilation=1,
                                                   activation_fn=self.prediction_activation,
                                                   num_extra_conv=0,
                                                   use_gate=self.use_gate,
                                                   use_residual=self.use_residual,
                                                   zero_padding=True)
                prediction["mu"] = mu
            if self.mle_normal:
                with tf.variable_scope('out_tcn_sigma_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                    sigma, _ = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                          num_filters=output_size,
                                                          kernel_size=kernel_size,
                                                          dilation=1,
                                                          activation_fn=None,
                                                          num_extra_conv=0,
                                                          use_gate=self.use_gate,
                                                          use_residual=self.use_residual,
                                                          zero_padding=True)
                    prediction["sigma"] = tf.clip_by_value(tf.nn.softplus(sigma), 1e-4, 5.0)
        else:
            raise Exception("Layer type not recognized.")
        return prediction

    def step(self, session):
        """
        Run a step of the model feeding the given inputs.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
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
          unique sample id.
        """
        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        targets = data_sample[:, self.source_seq_len:]

        seed_sequence = data_sample[:, :self.source_seq_len]
        prediction = self.sample(session=session,
                                 seed_sequence=seed_sequence,
                                 prediction_steps=self.target_seq_len)
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

        input_sequence = seed_sequence
        num_steps = prediction_steps
        dummy_frame = np.zeros([seed_sequence.shape[0], 1, seed_sequence.shape[2]])
        predictions = []
        for step in range(num_steps):
            end_idx = min(self.receptive_field_width, input_sequence.shape[1])
            # Insert a dummy frame since the model shifts the inputs by one step.
            model_inputs = np.concatenate([input_sequence[:, -end_idx:], dummy_frame], axis=1)
            model_outputs = session.run(self.outputs, feed_dict={self.data_inputs: model_inputs})
            prediction = model_outputs[:, -1:, :]
            # prediction = np.reshape(get_closest_rotmat(np.reshape(prediction, [-1, 3, 3])), prediction.shape)
            predictions.append(prediction)
            input_sequence = np.concatenate([input_sequence, predictions[-1]], axis=1)

        return np.concatenate(predictions, axis=1)

    @staticmethod
    def receptive_field_size(filter_size, dilation_size_list):
        return (filter_size - 1)*sum(dilation_size_list) + 1

    @staticmethod
    def causal_conv_layer(input_layer, num_filters, kernel_size, dilation, zero_padding, activation_fn):
        padded_input_layer = input_layer
        # Applies padding at the start of the sequence with (kernel_size-1)*dilation zeros.
        padding_steps = (kernel_size - 1)*dilation
        if zero_padding and padding_steps > 0:
            padded_input_layer = tf.pad(input_layer, tf.constant([(0, 0,), (1, 0), (0, 0)])*padding_steps, mode='CONSTANT')
            input_shape = input_layer.shape.as_list()
            if input_shape[1] is not None:
                input_shape[1] += padding_steps
            padded_input_layer.set_shape(input_shape)

        conv_layer = tf.layers.conv1d(inputs=padded_input_layer,
                                      filters=num_filters,
                                      kernel_size=kernel_size,
                                      strides=1,
                                      padding='valid',
                                      dilation_rate=dilation,
                                      activation=activation_fn)
        return conv_layer

    @staticmethod
    def causal_gated_layer(input_layer, kernel_size, num_filters, dilation, zero_padding):
        with tf.name_scope('filter_conv'):
            filter_op = Wavenet.causal_conv_layer(input_layer=input_layer,
                                                  num_filters=num_filters,
                                                  kernel_size=kernel_size,
                                                  dilation=dilation,
                                                  zero_padding=zero_padding,
                                                  activation_fn=tf.nn.tanh)
        with tf.name_scope('gate_conv'):
            gate_op = Wavenet.causal_conv_layer(input_layer=input_layer,
                                                num_filters=num_filters,
                                                kernel_size=kernel_size,
                                                dilation=dilation,
                                                zero_padding=zero_padding,
                                                activation_fn=tf.nn.sigmoid)
        with tf.name_scope('gating'):
            gated_dilation = gate_op*filter_op

        return gated_dilation

    @staticmethod
    def temporal_block_ccn(input_layer, num_filters, kernel_size, dilation, activation_fn, num_extra_conv=0,
                           use_gate=True, use_residual=True, zero_padding=False):
        if use_gate:
            with tf.name_scope('gated_causal_layer'):
                temp_out = Wavenet.causal_gated_layer(input_layer=input_layer,
                                                      kernel_size=kernel_size,
                                                      num_filters=num_filters,
                                                      dilation=dilation,
                                                      zero_padding=zero_padding)
        else:
            with tf.name_scope('causal_layer'):
                temp_out = Wavenet.causal_conv_layer(input_layer=input_layer,
                                                     kernel_size=kernel_size,
                                                     num_filters=num_filters,
                                                     dilation=dilation,
                                                     zero_padding=zero_padding,
                                                     activation_fn=activation_fn)
        with tf.name_scope('block_output'):
            temp_out = tf.layers.conv1d(inputs=temp_out,
                                        filters=num_filters,
                                        kernel_size=1,
                                        padding='valid',
                                        dilation_rate=1,
                                        activation=None)
        skip_out = temp_out
        if use_residual:
            with tf.name_scope('residual_layer'):
                res_layer = input_layer
                if input_layer.shape[2] != num_filters:
                    res_layer = tf.layers.conv1d(inputs=input_layer,
                                                 filters=num_filters,
                                                 kernel_size=1,
                                                 padding='valid',
                                                 dilation_rate=1,
                                                 activation=None)
                if zero_padding is False:
                    # Cut off input sequence so that it has the same width with outputs.
                    input_width_res = tf.shape(res_layer)[1] - tf.shape(temp_out)[1]
                    res_layer = tf.slice(res_layer, [0, input_width_res, 0], [-1, -1, -1])

                temp_out = temp_out + res_layer

        return temp_out, skip_out


class STCN(Wavenet):
    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(STCN, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse, dtype=dtype, **kwargs)
        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")
        self.use_future_steps_in_q = self.config.get('use_future_steps_in_q', False)
        self.bw_encoder_blocks = []
        self.bw_encoder_blocks_no_res = []
        self.latent_layer = None
        self.latent_samples = None
        self.latent_cell_loss = dict()

        # If it is in evaluation mode, model is fed with the slice [:, 0:-1, :].
        self.prediction_inputs = self.data_placeholders[C.BATCH_INPUT]  # The q and p models require all frames.
        self.prediction_targets = self.data_placeholders[C.BATCH_INPUT][:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

    def build_network(self):
        self.latent_layer = LatentLayer.get(config=self.latent_layer_config,
                                            layer_type=self.latent_layer_config["type"],
                                            mode=self.mode,
                                            reuse=self.reuse,
                                            global_step=self.global_step)

        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'],
                                                                  self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.prediction_inputs
        if self.input_dropout_rate is not None:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden,
                                                       rate=self.input_dropout_rate,
                                                       seed=self.config["seed"],
                                                       training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden,
                                                                                        self.num_encoder_blocks,
                                                                                        self.reuse,
                                                                                        self.cnn_layer_config['filter_size'])

        if self.use_future_steps_in_q:
            reuse_params_in_bw = True
            reversed_inputs = tf.manip.reverse(self.prediction_inputs, axis=[1])
            if reuse_params_in_bw:
                with tf.variable_scope("encoder", reuse=True):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs,
                                                                                                      self.num_encoder_blocks,
                                                                                                      True,
                                                                                                      self.cnn_layer_config['filter_size'])
            else:
                with tf.variable_scope("bw_encoder", reuse=self.reuse):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs,
                                                                                                      self.num_encoder_blocks,
                                                                                                      self.reuse,
                                                                                                      self.cnn_layer_config['filter_size'])

            self.bw_encoder_blocks = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks]
            self.bw_encoder_blocks_no_res = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in
                                             self.bw_encoder_blocks_no_res]

        with tf.variable_scope("latent", reuse=self.reuse):
            p_input = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks]
            if self.latent_layer_config.get('dynamic_prior', False):
                if self.use_future_steps_in_q:
                    q_input = [tf.concat([fw_enc[:, 1:], bw_enc], axis=-1) for fw_enc, bw_enc in
                               zip(self.encoder_blocks, self.bw_encoder_blocks)]
                else:
                    q_input = [enc_layer[:, 1:] for enc_layer in self.encoder_blocks]
            else:
                q_input = p_input
            self.latent_samples = self.latent_layer.build_latent_layer(q_input=q_input, p_input=p_input)
            latent_sample = tf.concat(self.latent_samples, axis=-1)

        decoder_inputs = [latent_sample]
        if self.decoder_use_enc_skip:
            skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks_no_res]
            decoder_inputs.append(self.activation_fn(sum(skip_connections)))
        if self.decoder_use_enc_last:
            decoder_inputs.append(self.encoder_blocks[-1][:, 0:-1])  # Top-most convolutional layer.
        if self.decoder_use_raw_inputs:
            decoder_inputs.append(self.prediction_inputs[:, 0:-1])

        # Build causal decoder blocks if we have any. Otherwise, we just use a number of 1x1 convolutions in
        # build_output_layer. Note that there are several input options.
        if self.num_decoder_blocks > 0:
            with tf.variable_scope("decoder", reuse=self.reuse):
                decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer,
                                                                                            self.num_decoder_blocks,
                                                                                            self.reuse,
                                                                                            kernel_size=decoder_filter_size)
                self.prediction_representation = self.decoder_blocks[-1]
        else:
            self.prediction_representation = tf.concat(decoder_inputs, axis=-1)

        self.output_width = tf.shape(self.prediction_representation)[1]
        self.build_output_layer()

    def build_loss(self):
        super(STCN, self).build_loss()

        # KLD Loss.
        if self.is_training:
            def kl_reduce_fn(x):
                return tf.reduce_mean(tf.reduce_sum(x, axis=[1, 2]))

            loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.prediction_seq_len, dtype=tf.float32), -1)
            self.latent_cell_loss = self.latent_layer.build_loss(loss_mask, kl_reduce_fn)
            for loss_key, loss_op in self.latent_cell_loss.items():
                self.loss += loss_op

    def summary_routines(self):
        for loss_key, loss_op in self.latent_cell_loss.items():
            tf.summary.scalar(str(loss_key), loss_op, collections=[self.mode + "/model_summary"])
        tf.summary.scalar(self.mode + "/kld_weight", self.latent_layer.kld_weight, collections=[self.mode + "/model_summary"])
        super(STCN, self).summary_routines()


class StructuredSTCN(STCN):
    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(StructuredSTCN, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse, dtype=dtype, **kwargs)

    def build_network(self):
        self.latent_layer = LatentLayer.get(config=self.latent_layer_config,
                                            layer_type=C.LATENT_STRUCTURED_HUMAN,
                                            mode=self.mode,
                                            reuse=self.reuse,
                                            global_step=self.global_step,
                                            structure=self.structure)

        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.prediction_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden, rate=self.input_layer_config.get("dropout_rate"), seed=self.config["seed"], training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

        if self.use_future_steps_in_q:
            reuse_params_in_bw = True
            reversed_inputs = tf.manip.reverse(self.prediction_inputs, axis=[1])
            if reuse_params_in_bw:
                with tf.variable_scope("encoder", reuse=True):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, True, self.cnn_layer_config['filter_size'])
            else:
                with tf.variable_scope("bw_encoder", reuse=self.reuse):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

            self.bw_encoder_blocks = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks]
            self.bw_encoder_blocks_no_res = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks_no_res]

        with tf.variable_scope("latent", reuse=self.reuse):
            p_input = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks]
            if self.latent_layer_config.get('dynamic_prior', False):
                if self.use_future_steps_in_q:
                    q_input = [tf.concat([fw_enc[:, 1:], bw_enc], axis=-1) for fw_enc, bw_enc in zip(self.encoder_blocks, self.bw_encoder_blocks)]
                else:
                    q_input = [enc_layer[:, 1:] for enc_layer in self.encoder_blocks]
            else:
                q_input = p_input

            self.latent_samples = self.latent_layer.build_latent_layer(q_input=q_input, p_input=p_input)

        self.output_width = tf.shape(self.latent_samples[0])[1]
        self.build_output_layer()

    def build_output_layer(self):
        """
        Builds layers to make predictions. The structured latent space has a random variable per joint.
        """
        if self.mle_normal and self.joint_prediction_model == "plain":
            raise Exception("Normal distribution doesn't work in this setup.")

        with tf.variable_scope('output_layer', reuse=self.reuse):
            for joint_key in sorted(self.structure_indexed.keys()):
                parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                joint_sample = self.latent_samples[joint_key]
                if self.joint_prediction_model == "plain":
                    self.parse_outputs(joint_sample)
                elif self.joint_prediction_model == "separate_joints":
                    self.parse_outputs(self.build_predictions(joint_sample, self.JOINT_SIZE, joint_name))
                elif self.joint_prediction_model == "fk_joints":
                    joint_inputs = [joint_sample]
                    self.traverse_parents(joint_inputs, parent_joint_idx)
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))
                else:
                    raise Exception("Prediction model not recognized.")

            self.aggregate_outputs()
            pose_prediction = self.outputs_mu

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                if self.residual_velocities_type == "plus":
                    pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]
                elif self.residual_velocities_type == "matmul":
                    # add regularizer to the predicted rotations
                    self.residual_velocities_reg = self.rot_mat_regularization(pose_prediction,
                                                                               summary_name="velocity_rot_mat_reg")
                    # now perform the multiplication
                    preds = tf.reshape(pose_prediction, [-1, 3, 3])
                    inputs = tf.reshape(self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE],
                                        [-1, 3, 3])
                    preds = tf.matmul(inputs, preds, transpose_b=True)
                    pose_prediction = tf.reshape(preds, tf.shape(pose_prediction))
                else:
                    raise ValueError("residual velocity type {} unknown".format(self.residual_velocities_type))

            # Enforce valid rotations as the very last step, this currently doesn't do anything with rotation matrices.
            # TODO(eaksan) Not sure how to handle probabilistic predictions. For now we use only the mu predictions.
            pose_prediction = self.build_valid_rot_layer(pose_prediction)
            self.outputs_mu = pose_prediction
            self.outputs = self.get_joint_prediction(joint_idx=-1)

    def build_loss(self):
        super(StructuredSTCN, self).build_loss()
