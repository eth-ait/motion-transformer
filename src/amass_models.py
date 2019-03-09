from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import rnn_cell_extensions  # my extensions of the tf repos

# ETH imports
from constants import Constants as C
from tf_model_utils import get_activation_fn
from tf_models import LatentLayer


class BaseModel(object):
    def __init__(self, config, data_pl, mode, reuse, dtype, **kwargs):
        self.config = config
        self.data_placeholders = data_pl
        self.mode = mode
        self.reuse = reuse
        self.dtype = dtype
        self.source_seq_len = config["source_seq_len"]
        self.target_seq_len = config["target_seq_len"]
        self.batch_size = config["batch_size"]
        self.autoregressive_input = config["autoregressive_input"]
        self.residual_velocities = config["residual_velocities"]
        self.angle_loss_type = config["angle_loss_type"]
        self.joint_prediction_model = config["joint_prediction_model"]
        self.grad_clip_by_norm = config["grad_clip_by_norm"]
        self.loss_on_encoder_outputs = config['loss_on_encoder_outputs']
        self.output_layer_config = config.get('output_layer', dict())
        self.activation_fn = get_activation_fn(self.output_layer_config.get('activation_fn', None))

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN

        self.data_inputs = data_pl[C.BATCH_INPUT]
        self.data_targets = data_pl[C.BATCH_TARGET]
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]

        # Defines how to employ structured latent variables to make predictions.
        # Options are
        # (1) "plain": latent samples correspond to joint predictions. dimensions must meet.
        # (2) "separate_joints": each latent variable is transformed into a joint prediction by using separate networks.
        # (3) "fk_joints": latent samples on the forward kinematic chain are concatenated and used as in (2).
        self.joint_prediction_model = config.get('joint_prediction_model', "plain")

        # Set by the child model class.
        self.outputs_tensor = None  # Tensor of predicted frames.
        self.outputs = None  # List of predicted frames.
        self.prediction_targets = None  # Targets in pose loss term.
        self.prediction_inputs = None  # Inputs that are used to make predictions.
        self.prediction_representation = None  # Intermediate representation of the model to make predictions.
        self.loss = None  # Loss op to be used in training.
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.gradient_norms = None
        self.parameter_update = None
        self.summary_update = None

        self.loss_summary = None
        self.learning_rate_summary = None  # Only used in training mode.
        self.gradient_norm_summary = None  # Only used in training mode.

        # Hard-coded parameters.
        self.JOINT_SIZE = 9
        self.NUM_JOINTS = 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

        # [(Parent ID, Joint ID, Joint Name), (...)] where each entry in a list corresponds to the joints at the same
        # level in the joint tree.
        # TODO Need to adapt to SMPL skeleton.
        self.structure = [[(-1, 0, "Hips")],
                          [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
                          [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
                          [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"), (10, 13, "LeftShoulder"), (10, 11, "Neck")],
                          [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"), (11, 12, "Head")],
                          [(18, 19, "RightForeArm"), (14, 15, "LeftForeArm")],
                          [(19, 20, "RightHand"), (15, 16, "LeftHand")]]

        # Reorder the structure so that we can access joint information by using its index.
        self.structure_indexed = dict()
        for joint_list in self.structure:
            for joint_entry in joint_list:
                joint_id = joint_entry[1]
                self.structure_indexed[joint_id] = joint_entry

        # Setup learning rate scheduler.
        self.global_step = tf.train.get_global_step(graph=None)
        self.learning_rate_decay_type = config.get('learning_rate_decay_type')
        if self.is_training:
            if config.get('learning_rate_decay_type') == 'exponential':
                self.learning_rate = tf.train.exponential_decay(config.get('learning_rate'),
                                                                global_step=self.global_step,
                                                                decay_steps=config.get('learning_rate_decay_steps'),
                                                                decay_rate=config.get('learning_rate_decay_rate'),
                                                                staircase=True)
            elif config.get('learning_rate_decay_type') == 'piecewise':
                self.learning_rate = tf.Variable(float(config.get('learning_rate')),
                                                 trainable=False,
                                                 dtype=dtype,
                                                 name="learning_rate_op")
                self.learning_rate_scheduler = self.learning_rate.assign(self.learning_rate*config.get('learning_rate_decay_rate'))
            elif config.get('learning_rate_decay_type') == 'fixed':
                self.learning_rate = config.get('learning_rate')
            else:
                raise Exception("Invalid learning rate type")

    def build_graph(self):
        self.build_network()

    def build_network(self):
        pass

    def build_loss(self):
        if self.is_eval or not self.loss_on_encoder_outputs:
            predictions = self.outputs_tensor[:, -self.target_seq_len:, :]
            targets = self.prediction_targets[:, -self.target_seq_len:, :]
            seq_len = self.target_seq_len
        else:
            predictions = self.outputs_tensor
            targets = self.prediction_targets
            seq_len = tf.shape(self.outputs_tensor)[1]

        targets_pose = targets[:, :, :self.HUMAN_SIZE]
        predictions_pose = predictions[:, :, :self.HUMAN_SIZE]

        with tf.name_scope("loss_angles"):
            diff = targets_pose - predictions_pose
            if self.angle_loss_type == C.LOSS_POSE_ALL_MEAN:
                pose_loss = tf.reduce_mean(tf.square(diff))
                tf.summary.scalar(self.mode + "/pose_loss", pose_loss, collections=[self.mode + "/model_summary"])
                self.loss = pose_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_MEAN:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                tf.summary.scalar(self.mode + "/pose_loss", per_joint_loss, collections=[self.mode + "/model_summary"])
                self.loss = per_joint_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_SUM:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                tf.summary.scalar(self.mode + "/pose_loss", per_joint_loss, collections=[self.mode + "/model_summary"])
                self.loss = per_joint_loss
            else:
                raise Exception("Unknown angle loss.")

    def optimization_routines(self):
        pass

    def step(self, session):
        pass

    def sampled_step(self, session):
        pass

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        with tf.variable_scope('output_layer', reuse=self.reuse):
            prediction = []

            if self.joint_prediction_model == "plain":
                prediction.append(self.build_predictions(self.prediction_representation, self.HUMAN_SIZE, "all"))

            elif self.joint_prediction_model == "separate_joints":
                raise Exception("SMPL skeleton is not ready yet.")
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    prediction.append(self.build_predictions(self.prediction_representation, self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model == "fk_joints":
                raise Exception("SMPL skeleton is not ready yet.")
                def traverse_parents(tree, source_list, output_list, parent_id):
                    if parent_id >= 0:
                        output_list.append(source_list[parent_id])
                        traverse_parents(tree, source_list, output_list, tree[parent_id][0])

                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = []
                    traverse_parents(self.structure_indexed, prediction, joint_inputs, parent_joint_idx)
                    joint_inputs.append(self.prediction_representation)
                    prediction.append(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))
            else:
                raise Exception("Prediction model not recognized.")

            pose_prediction = tf.concat(prediction, axis=-1)
            assert pose_prediction.get_shape()[-1] == self.HUMAN_SIZE, "Prediction not matching with the skeleton."

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]

            self.outputs_tensor = pose_prediction
            self.outputs = self.outputs_tensor

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
        hidden_size = self.output_layer_config.get('size', 0)
        num_hidden_layers = self.output_layer_config.get('num_layers', 0)

        current_layer = inputs
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer, units=hidden_size, activation=self.activation_fn)

        with tf.variable_scope('out_dense_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
            prediction = tf.layers.dense(inputs=current_layer, units=output_size, activation=None)
        return prediction

    def summary_routines(self):
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        self.loss_summary = tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        # Keep track of the learning rate
        if self.is_training:
            self.learning_rate_summary = tf.summary.scalar(self.mode+"/learning_rate", self.learning_rate, collections=[self.mode+"/model_summary"])
            self.gradient_norm_summary = tf.summary.scalar(self.mode+"/gradient_norms", self.gradient_norms, collections=[self.mode+"/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")


class Seq2SeqModel(BaseModel):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(Seq2SeqModel, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse,
                                           dtype=dtype, **kwargs)
        self.num_layers = self.config["num_layers"]
        self.architecture = self.config["architecture"]
        self.rnn_size = self.config["rnn_size"]
        self.states = None

        if self.reuse is False:
            print("Input size is %d" % self.input_size)
            print('rnn_size = {0}'.format(self.rnn_size))

        # === Transform the inputs ===
        with tf.name_scope("inputs"):
            """
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1, :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]
            """
            self.encoder_inputs = self.data_inputs[:, 0:self.source_seq_len - 1]
            self.decoder_inputs = self.data_inputs[:, self.source_seq_len - 1:-1]
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:]

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
        cell = tf.contrib.rnn.GRUCell(self.rnn_size)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_layers)])

        with tf.variable_scope("seq2seq", reuse=self.reuse):
            # === Add space decoder ===
            cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)

            # Finally, wrap everything in a residual layer if we want to model velocities
            if self.residual_velocities:
                cell = rnn_cell_extensions.ResidualWrapper(cell)

            # Define the loss function
            if self.is_eval:
                self.autoregressive_input = "sampling_based"

            loop_function = None
            if self.autoregressive_input == "sampling_based":
                def loop_function(prev, i):  # function for sampling_based loss
                    return prev
            elif self.autoregressive_input == "supervised":
                pass
            else:
                raise (ValueError, "unknown loss: %s" % self.autoregressive_input)

            # Build the RNN
            if self.architecture == "basic":
                # Basic RNN does not have a loop function in its API, so copying here.
                with tf.variable_scope("basic_rnn_seq2seq"):
                    _, enc_state = tf.contrib.rnn.static_rnn(cell, self.enc_in, dtype=tf.float32)  # Encoder
                    self.outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(self.dec_in, enc_state, cell, loop_function=loop_function)  # Decoder

            elif self.architecture == "tied":
                self.outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(self.enc_in, self.dec_in, cell, loop_function=loop_function)
            else:
                raise (ValueError, "Unknown architecture: %s" % self.architecture)

        self.outputs = tf.transpose(tf.stack(self.outputs), (1, 0, 2))
        self.outputs_tensor = self.outputs
        self.build_loss()

    def optimization_routines(self):
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # Update all the trainable parameters
        gradients = tf.gradients(self.loss, params)
        # Apply gradient clipping.
        if self.grad_clip_by_norm > 0:
            clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.grad_clip_by_norm)
        else:
            self.gradient_norms = tf.linalg.global_norm(gradients)
            clipped_gradients = gradients
        self.parameter_update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

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
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets and seed sequence.
        """
        assert self.is_eval, "Only works in sampling mode."
        prediction, targets, seed_sequence = session.run([self.outputs, self.decoder_outputs, self.encoder_inputs])
        return prediction, targets, seed_sequence

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
        decoder_input = np.concatenate([seed_sequence[:, -1:], np.zeros((batch_size, prediction_steps - 1, feature_size))], axis=1)

        prediction = session.run(self.outputs, feed_dict={self.encoder_inputs: encoder_input,
                                                          self.decoder_inputs: decoder_input})
        return prediction


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
        self.summary_ops = dict()  # A container for summary ops of this model. We use "model_summary" collection name.
        self.inputs_hidden = None
        self.receptive_field_width = None
        self.prediction_representation = None
        self.output_width = None

        with tf.name_scope("inputs"):
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
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden,
                                                       rate=self.input_layer_config.get("dropout_rate"),
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
        self.build_loss()

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

    def build_predictions(self, inputs, output_size, name):
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

        current_layer = inputs
        if out_layer_type == C.LAYER_CONV1:
            for layer_idx in range(num_hidden_layers):
                with tf.variable_scope('out_conv1d_' + name + "_" + str(layer_idx), reuse=self.reuse):
                    current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                     filters=num_filters, dilation_rate=1,
                                                     activation=self.activation_fn)

            with tf.variable_scope('out_conv1d_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                prediction = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                              filters=output_size, dilation_rate=1, activation=None)

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
                prediction, _ = Wavenet.temporal_block_ccn(input_layer=inputs,
                                                           num_filters=output_size,
                                                           kernel_size=kernel_size,
                                                           dilation=1,
                                                           activation_fn=None,
                                                           num_extra_conv=0,
                                                           use_gate=self.use_gate,
                                                           use_residual=self.use_residual,
                                                           zero_padding=True)
        else:
            raise Exception("Layer type not recognized.")
        return prediction

    def optimization_routines(self):
        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # Gradient clipping.
            gradients = tf.gradients(self.loss, params)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.config.get('grad_clip_by_norm'))
            else:
                self.gradient_norms = tf.global_norm(gradients)

            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=self.global_step)

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
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets and seed sequence.
        """
        dataset_sample = session.run(self.data_placeholders)[C.BATCH_INPUT]
        targets = dataset_sample[:, self.source_seq_len:]

        seed_sequence = dataset_sample[:, :self.source_seq_len]
        prediction = self.sample(session=session,
                                 seed_sequence=seed_sequence,
                                 prediction_steps=self.target_seq_len)
        return prediction, targets, seed_sequence

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

            model_outputs = session.run(self.outputs_tensor, feed_dict={self.data_inputs: model_inputs})

            predictions.append(model_outputs[:, -1:, :])
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
            padded_input_layer = tf.pad(input_layer, tf.constant([(0, 0,), (1, 0), (0, 0)])*padding_steps,
                                        mode='CONSTANT')
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
        self.kld_weight_summary = None

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
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden,
                                                       rate=self.input_layer_config.get("dropout_rate"),
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
        self.build_loss()

    def build_loss(self):
        super(STCN, self).build_loss()

        # KLD Loss.
        if self.is_training:
            loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.prediction_seq_len, dtype=tf.float32), -1)
            latent_loss_dict = self.latent_layer.build_loss(loss_mask, tf.reduce_mean)
            for loss_key, loss_op in latent_loss_dict.items():
                self.loss += loss_op
                self.summary_ops[loss_key] = tf.summary.scalar(str(loss_key), loss_op, collections=[self.mode+"/model_summary"])

    def summary_routines(self):
        self.kld_weight_summary = tf.summary.scalar(self.mode + "/kld_weight",
                                                    self.latent_layer.kld_weight,
                                                    collections=[self.mode + "/model_summary"])
        super(STCN, self).summary_routines()