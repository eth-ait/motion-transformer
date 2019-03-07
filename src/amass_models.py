from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import rnn_cell_extensions  # my extensions of the tf repos

# ETH imports
from constants import Constants as C
from tf_model_utils import get_activation_fn


class BaseModel(object):
    def __init__(self, config, data_pl, mode, reuse, dtype, **kwargs):
        self.config = config
        self.data_pl = data_pl
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
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    prediction.append(self.build_predictions(self.prediction_representation, self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model == "fk_joints":
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

            # This code repository expects the outputs to be a list of time-steps.
            # outputs_list = tf.split(self.outputs_tensor, self.sequence_length, axis=1)
            # Select only the "decoder" predictions.
            self.outputs = [tf.squeeze(out_frame, axis=1) for out_frame in tf.split(self.outputs_tensor[:, -self.target_seq_len:], self.target_seq_len, axis=1)]

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
        # with tf.name_scope(self.mode):
        with tf.name_scope("inputs"):
            """
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1, :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]
            """
            self.encoder_inputs = self.data_pl[C.BATCH_INPUT][:, 0:self.source_seq_len-1]
            self.decoder_inputs = self.data_pl[C.BATCH_INPUT][:, self.source_seq_len-1:-1]
            self.decoder_outputs = self.data_pl[C.BATCH_INPUT][:, self.source_seq_len:]

            enc_in = tf.transpose(self.encoder_inputs, [1, 0, 2])
            dec_in = tf.transpose(self.decoder_inputs, [1, 0, 2])
            dec_out = tf.transpose(self.decoder_outputs, [1, 0, 2])

            enc_in = tf.reshape(enc_in, [-1, self.input_size])
            dec_in = tf.reshape(dec_in, [-1, self.input_size])
            dec_out = tf.reshape(dec_out, [-1, self.input_size])

            self.enc_in = tf.split(enc_in, self.source_seq_len - 1, axis=0)
            self.dec_in = tf.split(dec_in, self.target_seq_len, axis=0)
            self.dec_out = tf.split(dec_out, self.target_seq_len, axis=0)
            self.prediction_inputs = tf.stack(self.enc_in)
            self.prediction_targets = tf.stack(self.dec_out)

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

        self.outputs_tensor = tf.stack(self.outputs)
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
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."
        return self.step(session)[2]
