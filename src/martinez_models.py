"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import rnn_cell_extensions  # my extensions of the tf repos

# ETH imports
from constants import Constants as C
from tf_model_utils import get_activation_fn, get_rnn_cell
from tf_models import LatentLayer
from data_utils import softmax
import tf_tr_quat


class BaseModel(object):
    def __init__(self, config, session, mode, reuse, dtype, **kwargs):
        self.config = config
        self.session = session
        self.mode = mode
        self.reuse = reuse
        self.dtype = dtype
        self.source_seq_len = config["source_seq_len"]
        self.target_seq_len = config["target_seq_len"]
        self.batch_size = config["batch_size"]
        self.number_of_actions = config["number_of_actions"]
        self.one_hot = config["one_hot"]
        self.autoregressive_input = config["autoregressive_input"]
        self.residual_velocities = config["residual_velocities"]
        self.action_loss_type = config["action_loss_type"]
        self.angle_loss_type = config["angle_loss_type"]
        self.joint_prediction_model = config["joint_prediction_model"]
        self.grad_clip_by_norm = config["grad_clip_by_norm"]
        self.loss_on_encoder_outputs = config['loss_on_encoder_outputs']
        self.output_layer_config = config.get('output_layer', dict())
        self.activation_fn = get_activation_fn(self.output_layer_config.get('activation_fn', None))
        self.rep = config["rep"]

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN

        # Defines how to employ structured latent variables to make predictions.
        # Options are
        # (1) "plain": latent samples correspond to joint predictions. dimensions must meet.
        # (2) "separate_joints": each latent variable is transformed into a joint prediction by using separate networks.
        # (3) "fk_joints": latent samples on the forward kinematic chain are concatenated and used as in (2).
        self.joint_prediction_model = config.get('joint_prediction_model', "plain")
        self.use_sparse_fk_joints = config.get('use_sparse_fk_joints', False)

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

        # Hard-coded parameters.
        self.ACTION_SIZE = self.number_of_actions  # 15
        self.JOINT_SIZE = 3 if self.rep == "aa" else 9
        self.NUM_JOINTS = 21
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE  # 159 if rot_mat and not using new preprocessing
        self.input_size = self.HUMAN_SIZE + self.ACTION_SIZE if self.one_hot else self.HUMAN_SIZE
        # self.HUMAN_SIZE = 54

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

        self.normalization_std = kwargs.get('std', None)
        self.normalization_mean = kwargs.get('mean', None)

        if self.normalization_std is not None:
            # if we are using one hot encoded vectors, pad
            self.normalization_std = np.concatenate([self.normalization_std, [1.0]*self.number_of_actions])
            self.normalization_mean = np.concatenate([self.normalization_mean, [0.0]*self.number_of_actions])

    def build_graph(self):
        self.build_network()
        self.build_loss()

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

        with tf.name_scope("action_label_loss"):
            if self.action_loss_type == C.LOSS_ACTION_CENT:
                assert targets.get_shape()[-1].value == self.HUMAN_SIZE + self.ACTION_SIZE
                targets_action = targets[:, :, self.HUMAN_SIZE:]
                predictions_action = predictions[:, :, self.HUMAN_SIZE:]

                action_loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets_action, logits=predictions_action)
                action_loss = tf.reduce_mean(action_loss)
                tf.summary.scalar(self.mode + "/action_loss", action_loss, collections=[self.mode + "/model_summary"])
                self.loss = self.loss + action_loss

            elif self.action_loss_type == C.LOSS_ACTION_L2:
                assert targets.get_shape()[-1].value == self.HUMAN_SIZE + self.ACTION_SIZE
                targets_action = targets[:, :, self.HUMAN_SIZE:]
                predictions_action = predictions[:, :, self.HUMAN_SIZE:]

                action_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_action - targets_action), axis=-1))
                tf.summary.scalar(self.mode + "/action_loss", action_loss, collections=[self.mode + "/model_summary"])
                self.loss = self.loss + action_loss

            elif self.action_loss_type == C.LOSS_ACTION_NONE:
                pass
            else:
                raise Exception("Unknown action loss.")

    def optimization_routines(self):
        pass

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
                    joint_inputs = [self.prediction_representation]
                    if self.use_sparse_fk_joints:
                        if parent_joint_idx >= 0:
                            joint_inputs.append(prediction[parent_joint_idx])
                    else:
                        traverse_parents(self.structure_indexed, prediction, joint_inputs, parent_joint_idx)
                    prediction.append(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))
            else:
                raise Exception("Prediction model not recognized.")

            pose_prediction = tf.concat(prediction, axis=-1)
            assert pose_prediction.get_shape()[-1] == self.HUMAN_SIZE, "Prediction not matching with the skeleton."

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]

            if self.one_hot:
                # Replicate the input action labels.
                if self.action_loss_type == C.LOSS_ACTION_NONE:
                    action_prediction = tf.tile(self.prediction_inputs[0:1, 0:1, -self.ACTION_SIZE:], (tf.shape(pose_prediction)[0], tf.shape(pose_prediction)[1], 1))
                else:  # self.action_loss_type in [C.LOSS_ACTION_L2 or C.LOSS_ACTION_CENT]:
                    action_prediction = self.build_predictions(self.prediction_representation, self.ACTION_SIZE, "actions")
                self.outputs_tensor = tf.concat([pose_prediction, action_prediction], axis=-1)
            else:
                self.outputs_tensor = pose_prediction

            # This code repository expects the outputs to be a list of time-steps.
            # outputs_list = tf.split(self.outputs_mu, self.sequence_length, axis=1)
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
        self.euler_summaris()

    def euler_summaris(self):
        # === variables for loss in Euler Angles -- for each action
        if self.is_eval:
            with tf.name_scope("euler_error_all_mean"):
                self.all_mean_err = tf.placeholder(tf.float32, name="all_mean_err")
                self.all_mean_err_summary = tf.summary.scalar('euler_error_all_mean_err', self.all_mean_err)

                self.all_mean_err80 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0080")
                self.all_mean_err160 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0160")
                self.all_mean_err320 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0320")
                self.all_mean_err400 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0400")
                self.all_mean_err560 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0560")
                self.all_mean_err1000 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_1000")
                self.all_mean_err80_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0080', self.all_mean_err80)
                self.all_mean_err160_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0160', self.all_mean_err160)
                self.all_mean_err320_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0320', self.all_mean_err320)
                self.all_mean_err400_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0400', self.all_mean_err400)
                self.all_mean_err560_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0560', self.all_mean_err560)
                self.all_mean_err1000_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_1000', self.all_mean_err1000)

            with tf.name_scope("euler_error_walking"):
                self.walking_err80 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0080")
                self.walking_err160 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0160")
                self.walking_err320 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0320")
                self.walking_err400 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0400")
                self.walking_err560 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0560")
                self.walking_err1000 = tf.placeholder(tf.float32, name="walking_srnn_seeds_1000")
                self.walking_err80_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0080', self.walking_err80)
                self.walking_err160_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0160', self.walking_err160)
                self.walking_err320_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0320', self.walking_err320)
                self.walking_err400_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0400', self.walking_err400)
                self.walking_err560_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0560', self.walking_err560)
                self.walking_err1000_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_1000', self.walking_err1000)

            with tf.name_scope("euler_error_eating"):
                self.eating_err80 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0080")
                self.eating_err160 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0160")
                self.eating_err320 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0320")
                self.eating_err400 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0400")
                self.eating_err560 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0560")
                self.eating_err1000 = tf.placeholder(tf.float32, name="eating_srnn_seeds_1000")
                self.eating_err80_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0080', self.eating_err80)
                self.eating_err160_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0160', self.eating_err160)
                self.eating_err320_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0320', self.eating_err320)
                self.eating_err400_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0400', self.eating_err400)
                self.eating_err560_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0560', self.eating_err560)
                self.eating_err1000_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_1000', self.eating_err1000)

            with tf.name_scope("euler_error_smoking"):
                self.smoking_err80 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0080")
                self.smoking_err160 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0160")
                self.smoking_err320 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0320")
                self.smoking_err400 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0400")
                self.smoking_err560 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0560")
                self.smoking_err1000 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_1000")
                self.smoking_err80_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0080', self.smoking_err80)
                self.smoking_err160_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0160', self.smoking_err160)
                self.smoking_err320_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0320', self.smoking_err320)
                self.smoking_err400_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0400', self.smoking_err400)
                self.smoking_err560_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0560', self.smoking_err560)
                self.smoking_err1000_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_1000', self.smoking_err1000)

            with tf.name_scope("euler_error_discussion"):
                self.discussion_err80 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0080")
                self.discussion_err160 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0160")
                self.discussion_err320 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0320")
                self.discussion_err400 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0400")
                self.discussion_err560 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0560")
                self.discussion_err1000 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_1000")
                self.discussion_err80_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0080', self.discussion_err80)
                self.discussion_err160_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0160', self.discussion_err160)
                self.discussion_err320_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0320', self.discussion_err320)
                self.discussion_err400_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0400', self.discussion_err400)
                self.discussion_err560_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0560', self.discussion_err560)
                self.discussion_err1000_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_1000', self.discussion_err1000)

            with tf.name_scope("euler_error_directions"):
                self.directions_err80 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0080")
                self.directions_err160 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0160")
                self.directions_err320 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0320")
                self.directions_err400 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0400")
                self.directions_err560 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0560")
                self.directions_err1000 = tf.placeholder(tf.float32, name="directions_srnn_seeds_1000")
                self.directions_err80_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0080', self.directions_err80)
                self.directions_err160_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0160', self.directions_err160)
                self.directions_err320_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0320', self.directions_err320)
                self.directions_err400_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0400', self.directions_err400)
                self.directions_err560_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0560', self.directions_err560)
                self.directions_err1000_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_1000', self.directions_err1000)

            with tf.name_scope("euler_error_greeting"):
                self.greeting_err80 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0080")
                self.greeting_err160 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0160")
                self.greeting_err320 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0320")
                self.greeting_err400 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0400")
                self.greeting_err560 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0560")
                self.greeting_err1000 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_1000")
                self.greeting_err80_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0080', self.greeting_err80)
                self.greeting_err160_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0160', self.greeting_err160)
                self.greeting_err320_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0320', self.greeting_err320)
                self.greeting_err400_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0400', self.greeting_err400)
                self.greeting_err560_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0560', self.greeting_err560)
                self.greeting_err1000_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_1000', self.greeting_err1000)

            with tf.name_scope("euler_error_phoning"):
                self.phoning_err80 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0080")
                self.phoning_err160 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0160")
                self.phoning_err320 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0320")
                self.phoning_err400 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0400")
                self.phoning_err560 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0560")
                self.phoning_err1000 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_1000")
                self.phoning_err80_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0080', self.phoning_err80)
                self.phoning_err160_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0160', self.phoning_err160)
                self.phoning_err320_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0320', self.phoning_err320)
                self.phoning_err400_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0400', self.phoning_err400)
                self.phoning_err560_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0560', self.phoning_err560)
                self.phoning_err1000_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_1000', self.phoning_err1000)

            with tf.name_scope("euler_error_posing"):
                self.posing_err80 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0080")
                self.posing_err160 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0160")
                self.posing_err320 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0320")
                self.posing_err400 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0400")
                self.posing_err560 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0560")
                self.posing_err1000 = tf.placeholder(tf.float32, name="posing_srnn_seeds_1000")
                self.posing_err80_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0080', self.posing_err80)
                self.posing_err160_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0160', self.posing_err160)
                self.posing_err320_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0320', self.posing_err320)
                self.posing_err400_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0400', self.posing_err400)
                self.posing_err560_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0560', self.posing_err560)
                self.posing_err1000_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_1000', self.posing_err1000)

            with tf.name_scope("euler_error_purchases"):
                self.purchases_err80 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0080")
                self.purchases_err160 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0160")
                self.purchases_err320 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0320")
                self.purchases_err400 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0400")
                self.purchases_err560 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0560")
                self.purchases_err1000 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_1000")
                self.purchases_err80_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0080', self.purchases_err80)
                self.purchases_err160_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0160', self.purchases_err160)
                self.purchases_err320_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0320', self.purchases_err320)
                self.purchases_err400_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0400', self.purchases_err400)
                self.purchases_err560_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0560', self.purchases_err560)
                self.purchases_err1000_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_1000', self.purchases_err1000)

            with tf.name_scope("euler_error_sitting"):
                self.sitting_err80 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0080")
                self.sitting_err160 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0160")
                self.sitting_err320 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0320")
                self.sitting_err400 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0400")
                self.sitting_err560 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0560")
                self.sitting_err1000 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_1000")
                self.sitting_err80_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0080', self.sitting_err80)
                self.sitting_err160_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0160', self.sitting_err160)
                self.sitting_err320_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0320', self.sitting_err320)
                self.sitting_err400_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0400', self.sitting_err400)
                self.sitting_err560_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0560', self.sitting_err560)
                self.sitting_err1000_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_1000', self.sitting_err1000)

            with tf.name_scope("euler_error_sittingdown"):
                self.sittingdown_err80 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0080")
                self.sittingdown_err160 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0160")
                self.sittingdown_err320 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0320")
                self.sittingdown_err400 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0400")
                self.sittingdown_err560 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0560")
                self.sittingdown_err1000 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_1000")
                self.sittingdown_err80_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0080', self.sittingdown_err80)
                self.sittingdown_err160_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0160', self.sittingdown_err160)
                self.sittingdown_err320_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0320', self.sittingdown_err320)
                self.sittingdown_err400_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0400', self.sittingdown_err400)
                self.sittingdown_err560_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0560', self.sittingdown_err560)
                self.sittingdown_err1000_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_1000', self.sittingdown_err1000)

            with tf.name_scope("euler_error_takingphoto"):
                self.takingphoto_err80 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0080")
                self.takingphoto_err160 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0160")
                self.takingphoto_err320 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0320")
                self.takingphoto_err400 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0400")
                self.takingphoto_err560 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0560")
                self.takingphoto_err1000 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_1000")
                self.takingphoto_err80_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0080', self.takingphoto_err80)
                self.takingphoto_err160_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0160', self.takingphoto_err160)
                self.takingphoto_err320_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0320', self.takingphoto_err320)
                self.takingphoto_err400_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0400', self.takingphoto_err400)
                self.takingphoto_err560_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0560', self.takingphoto_err560)
                self.takingphoto_err1000_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_1000', self.takingphoto_err1000)

            with tf.name_scope("euler_error_waiting"):
                self.waiting_err80 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0080")
                self.waiting_err160 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0160")
                self.waiting_err320 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0320")
                self.waiting_err400 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0400")
                self.waiting_err560 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0560")
                self.waiting_err1000 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_1000")
                self.waiting_err80_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0080', self.waiting_err80)
                self.waiting_err160_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0160', self.waiting_err160)
                self.waiting_err320_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0320', self.waiting_err320)
                self.waiting_err400_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0400', self.waiting_err400)
                self.waiting_err560_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0560', self.waiting_err560)
                self.waiting_err1000_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_1000', self.waiting_err1000)

            with tf.name_scope("euler_error_walkingdog"):
                self.walkingdog_err80 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0080")
                self.walkingdog_err160 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0160")
                self.walkingdog_err320 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0320")
                self.walkingdog_err400 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0400")
                self.walkingdog_err560 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0560")
                self.walkingdog_err1000 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_1000")
                self.walkingdog_err80_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0080', self.walkingdog_err80)
                self.walkingdog_err160_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0160', self.walkingdog_err160)
                self.walkingdog_err320_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0320', self.walkingdog_err320)
                self.walkingdog_err400_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0400', self.walkingdog_err400)
                self.walkingdog_err560_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0560', self.walkingdog_err560)
                self.walkingdog_err1000_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_1000', self.walkingdog_err1000)

            with tf.name_scope("euler_error_walkingtogether"):
                self.walkingtogether_err80 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0080")
                self.walkingtogether_err160 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0160")
                self.walkingtogether_err320 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0320")
                self.walkingtogether_err400 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0400")
                self.walkingtogether_err560 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0560")
                self.walkingtogether_err1000 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_1000")
                self.walkingtogether_err80_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0080', self.walkingtogether_err80)
                self.walkingtogether_err160_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0160', self.walkingtogether_err160)
                self.walkingtogether_err320_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0320', self.walkingtogether_err320)
                self.walkingtogether_err400_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0400', self.walkingtogether_err400)
                self.walkingtogether_err560_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0560', self.walkingtogether_err560)
                self.walkingtogether_err1000_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_1000', self.walkingtogether_err1000)

    def get_batch(self, data, actions):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys = list(data.keys())
        chosen_keys = np.random.choice(len(all_keys), self.batch_size)

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in range(self.batch_size):
            the_key = all_keys[chosen_keys[i]]
            # Get the number of frames
            n, _ = data[the_key].shape
            # Sample somewhere in the middle
            idx = np.random.randint(16, n - total_frames)
            # Select the data around the sampled points
            data_sel = data[the_key][idx:idx + total_frames, :]
            # Add the data
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1, :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]
        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn(self, data, action):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """
        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        seed = 1234567890
        rng = np.random.RandomState(seed)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        t1 = data[(subject, action, subaction1, 'even')].shape[0]
        t2 = data[(subject, action, subaction2, 'even')].shape[0]
        prefix, suffix = 50, 100

        idx = list()
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        return idx

    def get_batch_srnn(self, data, action):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                   "posing", "purchases", "sitting", "sittingdown", "smoking",
                   "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if action not in actions:
            raise ValueError("Unrecognized action {0}".format(action))

        frames = dict()
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8  # we always evaluate 8 seeds
        subject = 5  # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [(action, (i%2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence, 'even')]
            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1), :]
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]
        return encoder_inputs, decoder_inputs, decoder_outputs


class Seq2SeqModel(BaseModel):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        """Create the model.

        Related parameters in config :
          architecture: [basic, tied] whether to tie the decoder and decoder.
          source_seq_len: length of the input sequence.
          target_seq_len: length of the target sequence.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          grad_clip_by_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_rate: decay learning rate by this much when needed.
          autoregressive_input: [supervised, sampling_based]. Whether to use ground truth in
            each timestep to compute the loss after decoding, or to feed back the
            prediction from the previous time-step.
          number_of_actions: number of classes we have.
          one_hot: whether to use one_hot encoding during train/test (sup models).
          residual_velocities: whether to use a residual connection that models velocities.
          dtype: the data type to use to store internal variables.
        """
        super(Seq2SeqModel, self).__init__(config=config, session=session, mode=mode, reuse=reuse,
                                           dtype=dtype, **kwargs)
        self.num_layers = self.config["num_layers"]
        self.architecture = self.config["architecture"]
        self.rnn_size = self.config["rnn_size"]
        self.states = None

        if self.reuse is False:
            print("One hot is ", self.one_hot)
            print("Input size is %d" % self.input_size)
            print('rnn_size = {0}'.format(self.rnn_size))

        # === Transform the inputs ===
        # with tf.name_scope(self.mode):
        with tf.name_scope("inputs"):
            self.encoder_inputs = tf.placeholder(self.dtype, shape=[None, self.source_seq_len - 1, self.input_size], name="enc_in")
            self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
            self.decoder_outputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

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
        if self.config['cell_type'] == C.GRU:
            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
        elif self.config['cell_type'] == C.LSTM:
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        else:
            raise Exception("Cell not found.")

        dropout_rate = self.config['input_layer'].get('dropout_rate', 0)
        if dropout_rate > 0:
            cell = rnn_cell_extensions.InputDropoutWrapper(cell, self.is_training, dropout_rate)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_layers)])

        with tf.variable_scope("seq2seq", reuse=self.reuse):
            # === Add space decoder ===
            ignore_actions = False
            if self.joint_prediction_model == "fk_joints":
                cell = rnn_cell_extensions.StructuredOutputWrapper(cell,
                                                                   self.structure_indexed,
                                                                   hidden_size=self.output_layer_config.get('size', 0),
                                                                   num_hidden_layers=self.output_layer_config.get('num_layers', 0),
                                                                   activation_fn=self.activation_fn,
                                                                   joint_size=self.JOINT_SIZE,
                                                                   human_size=self.HUMAN_SIZE,
                                                                   reuse=self.reuse,
                                                                   is_sparse=self.use_sparse_fk_joints)
                ignore_actions = True
            else:
                cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)

            # Finally, wrap everything in a residual layer if we want to model velocities
            if self.residual_velocities:
                cell = rnn_cell_extensions.ResidualWrapper(cell, action_len=self.ACTION_SIZE, ignore_actions=ignore_actions)

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

    def optimization_routines(self):
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if self.config["optimizer"] == "adam":
            opt = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config["optimizer"] == "sgd":
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise Exception()
        # Update all the trainable parameters
        gradients = tf.gradients(self.loss, params)
        # Apply gradient clipping.
        if self.grad_clip_by_norm > 0:
            clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.grad_clip_by_norm)
        else:
            self.gradient_norms = tf.linalg.global_norm(gradients)
            clipped_gradients = gradients
        self.parameter_update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
        input_feed = {self.encoder_inputs : encoder_inputs,
                      self.decoder_inputs : decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."
        return self.step(encoder_inputs, decoder_inputs, decoder_outputs)[2]


class AGED(Seq2SeqModel):
    """
    Implementation of Adversarial Geometry-Aware Human Motion Prediction:
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Adversarial_Geometry-Aware_Human_ECCV_2018_paper.pdf
    """
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(AGED, self).__init__(config=config, session=session, mode=mode, reuse=reuse,
                                   dtype=dtype, **kwargs)

        self.d_weight = config['discriminator_weight']
        self.use_adversarial = config['use_adversarial']
        self.fidelity_real = None
        self.fidelity_fake = None
        self.continuity_real = None
        self.continuity_fake = None
        self.pred_loss = None
        self.g_loss = None
        self.fidelity_loss = None
        self.continuity_loss = None
        self.d_loss = None
        self.g_param_update = None
        self.d_param_update = None
        self.g_gradient_norms = None
        self.d_gradient_norms = None

    def build_network(self):
        # Build the predictor
        with tf.variable_scope("AGED/generator", reuse=self.reuse):
            super(AGED, self).build_network()

        self.outputs_tensor = tf.transpose(self.outputs_tensor, [1, 0, 2])

        # TODO(kamanuel) are input dimensions correct here?
        if self.use_adversarial:
            # Fidelity Discriminator
            # real inputs
            self.fidelity_real = self.fidelity_discriminator(tf.transpose(self.prediction_targets, [1, 0, 2]),
                                                             reuse=not self.is_training)
            # fake inputs
            self.fidelity_fake = self.fidelity_discriminator(self.outputs_tensor, reuse=True)

            # Continuity Discriminator
            # real inputs
            data_inputs = tf.concat([self.encoder_inputs, self.decoder_inputs, self.decoder_outputs[:, -1:]], axis=1)
            self.continuity_real = self.continuity_discriminator(data_inputs, reuse=not self.is_training)
            # fake inputs (real seed + prediction)
            c_inputs = tf.concat([self.encoder_inputs, self.decoder_inputs[:, :1], self.outputs_tensor], axis=1)
            self.continuity_fake = self.continuity_discriminator(c_inputs, reuse=True)

    def fidelity_discriminator(self, inputs, reuse=False):
        """Judges fidelity of predictions"""
        input_hidden = self.config['fidelity_input_layer_size']
        cell_size = self.config['fidelity_cell_size']
        cell_type = self.config['fidelity_cell_type']

        with tf.variable_scope("AGED/discriminator/fidelity", reuse=reuse):
            inputs = tf.layers.dense(inputs, input_hidden, activation=tf.nn.relu, reuse=reuse)
            logits = self._build_discriminator_rnn(inputs, cell_type, cell_size, reuse)
        return logits

    def continuity_discriminator(self, inputs, reuse=False):
        """Judges continuity of the entire motion sequence."""
        input_hidden = self.config['continuity_input_layer_size']
        cell_size = self.config['continuity_cell_size']
        cell_type = self.config['continuity_cell_type']

        with tf.variable_scope("AGED/discriminator/continuity", reuse=reuse):
            inputs = tf.layers.dense(inputs, input_hidden, activation=tf.nn.relu, reuse=reuse)
            logits = self._build_discriminator_rnn(inputs, cell_type, cell_size, reuse)
        return logits

    def _build_discriminator_rnn(self, inputs, cell_type, cell_size, reuse):
        if cell_type == C.GRU:
            cell = tf.nn.rnn_cell.GRUCell(cell_size, reuse=reuse, name="cell")
        elif cell_type == C.LSTM:
            cell = tf.nn.rnn_cell.LSTMCell(cell_size, reuse=reuse, name="cell")
        else:
            raise Exception("Cell type unknown.")

        _, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        logits = tf.layers.dense(final_state, 1, reuse=reuse, name="out")
        return logits

    def build_loss(self):
        # compute prediction loss using the geodesic loss
        self.pred_loss = self.geodesic_loss(self.outputs_tensor,
                                            tf.transpose(self.prediction_targets, [1, 0, 2]))

        if self.use_adversarial:
            # fidelity discriminator loss
            f_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fidelity_real,
                                                             labels=tf.ones_like(self.fidelity_real))
            f_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fidelity_fake,
                                                             labels=tf.zeros_like(self.fidelity_fake))
            self.fidelity_loss = tf.reduce_mean(f_real + f_fake)

            # continuity discriminator loss
            c_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.continuity_real,
                                                             labels=tf.ones_like(self.continuity_real))
            c_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.continuity_fake,
                                                             labels=tf.zeros_like(self.continuity_fake))
            self.continuity_loss = tf.reduce_mean(c_real + c_fake)
            self.d_loss = self.continuity_loss + self.fidelity_loss

            fid_loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fidelity_fake,
                                                                 labels=tf.ones_like(self.fidelity_fake))

            con_loss_g = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fidelity_fake,
                                                                 labels=tf.ones_like(self.continuity_fake))

            self.g_loss = tf.reduce_mean(con_loss_g + fid_loss_g)
            self.loss = self.pred_loss + self.d_weight*self.g_loss
        else:
            self.loss = self.pred_loss

    def geodesic_loss(self, predictions, targets):
        # convert to rotation matrices
        # assert self.use_aa

        # must unnormalize before computing the geodesic loss
        pred = self._unnormalize(predictions)
        targ = self._unnormalize(targets)

        pred = tf.reshape(pred[:, :, :-self.number_of_actions], [-1, self.target_seq_len, self.NUM_JOINTS, 3])
        targ = tf.reshape(targ[:, :, :-self.number_of_actions], [-1, self.target_seq_len, self.NUM_JOINTS, 3])

        pred_rot = tf_tr_quat.from_axis_angle(pred)
        targ_rot = tf_tr_quat.from_axis_angle(targ)
        # pred_rot = aa2rotmat(pred)
        # targ_rot = aa2rotmat(targ)

        # A = (tf.matmul(pred_rot, targ_rot, transpose_b=True) - tf.matmul(targ_rot, pred_rot, transpose_b=True)) / 2.0
        # A = tf.stack([-A[:, :, :, 1, 2], A[:, :, :, 0, 0], -A[:, :, :, 0, 1]], axis=-1)
        #
        # # The norm of A is equivalent to sin(theta)
        # A_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.multiply(A, A), axis=-1), 1e-12))
        # theta = tf.asin(tf.clip_by_value(A_norm, -1.0, 1.0))
        # geodesic_loss = tf.reduce_mean(tf.reduce_sum(theta, axis=(1, 2)))

        theta = tf_tr_quat.relative_angle(pred_rot, targ_rot)
        geodesic_loss = tf.reduce_mean(tf.reduce_sum(theta, axis=(1, 2)))
        return geodesic_loss

    def _unnormalize(self, data):
        if self.normalization_std is not None:
            return data*self.normalization_std + self.normalization_mean
        else:
            return data

    def optimization_routines(self):
        self._generator_optimizer()
        if self.use_adversarial:
            self._discriminator_optimizer()

    def _get_optimizer(self):
        if self.config["optimizer"] == C.OPTIMIZER_ADAM:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config["optimizer"] == C.OPTIMIZER_SGD:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise Exception("Optimization not found.")
        return optimizer

    def _generator_optimizer(self):
        """Optimizer for the generator."""
        optimizer = self._get_optimizer()
        g_vars = tf.trainable_variables(scope="AGED/generator")

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="AGED/generator")):
            # update the generator variables w.r.t. the total loss
            gradients = tf.gradients(self.loss, g_vars)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.g_gradient_norms = tf.clip_by_global_norm(gradients,
                                                                          self.config.get('grad_clip_by_norm'))
            else:
                self.g_gradient_norms = tf.global_norm(gradients)
            self.g_param_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, g_vars),
                                                            global_step=self.global_step)

    def _discriminator_optimizer(self):
        """Optimizer for both discriminators."""
        optimizer = self._get_optimizer()
        d_vars = tf.trainable_variables(scope="AGED/discriminator")

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="AGED/discriminator")):
            # update the generator variables w.r.t. the total loss
            gradients = tf.gradients(self.d_loss, d_vars)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.d_gradient_norms = tf.clip_by_global_norm(gradients,
                                                                          self.config.get('grad_clip_by_norm'))
            else:
                self.d_gradient_norms = tf.global_norm(gradients)
            self.d_param_update = optimizer.apply_gradients(
                grads_and_vars=zip(gradients, d_vars))

    def summary_routines(self):
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        if self.use_adversarial:
            # prediction loss only
            tf.summary.scalar(self.mode+"/pred_loss", self.pred_loss, collections=[self.mode + "/model_summary"])
            # adversarial loss for generator
            tf.summary.scalar(self.mode+"/g_loss", self.g_loss, collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode+"/d_fid_loss", self.fidelity_loss, collections=[self.mode+"/model_summary"])
            tf.summary.scalar(self.mode+"/d_con_loss", self.continuity_loss, collections=[self.mode+"/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/learning_rate",
                              self.learning_rate,
                              collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/g_gradient_norms",
                              self.g_gradient_norms,
                              collections=[self.mode + "/model_summary"])
            if self.use_adversarial:
                tf.summary.scalar(self.mode + "/d_gradient_norms",
                                  self.d_gradient_norms,
                                  collections=[self.mode + "/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")
        self.euler_summaris()

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_inputs: decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.g_param_update]
            if self.use_adversarial:
                output_feed.append(self.d_param_update)
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            if self.use_adversarial:
                output_feed.extend([self.g_loss,
                                    self.continuity_loss,
                                    self.d_loss])
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]


class Seq2SeqFeedbackModel(Seq2SeqModel):
    """Sequence-to-sequence model with error feedback,"""
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(Seq2SeqFeedbackModel, self).__init__(config=config, session=session, mode=mode, reuse=reuse,
                                                   dtype=dtype, **kwargs)
        self.feed_error_to_encoder = config['feed_error_to_encoder']

    def create_cell(self, scope, reuse, error_signal_size=0):
        with tf.variable_scope(scope, reuse=reuse):
            rnn_cell = tf.nn.rnn_cell.GRUCell
            cell = rnn_cell(self.rnn_size)
            if self.num_layers > 1:
                cells = [rnn_cell(self.rnn_size) for _ in range(self.num_layers)]
                cell = tf.contrib.rnn.MultiRNNCell(cells)

            # === Add space decoder ===
            cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)
            # Finally, wrap everything in a residual layer if we want to model velocities
            if self.residual_velocities:
                cell = rnn_cell_extensions.ResidualWrapper(cell, error_signal_size)

            return cell

    def build_network(self):
        # === Create the RNN that will keep the state ===
        if self.architecture == "basic":
            # may be we don't feed the error in the encoder
            error_signal_size = self.HUMAN_SIZE if self.feed_error_to_encoder else 0
            encoder_cell = self.create_cell("encoder_cell", self.reuse, error_signal_size)
            decoder_cell = self.create_cell("decoder_cell", self.reuse, self.HUMAN_SIZE)
        elif self.architecture == "tied":
            # in tied architecture we always feed the error (not possible to not feed it)
            encoder_cell = self.create_cell("tied_cell", self.reuse, self.HUMAN_SIZE)
            decoder_cell = encoder_cell
        else:
            raise Exception()

        with tf.variable_scope("seq2seq", reuse=self.reuse):
            def loop_fn_sampling(pred, current):
                """
                Computes error between prediction and ground-trugh and appends it to the prediction.
                Args:
                    pred: model prediction
                    current: ground-truth

                Returns:
                    the predicted pose with the error appended
                """
                # compute error w.r.t. ground-truth, but we don't want to include one-hot encoded actions
                c = current[:, :self.HUMAN_SIZE]
                p = pred[:, :self.HUMAN_SIZE]
                error = tf.sqrt(tf.square(c - p))
                # treat error as a constant
                error = tf.stop_gradient(error)
                # feed back the models own prediction
                return tf.concat([pred, error], axis=-1)

            def loop_fn_supervised(pred, current):
                """
                Computes error between prediction and ground-truth and appends it the ground-truth.
                Args:
                    pred: model prediction
                    current: ground-truth

                Returns:
                    the ground-truth with the error appended
                """
                # compute error w.r.t. ground-truth, but we don't want to include one-hot encoded actions
                c = current[:, :self.HUMAN_SIZE]
                p = pred[:, :self.HUMAN_SIZE]
                error = tf.sqrt(tf.square(c - p))
                # treat error as a constant
                error = tf.stop_gradient(error)
                # feed back the ground truth sample (this doesn't make much sense, but here for completeness
                return tf.concat([current, error], axis=-1)

            def loop_fn_gt(pred, current):
                """
                Args:
                    pred: model prediction
                    current: ground-truth

                Returns:
                    the ground-truth
                """
                return current

            if self.architecture == "tied":
                # encoder always uses ground truth and also computes error
                encoder_loop_fn = loop_fn_supervised
            else:
                # in untied mode we can choose
                encoder_loop_fn = loop_fn_supervised if self.feed_error_to_encoder else loop_fn_gt

            # decoder depends on configuration
            if self.autoregressive_input == "sampling_based":
                decoder_loop_fn = loop_fn_sampling
            elif self.autoregressive_input == "supervised":
                print("WARNING: using ground-truth poses in decoder might be nonsense")
                decoder_loop_fn = loop_fn_supervised
            else:
                raise ValueError("'{}' loss unknown".format(self.autoregressive_input))

            with tf.variable_scope("rnn_encoder"):
                outputs = []
                prev = self.enc_in[0]
                state = encoder_cell.zero_state(batch_size=tf.shape(prev)[0], dtype=prev.dtype)
                for i, inp in enumerate(self.enc_in):
                    with tf.variable_scope("encoder_loop_fn", reuse=True):
                        inp = encoder_loop_fn(prev, inp)
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = encoder_cell(inp, state)
                    outputs.append(output)
                    prev = output
            enc_state = state

            # Decoder with error feedback.
            # loop is from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L142
            with tf.variable_scope("rnn_decoder"):
                state = enc_state
                outputs = []
                prev = self.dec_in[0]  # self.dec_in contains ground truth
                for i, inp in enumerate(self.dec_in):
                    with tf.variable_scope("decoder_loop_fn", reuse=True):
                        inp = decoder_loop_fn(prev, inp)
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = decoder_cell(inp, state)
                    outputs.append(output)
                    prev = output

            self.outputs, self.states = outputs, state
            self.outputs_tensor = tf.stack(self.outputs)
        self.build_loss()


class Wavenet(BaseModel):
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(Wavenet, self).__init__(config=config, session=session, mode=mode, reuse=reuse, dtype=dtype, **kwargs)
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
            # We concatenate encoder and decoder inputs. The model is fed with a slice of [0:-1] while the target
            # is shifted by one step (i.e., a slice of [1:]).
            # If we have a sequence of [0,1...13,14], source_seq_len and target_seq_len with values 10 and 5:
            # [0,1,2,3,4,5,6,7,8]
            self.encoder_inputs = tf.placeholder(self.dtype, shape=[None, self.source_seq_len - 1, self.input_size], name="enc_in")
            # [9,10,11,12,13]
            if self.is_training:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
                self.sequence_length = self.source_seq_len + self.target_seq_len - 1
            else:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, None, self.input_size], name="dec_in")
                self.sequence_length = tf.shape(self.decoder_inputs)[1]
            # [10,11,12,13,14]
            self.decoder_outputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

        # Get the last frame of decoder_outputs in order to use in approximate inference.
        last_frame = self.decoder_outputs[:, -1:, :]
        all_frames = tf.concat([self.encoder_inputs, self.decoder_inputs, last_frame], axis=1)
        self.prediction_inputs = all_frames[:, :-1, :]
        self.prediction_targets = all_frames[:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

    def build_network(self):
        # We always pad the input sequences such that the output sequence has the same length with input sequence.
        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.prediction_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden, rate=self.input_layer_config.get("dropout_rate"), seed=self.config["seed"], training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

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
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer, self.num_decoder_blocks, self.reuse, kernel_size=decoder_filter_size)
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
        num_filters = self.cnn_layer_config['num_filters'] if self.output_layer_config.get('size', 0) < 1 else self.output_layer_config.get('size')
        num_hidden_layers = self.output_layer_config.get('num_layers', 0)

        current_layer = inputs
        if out_layer_type == C.LAYER_CONV1:
            for layer_idx in range(num_hidden_layers):
                with tf.variable_scope('out_conv1d_' + name + "_" + str(layer_idx), reuse=self.reuse):
                    current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                     filters=num_filters, dilation_rate=1, activation=self.activation_fn)

            with tf.variable_scope('out_conv1d_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
                prediction = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                              filters=output_size, dilation_rate=1, activation=None)

        elif out_layer_type == C.LAYER_TCN:
            kernel_size = self.cnn_layer_config['filter_size'] if self.output_layer_config.get('filter_size', 0) < 1 else self.output_layer_config.get('filter_size', 0)
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
                prediction, _ = Wavenet.temporal_block_ccn(input_layer=current_layer,
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

            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params), global_step=self.global_step)

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
        input_feed = {self.encoder_inputs : encoder_inputs,
                      self.decoder_inputs : decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."

        input_sequence = np.concatenate([encoder_inputs, decoder_inputs[:, 0:1, :]], axis=1)
        predictions = []
        for step in range(self.target_seq_len):
            end_idx = min(self.receptive_field_width, input_sequence.shape[1])
            model_inputs = input_sequence[:, -end_idx:]

            # get the prediction
            model_outputs = self.session.run(self.outputs_tensor, feed_dict={self.prediction_inputs: model_inputs})
            prediction = model_outputs[:, -1, :]

            # if action vector is predicted, must convert the logits to one-hot vectors
            if self.action_loss_type == C.LOSS_ACTION_CENT:
                action_logits = prediction[:, self.HUMAN_SIZE:]
                action_probs = softmax(action_logits)
                max_idx = np.argmax(action_probs, axis=-1)
                one_hot = np.eye(self.ACTION_SIZE)[max_idx]
                prediction[:, self.HUMAN_SIZE:] = one_hot

            predictions.append(prediction)
            input_sequence = np.concatenate([input_sequence, np.expand_dims(predictions[-1], axis=1)], axis=1)

        return predictions

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
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(STCN, self).__init__(config=config, session=session, mode=mode, reuse=reuse, dtype=dtype, **kwargs)
        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")
        self.use_future_steps_in_q = self.config.get('use_future_steps_in_q', False)
        self.bw_encoder_blocks = []
        self.bw_encoder_blocks_no_res = []
        self.latent_layer = None
        self.latent_samples = None

        # Get the last frame of decoder_outputs in order to use in approximate inference.
        last_frame = self.decoder_outputs[:, -1:, :]
        all_frames = tf.concat([self.encoder_inputs, self.decoder_inputs, last_frame], axis=1)
        self.prediction_inputs = all_frames  # The q and p models require all frames.
        self.prediction_targets = all_frames[:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

    def build_network(self):
        self.latent_layer = LatentLayer.get(config=self.latent_layer_config,
                                            layer_type=self.latent_layer_config["type"],
                                            mode=self.mode,
                                            reuse=self.reuse,
                                            global_step=self.global_step)

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
        self.kld_weight_summary = tf.summary.scalar(self.mode + "/kld_weight", self.latent_layer.kld_weight,
                                                    collections=[self.mode + "/model_summary"])
        super(STCN, self).summary_routines()

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."

        input_sequence = np.concatenate([encoder_inputs, decoder_inputs[:, 0:1, :]], axis=1)
        dummy_frame = np.zeros([input_sequence.shape[0], 1, input_sequence.shape[2]])
        predictions = []
        for step in range(self.target_seq_len):
            end_idx = min(self.receptive_field_width, input_sequence.shape[1])
            model_inputs = input_sequence[:, -end_idx:]

            # Insert a dummy frame since the sampling model ignores the last step.
            model_inputs = np.concatenate([model_inputs, dummy_frame], axis=1)

            # get the prediction
            model_outputs = self.session.run(self.outputs_tensor, feed_dict={self.prediction_inputs: model_inputs})
            prediction = model_outputs[:, -1, :]

            # if action vector is predicted, must convert the logits to one-hot vectors
            if self.action_loss_type == C.LOSS_ACTION_CENT:
                action_logits = prediction[:, self.HUMAN_SIZE:]
                action_probs = softmax(action_logits)
                max_idx = np.argmax(action_probs, axis=-1)
                one_hot = np.eye(self.ACTION_SIZE)[max_idx]
                prediction[:, self.HUMAN_SIZE:] = one_hot

            predictions.append(prediction)
            input_sequence = np.concatenate([input_sequence, np.expand_dims(predictions[-1], axis=1)], axis=1)

        return predictions


class StructuredSTCN(STCN):
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(StructuredSTCN, self).__init__(config=config, session=session, mode=mode, reuse=reuse, dtype=dtype,
                                             **kwargs)

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
        self.build_loss()

    def build_output_layer(self):
        """
        Builds layers to make predictions. The structured latent space has a random variable per joint.
        """

        def traverse_parents(tree, source_list, output_list, parent_id):
            """
            Traverses parent joints up to the root and appends the parent value in source_list into the output_list.
            """
            if parent_id >= 0:
                output_list.append(source_list[parent_id])
                traverse_parents(tree, source_list, output_list, tree[parent_id][0])

        with tf.variable_scope('output_layer', reuse=self.reuse):
            prediction = []
            for joint_key in sorted(self.structure_indexed.keys()):
                parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                joint_sample = self.latent_samples[joint_key]
                if self.joint_prediction_model == "plain":
                    prediction.append(joint_sample)

                elif self.joint_prediction_model == "separate_joints":
                    joint_prediction = self.build_predictions(joint_sample, self.JOINT_SIZE, joint_name)
                    prediction.append(joint_prediction)

                elif self.joint_prediction_model == "fk_joints":
                    joint_inputs = []
                    traverse_parents(self.structure_indexed, self.latent_layer.latent_samples_indexed, joint_inputs, parent_joint_idx)
                    joint_inputs.append(joint_sample)
                    prediction.append(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))
                else:
                    raise Exception("Prediction model not recognized.")

            # TODO action labels for skip connection.
            prediction.append(tf.zeros_like(self.prediction_inputs[:, 0:-1, -self.ACTION_SIZE:]))
            self.outputs_mu = tf.concat(prediction, axis=-1)

            if self.residual_velocities:
                self.outputs_mu += self.prediction_inputs[:, 0:-1]

            # This code repository expects the outputs to be a list of time-steps.
            # outputs_list = tf.split(self.outputs_mu, self.sequence_length, axis=1)
            # Select only the "decoder" predictions.
            self.outputs = [tf.squeeze(out_frame, axis=1) for out_frame in tf.split(self.outputs_tensor[:, -self.target_seq_len:], self.target_seq_len, axis=1)]

    def build_loss(self):
        super(StructuredSTCN, self).build_loss()


class RNN(BaseModel):
    """
    Autoregressive RNN.
    """
    def __init__(self, config, session, mode, reuse, dtype, **kwargs):
        super(RNN, self).__init__(config, session, mode, reuse, dtype, **kwargs)
        self.cell_config = self.config.get("cell")
        self.input_layer_config = config.get('input_layer', None)

        self.cell = None
        self.initial_states = None
        self.rnn_outputs = None  # Output of RNN layer.
        self.rnn_state = None  # Final state of RNN layer.
        self.inputs_hidden = None

        self.summary_ops = dict()  # A container for summary ops of this model. We use "model_summary" collection name.
        self.kld_weight_summary = None

        with tf.name_scope("inputs"):
            # If we have a sequence of [0,1...13,14], source_seq_len and target_seq_len with values 10 and 5:
            # [0,1,2,3,4,5,6,7,8]
            self.encoder_inputs = tf.placeholder(self.dtype, shape=[None, self.source_seq_len - 1, self.input_size], name="enc_in")
            # [9,10,11,12,13]
            if self.is_training:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
                self.sequence_length = self.source_seq_len + self.target_seq_len - 1
            else:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, None, self.input_size], name="dec_in")
                self.sequence_length = tf.shape(self.decoder_inputs)[1]
            # [10,11,12,13,14]
            self.decoder_outputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

        # Get the last frame of decoder_outputs in order to use in approximate inference.
        last_frame = self.decoder_outputs[:, -1:, :]
        all_frames = tf.concat([self.encoder_inputs, self.decoder_inputs, last_frame], axis=1)
        self.prediction_inputs = all_frames[:, :-1, :]
        self.prediction_targets = all_frames[:, 1:, :]
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            self.tf_batch_size = tf.shape(self.prediction_inputs)[0]

    def build_input_layer(self):
        current_layer = self.prediction_inputs
        if self.input_layer_config is not None:
            dropout_rate = self.input_layer_config.get("dropout_rate", 0)
            if dropout_rate > 0:
                with tf.variable_scope('input_dropout', reuse=self.reuse):
                    current_layer = tf.layers.dropout(current_layer,
                                                      rate=self.input_layer_config.get("dropout_rate"),
                                                      seed=self.config["seed"],
                                                      training=self.is_training)
            hidden_size = self.input_layer_config.get('size', 0)
            for layer_idx in range(self.input_layer_config.get("num_layers", 0)):
                with tf.variable_scope("inp_dense_" + str(layer_idx), reuse=self.reuse):
                    current_layer = tf.layers.dense(inputs=current_layer,
                                                    units=hidden_size,
                                                    activation=self.activation_fn)
        self.inputs_hidden = current_layer
        return self.inputs_hidden

    def create_cell(self):
        return get_rnn_cell(cell_type=self.cell_config["cell_type"],
                            size=self.cell_config["cell_size"],
                            num_layers=self.cell_config["cell_num_layers"],
                            mode=self.mode,
                            reuse=self.reuse)

    def build_network(self):
        self.cell = self.create_cell()
        self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)

        self.inputs_hidden = self.build_input_layer()

        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                                 self.inputs_hidden,
                                                                 sequence_length=self.prediction_seq_len,
                                                                 initial_state=self.initial_states,
                                                                 dtype=tf.float32)
            self.prediction_representation = self.rnn_outputs
        self.build_output_layer()
        self.build_loss()

    def build_loss(self):
        super(RNN, self).build_loss()

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

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
        input_feed = {self.encoder_inputs : encoder_inputs,
                      self.decoder_inputs : decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."
        assert self.action_loss_type != C.LOSS_ACTION_CENT, "Action labels not supported."

        num_samples = encoder_inputs.shape[0]
        input_sequence = np.concatenate([encoder_inputs, decoder_inputs[:, 0:1, :]], axis=1)
        # Get the model state by feeding the seed sequence.
        feed_dict = {self.prediction_inputs : input_sequence,
                     self.prediction_seq_len: np.ones(input_sequence.shape[0])*input_sequence.shape[1]}
        state, prediction = self.session.run([self.rnn_state, self.outputs_tensor], feed_dict=feed_dict)

        predictions = [prediction[:, -1]]
        for step in range(self.target_seq_len - 1):
            # get the prediction
            feed_dict = {self.prediction_inputs : prediction,
                         self.initial_states    : state,
                         self.prediction_seq_len: np.ones(num_samples)}
            state, prediction = self.session.run([self.rnn_state, self.outputs_tensor], feed_dict=feed_dict)
            predictions.append(prediction[:, -1])

        return predictions
