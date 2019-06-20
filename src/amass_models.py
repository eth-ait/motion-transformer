from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf

import rnn_cell_extensions  # my extensions of the tf repos

# ETH imports
from constants import Constants as C
from tf_model_utils import get_activation_fn, get_rnn_cell, get_decay_variable
from tf_rnn_cells import LatentCell
from tf_loss_quat import quaternion_norm
from tf_loss_quat import quaternion_loss
from tf_loss import logli_normal_isotropic

import tf_tr_quat


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
        self.residual_velocities_type = config.get("residual_velocities_type", "plus")
        self.residual_velocities_reg = None  # a regularizer in the residual velocity to be added to the loss
        self.angle_loss_type = config["angle_loss_type"]
        self.joint_prediction_model = config["joint_prediction_model"]
        self.grad_clip_by_norm = config["grad_clip_by_norm"]
        self.loss_on_encoder_outputs = config['loss_on_encoder_outputs']
        self.force_valid_rot = config.get('force_valid_rot', False)
        self.output_layer_config = config.get('output_layer', dict())
        self.activation_fn = get_activation_fn(self.output_layer_config.get('activation_fn', None))
        self.rot_matrix_regularization = config.get('rot_matrix_regularization', False)
        self.prediction_activation = None if not self.rot_matrix_regularization else tf.nn.tanh
        self.use_quat = config.get('use_quat', False)
        self.use_aa = config.get('use_aa', False)
        self.h36m_martinez = config.get("use_h36m_martinez", False)
        # Model the outputs with Normal distribution.
        self.mle_normal = self.angle_loss_type == C.LOSS_POSE_NORMAL

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN

        self.data_inputs = data_pl[C.BATCH_INPUT]
        self.data_targets = data_pl[C.BATCH_TARGET]
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]
        self.data_ids = data_pl[C.BATCH_ID]

        # It is always 1 when we report the loss.
        if not self.is_training:
            self.kld_weight = 1.0

        # Defines how to employ structured latent variables to make predictions.
        # Options are
        # (1) "plain": latent samples correspond to joint predictions. dimensions must meet.
        # (2) "separate_joints": each latent variable is transformed into a joint prediction by using separate networks.
        # (3) "fk_joints": latent samples on the forward kinematic chain are concatenated and used as in (2).
        self.joint_prediction_model = config.get('joint_prediction_model', "plain")
        if config.get('use_sparse_fk_joints', False):
            # legacy, only used so that evaluation of old models can still work
            self.joint_prediction_model = "fk_joints_sparse"

        assert self.joint_prediction_model in ["plain", "separate_joints", "fk_joints",
                                               "fk_joints_sparse", "fk_joints_stop_gradients",
                                               "fk_joints_sparse_shared"]

        # Set by the child model class.
        self.outputs_mu = None  # Mu tensor of predicted frames (Normal distribution).
        self.outputs_sigma = None  # Sigma tensor of predicted frames (Normal distribution).
        self.outputs_mu_joints = list()  # List of individual joint predictions.
        self.outputs_sigma_joints = list()  # List of individual joint predictions.

        self.outputs = None  # List of predicted frames. If the model is probabilistic, a sample is drawn first.
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

        self.prediction_norm = None

        # Hard-coded parameters.
        self.JOINT_SIZE = 4 if self.use_quat else 3 if self.use_aa else 9
        self.NUM_JOINTS = 21 if self.h36m_martinez else 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

        # [(Parent ID, Joint ID, Joint Name), (...)] where each entry in a list corresponds to the joints at the same
        # level in the joint tree.
        self.structure = [[(-1, 0, "l_hip"), (-1, 1, "r_hip"), (-1, 2, "spine1")],
                          [(0, 3, "l_knee"), (1, 4, "r_knee"), (2, 5, "spine2")],
                          [(5, 6, "spine3")],
                          [(6, 7, "neck"), (6, 8, "l_collar"), (6, 9, "r_collar")],
                          [(7, 10, "head"), (8, 11, "l_shoulder"), (9, 12, "r_shoulder")],
                          [(11, 13, "l_elbow"), (12, 14, "r_elbow")]]

        if self.h36m_martinez:
            self.structure = [[(-1, 0, "Hips")],
                              [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
                              [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
                              [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"),
                               (10, 13, "LeftShoulder"), (10, 11, "Neck")],
                              [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"),
                               (11, 12, "Head")],
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

        # Annealing input dropout rate or using fixed rate.
        self.input_dropout_rate = None
        if config.get("input_layer", None) is not None:
            if isinstance(config["input_layer"].get("dropout_rate", 0), dict):
                self.input_dropout_rate = get_decay_variable(global_step=self.global_step,
                                                             config=config["input_layer"].get("dropout_rate"),
                                                             name="input_dropout_rate")
            elif config["input_layer"].get("dropout_rate", 0) > 0:
                self.input_dropout_rate = config["input_layer"].get("dropout_rate")

        self.normalization_var = kwargs.get('var_channel', None)
        self.normalization_mean = kwargs.get('mean_channel', None)

    def build_graph(self):
        self.build_network()
        self.build_loss()

    def build_network(self):
        pass

    def build_loss(self):
        if self.is_eval or not self.loss_on_encoder_outputs:
            predictions_pose = self.outputs[:, -self.target_seq_len:, :]
            targets_pose = self.prediction_targets[:, -self.target_seq_len:, :]
            seq_len = self.target_seq_len
        else:
            predictions_pose = self.outputs
            targets_pose = self.prediction_targets
            seq_len = tf.shape(self.outputs)[1]

        with tf.name_scope("loss_angles"):
            diff = targets_pose - predictions_pose
            if self.angle_loss_type == "quat_l2":
                assert self.use_quat
                # this is equivalent to log(R*R^T)
                loss_per_frame = quaternion_loss(targets_pose, predictions_pose, self.angle_loss_type)
                loss_per_sample = tf.reduce_sum(loss_per_frame, axis=-1)
                loss_per_batch = tf.reduce_mean(loss_per_sample)
                self.loss = loss_per_batch
            elif self.angle_loss_type == C.LOSS_POSE_ALL_MEAN:
                pose_loss = tf.reduce_mean(tf.square(diff))
                self.loss = pose_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_MEAN:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                self.loss = per_joint_loss
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_SUM:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                self.loss = per_joint_loss
            elif self.angle_loss_type == C.LOSS_POSE_NORMAL:
                pose_likelihood = logli_normal_isotropic(targets_pose, self.outputs_mu, self.outputs_sigma)
                pose_likelihood = tf.reduce_sum(pose_likelihood, axis=[1, 2])
                pose_likelihood = tf.reduce_mean(pose_likelihood)
                self.loss = -pose_likelihood
            else:
                raise Exception("Unknown angle loss.")

        if self.residual_velocities_reg is not None:
            self.loss += self.residual_velocities_reg

        if self.rot_matrix_regularization:
            with tf.name_scope("output_rot_mat_regularization"):
                rot_matrix_loss = self.rot_mat_regularization(predictions_pose, summary_name="rot_matrix_reg")
                self.loss += rot_matrix_loss

    def optimization_routines(self):
        if self.config["optimizer"] == C.OPTIMIZER_ADAM:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.config["optimizer"] == C.OPTIMIZER_SGD:
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise Exception("Optimization not found.")

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            # Gradient clipping.
            gradients = tf.gradients(self.loss, params)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.config.get('grad_clip_by_norm'))
            else:
                self.gradient_norms = tf.global_norm(gradients)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=self.global_step)

    def step(self, session):
        pass

    def sampled_step(self, session):
        pass

    def parse_outputs(self, prediction_dict):
        self.outputs_mu_joints.append(prediction_dict["mu"])
        if self.mle_normal:
            self.outputs_sigma_joints.append(prediction_dict["sigma"])

    def aggregate_outputs(self):
        self.outputs_mu = tf.concat(self.outputs_mu_joints, axis=-1)
        assert self.outputs_mu.get_shape()[-1] == self.HUMAN_SIZE, "Prediction not matching with the skeleton."
        if self.mle_normal:
            self.outputs_sigma = tf.concat(self.outputs_sigma_joints, axis=-1)

    def get_joint_prediction(self, joint_idx=-1):
        """
        Returns the predicted joint value or whole body.
        """
        if joint_idx < 0:  # whole body.
            assert self.outputs_mu is not None, "Whole body is not predicted yet."
            if self.mle_normal:
                return self.outputs_mu + tf.random.normal(tf.shape(self.outputs_sigma))*self.outputs_sigma
            else:
                return self.outputs_mu
        else:  # individual joint.
            assert joint_idx < len(self.outputs_mu_joints), "The required joint is not predicted yet."
            if self.mle_normal:
                return self.outputs_mu_joints[joint_idx] + tf.random.normal(
                    tf.shape(self.outputs_sigma_joints[joint_idx]))*self.outputs_sigma_joints[joint_idx]
            else:
                return self.outputs_mu_joints[joint_idx]

    def traverse_parents(self, output_list, parent_id):
        """
        Collects joint predictions recursively by following the kinematic chain.
        Args:
            output_list:
            parent_id:
        """
        if parent_id >= 0:
            output_list.append(self.get_joint_prediction(parent_id))
            self.traverse_parents(output_list, self.structure_indexed[parent_id][0])

    def traverse_parents_stop_gradients(self, output_list, parent_id, stop_gradients=False):
        """
        Collects joint predictions recursively by following the kinematic chain but optionally stops gradients.
        Args:
            output_list:
            parent_id:
            stop_gradients:

        """
        if parent_id >= 0:
            parent_prediction = self.get_joint_prediction(parent_id)
            if stop_gradients:
                parent_prediction = tf.stop_gradient(parent_prediction)
            output_list.append(parent_prediction)
            # after the first call we always stop gradients as we want gradients to flow only for the direct parent
            self.traverse_parents_stop_gradients(output_list, self.structure_indexed[parent_id][0], stop_gradients=True)

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        with tf.variable_scope('output_layer', reuse=self.reuse):
            if self.joint_prediction_model == "plain":
                self.parse_outputs(self.build_predictions(self.prediction_representation, self.HUMAN_SIZE, "all"))

            elif self.joint_prediction_model == "separate_joints":
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    self.parse_outputs(self.build_predictions(self.prediction_representation, self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model == "fk_joints":
                # each joint receives direct input from each ancestor in the kinematic chain
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    self.traverse_parents(joint_inputs, parent_joint_idx)
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))

            elif self.joint_prediction_model.startswith("fk_joints_sparse"):
                # each joint only receives the direct parent joint as input
                created_non_root_weights = False
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    if parent_joint_idx >= 0:
                        joint_inputs.append(self.outputs_mu_joints[parent_joint_idx])

                    if self.joint_prediction_model == "fk_joints_sparse_shared":
                        if parent_joint_idx == -1:
                            # this joint has no parent, so create its own layer
                            name = joint_name
                            share_weights = False
                        else:
                            # always share except for the first joint because we must create at least one layer
                            name = "non_root_shared"
                            share_weights = created_non_root_weights
                            if not created_non_root_weights:
                                created_non_root_weights = True
                    else:
                        name = joint_name
                        share_weights = False
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1),
                                                              self.JOINT_SIZE, name, share_weights))

            elif self.joint_prediction_model == "fk_joints_stop_gradients":
                # same as 'fk_joints' but gradients are stopped after the direct parent of each joint
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    self.traverse_parents_stop_gradients(joint_inputs, parent_joint_idx, stop_gradients=False)
                    self.parse_outputs(
                        self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))

            else:
                raise Exception("Joint prediction model '{}' unknown.".format(self.joint_prediction_model))

            self.aggregate_outputs()
            pose_prediction = self.outputs_mu

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                # some debugging
                self.prediction_norm = tf.linalg.norm(pose_prediction)
                # pose_prediction = tf.Print(pose_prediction, [tf.shape(pose_prediction)], "shape", summarize=100)
                # pose_prediction = tf.Print(pose_prediction, [tf.linalg.norm(pose_prediction[0])], "norm[0]", summarize=135)
                # pose_prediction = tf.Print(pose_prediction, [pose_prediction[0]], "pose_prediction[0]", summarize=135)
                # pose_prediction = tf.Print(pose_prediction, [self.prediction_inputs[0, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]], "inputs[0]", summarize=135)
                if self.residual_velocities_type == "plus":
                    pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]
                elif self.residual_velocities_type == "matmul":
                    # add regularizer to the predicted rotations
                    self.residual_velocities_reg = self.rot_mat_regularization(pose_prediction,
                                                                               summary_name="velocity_rot_mat_reg")
                    # now perform the multiplication
                    preds = tf.reshape(pose_prediction, [-1, 3, 3])
                    inputs = tf.reshape(self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE], [-1, 3, 3])
                    preds = tf.matmul(inputs, preds, transpose_b=True)
                    pose_prediction = tf.reshape(preds, tf.shape(pose_prediction))
                else:
                    raise ValueError("residual velocity type {} unknown".format(self.residual_velocities_type))

            # Enforce valid rotations as the very last step, this currently doesn't do anything with rotation matrices.
            # TODO(eaksan) Not sure how to handle probabilistic predictions. For now we use only the mu predictions.
            if self.force_valid_rot:
                pose_prediction = self.build_valid_rot_layer(pose_prediction)
            self.outputs_mu = pose_prediction
            self.outputs = self.get_joint_prediction(joint_idx=-1)

    def rot_mat_regularization(self, rotmats, summary_name="rot_matrix_reg"):
        """
        Computes || R * R^T - I ||_F and averages this over all joints, frames, and batch entries. Note that we
        do not enforce det(R) == 1.0 for now. The average is added to tensorboard as a summary.
        Args:
            rotmats: A tensor of shape (..., k*3*3)
            summary_name: Name for the summary

        Returns:
            The average deviation of all rotation matrices from being orthogonal.
        """
        rot_matrix = tf.reshape(rotmats, [-1, 3, 3])
        n = tf.shape(rot_matrix)[0]
        rrt = tf.matmul(rot_matrix, rot_matrix, transpose_b=True)
        eye = tf.eye(3, batch_shape=[n])
        rot_reg = tf.norm(rrt - eye, ord='fro', axis=(-1, -2))
        rot_reg = tf.reduce_mean(rot_reg)
        tf.summary.scalar(self.mode + "/" + summary_name, rot_reg, collections=[self.mode + "/model_summary"])
        return rot_reg

    def get_closest_rotmats(self, rotmats):
        """
        Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
        it computes the SVD as R = USV' and sets R_closest = UV'.

        WARNING: tf.svd is very slow - use at your own peril.

        Args:
            rotmats: A tensor of shape (N, seq_length, n_joints*9) containing the candidate rotation matrices.

        Returns:
            A tensor of the same shape as `rotmats` containing the closest rotation matrices.
        """
        assert not self.use_quat and not self.use_aa
        # reshape to (N, seq_len, n_joints, 3, 3)
        seq_length = tf.shape(rotmats)[1]
        dof = rotmats.get_shape()[-1].value
        rots = tf.reshape(rotmats, [-1, seq_length, dof//9, 3, 3])

        # add tanh activation function to map to [-1, 1]
        rots = tf.tanh(rots)

        # compute SVD
        # This is problematic when done on the GPU, see https://github.com/tensorflow/tensorflow/issues/13603
        s, u, v = tf.svd(rots, full_matrices=True)
        closest_rot = tf.matmul(u, v, transpose_b=True)
        closest_rot = tf.Print(closest_rot, [closest_rot], "done with SVD")

        # TODO(kamanuel) should we make sure that det == 1?

        raise ValueError("SVD on GPU is super slow, not recommended to use.")
        # return tf.reshape(closest_rot, [-1, seq_length, dof])

    def normalize_quaternions(self, quats):
        """
        Normalizes the input quaternions to have unit length.
        Args:
            quats: A tensor of shape (..., k*4) or (..., 4).

        Returns:
            A tensor of the same shape as the input but with unit length quaternions.
        """
        assert self.use_quat
        last_dim = quats.get_shape()[-1].value
        ori_shape = tf.shape(quats)
        if last_dim != 4:
            assert last_dim % 4 == 0
            new_shape = tf.concat([ori_shape[:-1], [last_dim // 4, 4]], axis=0)
            quats = tf.reshape(quats, new_shape)
        else:
            quats = quats

        quats_normalized = tf.linalg.l2_normalize(quats, axis=-1)
        quats_normalized = tf.reshape(quats_normalized, ori_shape)
        return quats_normalized

    def build_valid_rot_layer(self, input_):
        """
        Ensures that the given rotations are valid. Can handle quaternion and rotation matrix input.
        Args:
            input_: A tensor of shape (N, seq_length, n_joints*dof) containing the candidate orientations. For
              quaternions `dof` is expected to be 4, otherwise it's expected to be 3*3.

        Returns:
            A tensor of the same shape as `input_` containing valid rotations.
        """
        if self.use_quat:
            # activation function to map to [-1, 1]
            input_t = tf.tanh(input_)

            # monitor the average norm of the quaternions in tensorboard
            qn = tf.reduce_mean(quaternion_norm(input_t))
            tf.summary.scalar(self.mode + "/quat_norm_before", qn, collections=[self.mode + "/model_summary"])

            # now normalize
            return self.normalize_quaternions(input_t)
        elif self.use_aa:
            return input_
        else:
            return self.get_closest_rotmats(input_)

    def build_predictions(self, inputs, output_size, name, share=False):
        """
        Builds dense output layers given the inputs. First, creates a number of hidden layers if set in the config and
        then makes the prediction without applying an activation function.
        Args:
            inputs (tf.Tensor):
            output_size (int):
            name (str):
            share (bool): If true all joints share the same weights.
        Returns:
            (tf.Tensor) prediction.
        """
        hidden_size = self.output_layer_config.get('size', 0)
        num_hidden_layers = self.output_layer_config.get('num_layers', 0)

        prediction = dict()
        current_layer = inputs
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=share or self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer, units=hidden_size, activation=self.activation_fn)

        with tf.variable_scope('out_dense_' + name + "_" + str(num_hidden_layers), reuse=share or self.reuse):
            prediction["mu"] = tf.layers.dense(inputs=current_layer, units=output_size,
                                               activation=self.prediction_activation)

        if self.mle_normal:
            with tf.variable_scope('out_dense_sigma_' + name + "_" + str(num_hidden_layers), reuse=share or self.reuse):
                sigma = tf.layers.dense(inputs=current_layer,
                                        units=output_size,
                                        activation=tf.nn.softplus)
                # prediction["sigma"] = tf.clip_by_value(sigma, 1e-4, 5.0)
                prediction["sigma"] = sigma
        return prediction

    def summary_routines(self):
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        if self.mle_normal:
            tf.summary.scalar(self.mode + "/likelihood", -self.loss, collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/avg_sigma", tf.reduce_mean(self.outputs_sigma), collections=[self.mode + "/model_summary"])
        elif self.use_quat:
            tf.summary.scalar(self.mode + "/loss_quat", self.loss, collections=[self.mode + "/model_summary"])
        else:
            tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/learning_rate",
                              self.learning_rate,
                              collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/gradient_norms",
                              self.gradient_norms,
                              collections=[self.mode + "/model_summary"])

        if self.input_dropout_rate is not None and self.is_training:
            tf.summary.scalar(self.mode + "/input_dropout_rate",
                              self.input_dropout_rate,
                              collections=[self.mode + "/model_summary"])

        if self.prediction_norm is not None:
            tf.summary.scalar(self.mode + "/prediction_norm_before_residual",
                              self.prediction_norm,
                              collections=[self.mode + "/model_summary"])

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
        self.input_layer_size = self.config["input_layer_size"]
        self.states = None

        if self.reuse is False:
            print("Input size is %d" % self.input_size)
            print('rnn_size = {0}'.format(self.rnn_size))

        # === Transform the inputs ===
        with tf.name_scope("inputs"):
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
        if self.config['cell_type'] == C.GRU:
            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
        elif self.config['cell_type'] == C.LSTM:
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
        else:
            raise Exception("Cell not found.")

        if self.input_dropout_rate is not None:
            cell = rnn_cell_extensions.InputDropoutWrapper(cell, self.is_training, self.input_dropout_rate)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_layers)])

        with tf.variable_scope("seq2seq", reuse=self.reuse):
            # === Add space decoder ===
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
            else:
                cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)

            # Add an input layer the residual connection
            if self.input_layer_size is not None and self.input_layer_size > 0:
                cell = rnn_cell_extensions.InputEncoderWrapper(cell, self.input_layer_size, reuse=self.reuse)

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
                with tf.variable_scope("rnn_decoder_cell", reuse=self.reuse):
                    dec_cell = copy.deepcopy(cell)
                
                with tf.variable_scope("basic_rnn_seq2seq"):
                    _, enc_state = tf.contrib.rnn.static_rnn(cell, self.enc_in, dtype=tf.float32)  # Encoder
                    outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(self.dec_in, enc_state, dec_cell, loop_function=loop_function)  # Decoder

            elif self.architecture == "tied":
                outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(self.enc_in, self.dec_in, cell, loop_function=loop_function)
            else:
                raise (ValueError, "Unknown architecture: %s" % self.architecture)

        self.outputs_mu = tf.transpose(tf.stack(outputs), (1, 0, 2))  # (N, seq_length, n_joints*dof)
        self.outputs_mu = self.build_valid_rot_layer(self.outputs_mu)
        self.outputs = self.outputs_mu

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
        decoder_input = np.concatenate([seed_sequence[:, -1:], np.zeros((batch_size, prediction_steps - 1, feature_size))], axis=1)

        prediction = session.run(self.outputs, feed_dict={self.encoder_inputs: encoder_input,
                                                          self.decoder_inputs: decoder_input})
        return prediction


class AGED(Seq2SeqModel):
    """
    Implementation of Adversarial Geometry-Aware Human Motion Prediction:
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangyan_Gui_Adversarial_Geometry-Aware_Human_ECCV_2018_paper.pdf
    """
    def __init__(self,
                 config,
                 data_pl,
                 mode,
                 reuse,
                 dtype=tf.float32,
                 **kwargs):
        super(AGED, self).__init__(config=config, data_pl=data_pl, mode=mode, reuse=reuse,
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

        if self.use_adversarial:
            # Fidelity Discriminator
            # real inputs
            self.fidelity_real = self.fidelity_discriminator(self.prediction_targets, reuse=not self.is_training)
            # fake inputs
            self.fidelity_fake = self.fidelity_discriminator(self.outputs, reuse=True)

            # Continuity Discriminator
            # real inputs
            self.continuity_real = self.continuity_discriminator(self.data_inputs, reuse=not self.is_training)
            # fake inputs (real seed + prediction)
            c_inputs = tf.concat([self.encoder_inputs, self.outputs], axis=1)
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
        self.pred_loss = self.geodesic_loss(self.outputs, self.prediction_targets)

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
        assert self.use_aa

        # must unnormalize before computing the geodesic loss
        pred = self._unnormalize(predictions)
        targ = self._unnormalize(targets)

        pred = tf.reshape(pred, [-1, self.target_seq_len, self.NUM_JOINTS, 3])
        targ = tf.reshape(targ, [-1, self.target_seq_len, self.NUM_JOINTS, 3])

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
        if self.normalization_var is not None:
            return data*self.normalization_var + self.normalization_mean
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
        if self.mle_normal:
            tf.summary.scalar(self.mode + "/likelihood", -self.loss, collections=[self.mode + "/model_summary"])
            tf.summary.scalar(self.mode + "/avg_sigma", tf.reduce_mean(self.outputs_sigma), collections=[self.mode + "/model_summary"])
        elif self.use_quat:
            tf.summary.scalar(self.mode + "/loss_quat", self.loss, collections=[self.mode + "/model_summary"])
        else:
            # total generator loss
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

        if self.input_dropout_rate is not None and self.is_training:
            tf.summary.scalar(self.mode + "/input_dropout_rate",
                              self.input_dropout_rate,
                              collections=[self.mode + "/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")

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
            if self.use_adversarial:
                output_feed = [self.loss,
                               self.summary_update,
                               self.outputs,
                               self.g_param_update,
                               self.d_param_update]
            else:
                output_feed = [self.loss,
                               self.summary_update,
                               self.outputs,
                               self.g_param_update]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs,
                           self.g_loss,
                           self.continuity_loss,
                           self.d_loss]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]


class RNN(BaseModel):
    """
    Autoregressive RNN.
    """
    def __init__(self, config, data_pl, mode, reuse, dtype, **kwargs):
        super(RNN, self).__init__(config, data_pl, mode, reuse, dtype, **kwargs)

        self.cell_config = self.config.get("cell")
        self.input_layer_config = config.get('input_layer', None)

        self.cell = None
        self.initial_states = None
        self.rnn_outputs = None  # Output of RNN layer.
        self.rnn_state = None  # Final state of RNN layer.
        self.inputs_hidden = None

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

    def build_input_layer(self):
        current_layer = self.prediction_inputs
        if self.input_layer_config is not None:
            if self.input_dropout_rate is not None:
                with tf.variable_scope('input_dropout', reuse=self.reuse):
                    current_layer = tf.layers.dropout(current_layer,
                                                      rate=self.input_dropout_rate,
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

    def build_loss(self):
        super(RNN, self).build_loss()

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
            # prediction = np.reshape(get_closest_rotmat(np.reshape(prediction, [-1, 3, 3])), prediction.shape)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)


class VRNN(RNN):
    """
    Variational RNN.
    """
    def __init__(self, config, data_pl, mode, reuse, dtype, **kwargs):
        super(VRNN, self).__init__(config, data_pl, mode, reuse, dtype, **kwargs)

        self.latent_cell_loss = dict()  # Loss terms (i.e., kl-divergence) created by the latent cell.

        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len
        else:
            self.sequence_length = self.target_seq_len
        self.prediction_inputs = self.data_inputs
        self.prediction_targets = self.data_inputs
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

    def create_cell(self):
        return LatentCell.get(self.cell_config["type"],
                              self.cell_config,
                              self.mode,
                              self.reuse,
                              global_step=self.global_step)

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
            self.prediction_representation = self.rnn_outputs[-1]

        self.cell.register_sequence_components(self.rnn_outputs)
        self.build_output_layer()

    def build_loss(self):
        super(VRNN, self).build_loss()

        # KLD Loss.
        if self.is_training:
            loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.prediction_seq_len, dtype=tf.float32), -1)
            self.latent_cell_loss = self.cell.build_loss(loss_mask, tf.reduce_mean)
            for loss_key, loss_op in self.latent_cell_loss.items():
                self.loss += loss_op

    def summary_routines(self):
        for loss_key, loss_op in self.latent_cell_loss.items():
            tf.summary.scalar(str(loss_key), loss_op, collections=[self.mode + "/model_summary"])
        tf.summary.scalar(self.mode + "/kld_weight", self.cell.kld_weight, collections=[self.mode + "/model_summary"])

        super(VRNN, self).summary_routines()


class ASimpleYetEffectiveBaseline(Seq2SeqModel):
    """
    Does not predict anything, but just repeats the last known frame for as many frames necessary.
    """
    def __init__(self, config, data_pl, mode, reuse, dtype, **kwargs):
        super(ASimpleYetEffectiveBaseline, self).__init__(config, data_pl, mode, reuse, dtype, **kwargs)

    def build_network(self):
        # don't do anything, just repeat the last known pose
        last_known_pose = self.decoder_inputs[:, 0:1]
        self.outputs_mu = tf.tile(last_known_pose, [1, self.target_seq_len, 1])
        self.outputs = self.outputs_mu
        # dummy variable
        self._dummy = tf.Variable(0.0, name="imadummy")

    def build_loss(self):
        d = self._dummy - self._dummy
        self.loss = tf.reduce_mean(tf.reduce_sum(d*d))
