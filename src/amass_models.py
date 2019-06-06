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
from tf_models import LatentLayer
from tf_rnn_cells import LatentCell
from tf_loss_quat import quaternion_norm
from tf_loss_quat import quaternion_loss
from tf_loss import logli_normal_isotropic
from tf_rot_conversions import aa2rotmat
from motion_metrics import get_closest_rotmat

import tf_tr_quat
import tf_tr_axisangle
import tf_tr_rotmat


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
        self.use_sparse_fk_joints = config.get('use_sparse_fk_joints', False)

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
            if self.use_quat:
                # TODO(kamanuel) for now use the loss that is equivalent to log(R*R^T)
                loss_type = "quat_l2"
                loss_per_frame = quaternion_loss(targets_pose, predictions_pose, loss_type)
                loss_per_sample = tf.reduce_sum(loss_per_frame, axis=-1)
                loss_per_batch = tf.reduce_mean(loss_per_sample)
                self.loss = loss_per_batch
            else:
                diff = targets_pose - predictions_pose
                if self.angle_loss_type == C.LOSS_POSE_ALL_MEAN:
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
                for joint_key in sorted(self.structure_indexed.keys()):
                    parent_joint_idx, joint_idx, joint_name = self.structure_indexed[joint_key]
                    joint_inputs = [self.prediction_representation]
                    if self.use_sparse_fk_joints:
                        if parent_joint_idx >= 0:
                            joint_inputs.append(self.outputs_mu_joints[parent_joint_idx])
                    else:
                        self.traverse_parents(joint_inputs, parent_joint_idx)
                    self.parse_outputs(self.build_predictions(tf.concat(joint_inputs, axis=-1), self.JOINT_SIZE, joint_name))
            else:
                raise Exception("Prediction model not recognized.")

            self.aggregate_outputs()
            pose_prediction = self.outputs_mu

            # Apply residual connection on the pose only.
            if self.residual_velocities:
                # some debugging
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
        if self.force_valid_rot:
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
        else:
            return rotmats

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

        prediction = dict()
        current_layer = inputs
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=self.reuse):
                current_layer = tf.layers.dense(inputs=current_layer, units=hidden_size, activation=self.activation_fn)

        with tf.variable_scope('out_dense_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
            prediction["mu"] = tf.layers.dense(inputs=current_layer, units=output_size,
                                               activation=self.prediction_activation)

        if self.mle_normal:
            with tf.variable_scope('out_dense_sigma_' + name + "_" + str(num_hidden_layers), reuse=self.reuse):
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
        # generator loss uses the geodesic loss
        self.g_loss = self.geodesic_loss(self.outputs, self.prediction_targets)

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
            self.d_loss = self.d_weight * (self.continuity_loss + self.fidelity_loss)
            self.loss = self.g_loss + self.d_loss
        else:
            self.loss = self.g_loss

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
            tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])
            tf.summary.scalar(self.mode+"/g_loss", self.g_loss, collections=[self.mode + "/model_summary"])
            if self.use_adversarial:
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
