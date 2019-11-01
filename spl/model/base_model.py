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


A base model class to provide an interface between the models and training/evaluation pipeline. The base class
provides implements the basic functionality as well.

- We assume that experiment configuration including data files, hyper-parameters, architecture details, etc. are stored
in a dictionary (i.e., config) and passed through different modules of the pipeline. Each module reads only the
necessary information. Although we didn't pass it read-only, we assume that the config is not modified!

- Data is passed via a number of placeholders (i.e., data_pl) which we create by using tf.data. Our dataset class
(i.e., spl.data.amass_tf.TFRecordMotionDataset) wraps tf.data API to implement the required preprocessing and
normalization operations. Hence, the data is passed automatically. We no longer need feed_dict during training.
You can see `sampled_step` method implementation to do autoregressive sampling. Similarly, one can use tf.placeholder
instead of tf.data. Is should be straightforward. You just need to create placeholders and pass them within a
dictionary. There are 4 data placeholders in `data_pl` dictionary:
    - inputs
    - targets
    - seq_len
    - id
    where inputs and targets are the same since we pass the full sequence. In models, the seed and target sequences are
    determined depending on the functionality. `id` is required to keep track of the source dataset or evaluate a
    particular motion sample.

Overall functionality is decomposed into a number of methods to enable code reusing as much as possible. Every
trainable model class is expected to implement the following methods:
    - build_network
    - step
    - sampled_step
    while build_loss, build_prediction_layer, summary_routines, optimization_routines are implemented by the
    BaseModel class. See spl.model.rnn.RNN as an example.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from common.constants import Constants as C
from spl.model.spl import SPL


class BaseModel(object):
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.config = config
        self.data_placeholders = data_pl
        self.mode = mode
        self.reuse = reuse
        
        self.source_seq_len = config["source_seq_len"]
        self.target_seq_len = config["target_seq_len"]
        self.batch_size = config["batch_size"]
        
        # Data placeholders. Output of dataset iterator.
        self.data_inputs = data_pl[C.BATCH_INPUT]
        self.data_targets = data_pl[C.BATCH_TARGET]
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]
        self.data_ids = data_pl[C.BATCH_ID]

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN
        
        # Dataset and data representation.
        self.use_quat = config['data_type'] == C.QUATERNION
        self.use_aa = config['data_type'] == C.ANGLE_AXIS
        self.use_rotmat = config['data_type'] == C.ROT_MATRIX
        self.use_h36m = config.get("use_h36m", False)
        
        # Architecture.
        self.residual_velocity = config.get("residual_velocity", None)  # True or False
        self.loss_type = config.get("loss_type", None)  # all_mean or joint_sum
        self.joint_prediction_layer = config.get("joint_prediction_layer", None)  # plain, spl or spl_sparse
        self.activation_fn = tf.nn.relu

        # Set by the child model classes.
        self.outputs = None  # List of predicted frames. Set by `build_graph`.
        self.prediction_targets = None  # Targets in pose loss term.
        self.prediction_inputs = None  # Inputs that are used to make predictions.
        self.loss_all_frames = None  # Whether to apply training loss on the seed sequence or not.
        
        # Training
        self.loss = None  # Loss op to be used in training. Set by `build_graph`.
        self.gradient_norms = None  # Set by `optimization_routines`.
        self.parameter_update = None  # Parameter update op: optimizer output. Set by `optimization_routines`.
        self.summary_update = None  # Summary op to write summaries. Set by `summary_routines`.
        
        # Hard-coded parameters.
        self.JOINT_SIZE = 4 if self.use_quat else 3 if self.use_aa else 9
        self.NUM_JOINTS = 21 if self.use_h36m else 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

    def build_graph(self):
        """Creates Tensorflow training graph by building the actual network and calculating the loss operation."""
        self.outputs = self.build_network()
        self.loss = self.build_loss()

    def build_network(self):
        """Builds the network.
        
        Returns:
            output op.
        """
        pass

    def step(self, session):
        """Runs one training step by evaluating loss, parameter update, summary and output operations.
        
        Model receives data from the data pipeline automatically. In contrast to `sampled_step`, model's output is not
        fed back to the model.
        Args:
            session: TF session object.
        Returns:
            loss, summary proto, prediction
        """
        pass

    def sampled_step(self, session):
        """Runs an auto-regressive sampling step. It is used to evaluate the model.
        
        In contrast to `step`, predicted output step is fed back to the model to predict the next step.
        Args:
            session: TF session object.
        Returns:
            predicted sequence, actual target, input sequence, sample's data_id
        """
        pass
    
    def build_loss(self):
        """Calculates the loss between the predicted and ground-truth sequences.
        
        Some models (i.e., rnn) evaluate the prediction on the entire sequence while some (i.e., seq2seq) ignores the
        seed pose. If not training, we evaluate all models only on the target pose.
        Returns:
            loss op.
        """
        if self.is_eval or not self.loss_all_frames:
            predicted_pose = self.outputs[:, -self.target_seq_len:, :]
            target_pose = self.prediction_targets[:, -self.target_seq_len:, :]
            seq_len = self.target_seq_len
        else:
            predicted_pose = self.outputs
            target_pose = self.prediction_targets
            seq_len = tf.shape(self.outputs)[1]

        with tf.name_scope("loss_angles"):
            diff = target_pose - predicted_pose
            if self.loss_type == C.LOSS_POSE_ALL_MEAN:
                pose_loss = tf.reduce_mean(tf.square(diff))
                loss_ = pose_loss
            elif self.loss_type == C.LOSS_POSE_JOINT_SUM:
                per_joint_loss = tf.reshape(tf.square(diff), (-1, seq_len, self.NUM_JOINTS, self.JOINT_SIZE))
                per_joint_loss = tf.sqrt(tf.reduce_sum(per_joint_loss, axis=-1))
                per_joint_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                per_joint_loss = tf.reduce_mean(per_joint_loss)
                loss_ = per_joint_loss
            else:
                raise Exception("Unknown loss.")
            return loss_
    
    def build_prediction_layer(self, inputs):
        """Given a context representation (i.e., rnn outputs), makes pose prediction by either using structured
        prediction layer (SPL) or standard dense layer.
        
        Args:
            inputs: A tensor or (batch_size, seq_len, representation_size)
        Returns:
            predicted pose sequence: A tensor or (batch_size, seq_len, pose_size)
        """
        if self.joint_prediction_layer == "plain":
            # Create a number of hidden layers and predict the full pose vector.
            with tf.variable_scope('output_layer', reuse=self.reuse):
                hidden_layers = self.config.get("output_hidden_layers", 0)
                current_layer = inputs
                for layer_idx in range(hidden_layers):
                    with tf.variable_scope('out_dense_all_' + str(layer_idx), reuse=self.reuse):
                        current_layer = tf.layers.dense(inputs=current_layer, units=self.config["output_hidden_size"],
                                                        activation=tf.nn.relu)
                with tf.variable_scope('out_dense_all_' + str(hidden_layers), reuse=self.reuse):
                    pose_prediction = tf.layers.dense(inputs=current_layer, units=self.HUMAN_SIZE, activation=None)
            
        else:
            # Predict the pose vector by composing a hierarchy of joint specific networks.
            with tf.variable_scope('output_layer', reuse=self.reuse):
                spl_sparse = True if self.joint_prediction_layer == "spl_sparse" else False
                sp_layer = SPL(hidden_layers=self.config["output_hidden_layers"],
                               hidden_units=self.config["output_hidden_size"],
                               joint_size=self.JOINT_SIZE,
                               sparse=spl_sparse,
                               use_h36m=self.use_h36m,
                               reuse=self.reuse)
                pose_prediction = sp_layer.build(inputs)
        
        if self.residual_velocity:
            pose_prediction += self.prediction_inputs[:, 0:tf.shape(pose_prediction)[1], :self.HUMAN_SIZE]
        return pose_prediction
    
    def optimization_routines(self):
        """Creates and optimizer, applies gradient regularizations and sets the parameter_update operation."""
        global_step = tf.train.get_global_step(graph=None)
        learning_rate = tf.train.exponential_decay(self.config.get('learning_rate'),
                                                   global_step=global_step,
                                                   decay_steps=self.config.get('learning_rate_decay_steps', 1000),
                                                   decay_rate=self.config.get('learning_rate_decay_rate', 0.98),
                                                   staircase=True)
        if self.config["optimizer"] == C.OPTIMIZER_ADAM:
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.config["optimizer"] == C.OPTIMIZER_SGD:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise Exception("Optimization not found.")

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            # Gradient clipping.
            gradients = tf.gradients(self.loss, params)
            if self.config.get('grad_clip_norm', 0) > 0:
                gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.config.get('grad_clip_norm'))
            else:
                self.gradient_norms = tf.global_norm(gradients)
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=global_step)
    
    def summary_routines(self):
        """Creates Tensorboard summaries."""
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        # Training objective summary.
        tf.summary.scalar(self.mode + "/loss", self.loss, collections=[self.mode + "/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/gradient_norms",
                              self.gradient_norms,
                              collections=[self.mode + "/model_summary"])
        # If you would like to introduce more summaries, first create them and then call parent's method
        # (i.e., this one) because of tf.summary.merge_all. Otherwise, new summaries will not be considered,
        self.summary_update = tf.summary.merge_all(self.mode + "/model_summary")

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
        if from_config is None:
            config = dict()
            config['seed'] = args.seed
            config['model_type'] = args.model_type
            config['data_type'] = args.data_type
            config['use_h36m'] = args.use_h36m
    
            config['no_normalization'] = args.no_normalization
            config['batch_size'] = args.batch_size
            config['source_seq_len'] = args.source_seq_len
            config['target_seq_len'] = args.target_seq_len
            
            config['early_stopping_tolerance'] = args.early_stopping_tolerance
            config['num_epochs'] = args.num_epochs
            
            config['learning_rate'] = args.learning_rate
            config['learning_rate_decay_steps'] = args.learning_rate_decay_steps
            config['learning_rate_decay_rate'] = args.learning_rate_decay_rate
            config['grad_clip_norm'] = args.grad_clip_norm
            config['optimizer'] = args.optimizer
    
            config['input_hidden_layers'] = args.input_hidden_layers
            config['input_hidden_size'] = args.input_hidden_size
            config['input_dropout_rate'] = args.input_dropout_rate
    
            config["cell_type"] = args.cell_type
            config["cell_size"] = args.cell_size
            config["cell_layers"] = args.cell_layers
            
            config['output_hidden_layers'] = args.output_hidden_layers
            config['output_hidden_size'] = args.output_hidden_size
            
            config['residual_velocity'] = args.residual_velocity
            config['loss_type'] = args.loss_type
            config['joint_prediction_layer'] = args.joint_prediction_layer

            config['transformer_lr'] = args.transformer_lr
            config['transformer_d_model'] = args.transformer_d_model
            config['transformer_dropout_rate'] = args.transformer_dropout_rate
            config['transformer_dff'] = args.transformer_dff
            config['transformer_num_layers'] = args.transformer_num_layers
            config['transformer_num_heads_temporal'] = args.transformer_num_heads_temporal
            config['transformer_num_heads_spacial'] = args.transformer_num_heads_spacial
            config['transformer_warm_up_steps'] = args.warm_up_steps
            config['transformer_window_length'] = args.transformer_window_length

        else:
            config = from_config
        
        if args.new_experiment_id is not None:
            config["experiment_id"] = args.new_experiment_id
        else:
            config["experiment_id"] = str(int(time.time()))
        experiment_name_format = "{}-{}-{}_{}-b{}-in{}_out{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        args.model_type,
                                                        "h36m" if args.use_h36m else "amass",
                                                        args.data_type,
                                                        args.batch_size,
                                                        args.source_seq_len,
                                                        args.target_seq_len)
        return config, experiment_name
