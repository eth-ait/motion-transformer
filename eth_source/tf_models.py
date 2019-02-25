import tensorflow as tf
import numpy as np
import sys
import time
import math
import copy
import tf_loss
from tf_model_utils import get_reduce_loss_func, get_rnn_cell, linear, fully_connected_layer, get_activation_fn, get_decay_variable
from constants import Constants
from tf_rnn_cells import LatentCell, VRNNCell, ZForcingCell

"""
Vanilla variational recurrent neural network model.
The model is trained by using negative log-likelihood (reconstruction) and KL-divergence losses.
Assuming that model outputs are isotropic Gaussian distributions.

Model functionality is decomposed into basic functions (see build_graph method) so that variants of the model can easily
be constructed by inheriting from this vanilla architecture.

Note that different modes (i.e., training, validation, sampling) should be implemented as different graphs by reusing
the parameters. Therefore, validation functionality shouldn't be used by a model with training mode.
"""
C = Constants()


class LatentLayer(object):
    """
    Base class for latent layers.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        self.config = config
        self.reuse = reuse
        assert mode in [C.TRAIN, C.VALID, C.EVAL, C.SAMPLE]
        self.mode = mode
        self.is_sampling = mode == C.SAMPLE
        self.is_validation = mode in [C.VALID, C.EVAL]
        self.is_training = mode == C.TRAIN
        self.is_eval = mode == C.EVAL  # Similar to the validation mode, returns some details for analysis.
        self.layer_structure = config.get("layer_structure")
        self.layer_fc = self.layer_structure == C.LAYER_FC
        self.layer_tcn = self.layer_structure == C.LAYER_TCN
        self.global_step = kwargs.get("global_step", None)

        self.ops_loss = dict()
        self.summary_ops = dict()

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Given the inputs for approximate posterior and prior, builds corresponding latent distributions.
        Inserts latent ops into main model's containers. See BaseTemporalModel for details.
        Args:
            q_input: inputs for approximate posterior.
            p_input: inputs for prior.
            output_ops_dict:
            eval_ops_dict:
            summary_ops_dict
        Returns:
            A latent sample drawn from Q or P based on mode. In sampling mode, the sample is drawn from prior.
        """
        raise NotImplementedError('subclasses must override sample method')

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
        """
        Builds loss terms related with latent space.
        Args:
            sequence_mask: mask to be applied on variable-length sequences.
            reduce_loss_fn: function to get final loss value, i.e., average or sum.
            loss_ops_dict: container keeping loss terms.
        Returns:
            A dictionary of loss terms.
        """
        raise NotImplementedError('subclasses must override sample method')

    @staticmethod
    def build_tcn_layer(input_layer, num_latent_units, latent_activation_fn, kernel_size, dilation, num_hidden_layers, num_hidden_units, is_training):
        """
        Args:
            input_layer:
            num_latent_units:
            latent_activation_fn:
            kernel_size:
            dilation:
            num_hidden_layers:
            num_hidden_units:
            is_training:
        Returns:
        """
        # Whether to applies zero padding on the inputs or not. If kernel_size > 1 or dilation > 1, it needs to be True.
        zero_padding = True if kernel_size > 1 or dilation > 1 else False
        current_layer = [input_layer]
        for i in range(num_hidden_layers):
            current_layer = CCN.temporal_block_ccn(input_layer=current_layer[0], num_filters=num_hidden_units,
                                                   kernel_size=kernel_size, dilation=dilation, activation_fn=None,
                                                   num_extra_conv=0, use_gate=True, use_residual=False,
                                                   zero_padding=zero_padding)

        current_layer = CCN.temporal_block_ccn(input_layer=current_layer[0], num_filters=num_latent_units,
                                               kernel_size=kernel_size, dilation=dilation, activation_fn=None,
                                               num_extra_conv=0, use_gate=True, use_residual=False,
                                               zero_padding=zero_padding)

        layer = current_layer[0] if latent_activation_fn is None else latent_activation_fn(current_layer[0])
        flat_layer = tf.reshape(layer, [-1, num_latent_units])
        return layer, flat_layer

    @staticmethod
    def build_conv1_layer(input_layer, num_latent_units, latent_activation_fn, num_hidden_layers, num_hidden_units, hidden_activation_fn, is_training):
        current_layer = input_layer
        for i in range(num_hidden_layers):
            current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                             filters=num_hidden_units, dilation_rate=1,
                                             activation=hidden_activation_fn)

        current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                         filters=num_latent_units, dilation_rate=1,
                                         activation=latent_activation_fn)

        flat_layer = tf.reshape(current_layer, [-1, num_latent_units])
        return current_layer, flat_layer

    @staticmethod
    def build_fc_layer(input_layer, num_latent_units, latent_activation_fn, num_hidden_layers, num_hidden_units, hidden_activation_fn, is_training):
        flat_input = tf.reshape(input_layer, [-1, input_layer.shape.as_list()[-1]])
        flat_hidden = fully_connected_layer(input_layer=flat_input,
                                            is_training=is_training,
                                            activation_fn=hidden_activation_fn,
                                            num_layers=num_hidden_layers,
                                            size=num_hidden_units)
        flat_layer = linear(input_layer=flat_hidden,
                            output_size=num_latent_units,
                            activation_fn=latent_activation_fn,
                            is_training=is_training)
        layer = tf.reshape(flat_layer, [tf.shape(input_layer)[0], -1, num_latent_units])
        return layer, flat_layer

    @staticmethod
    def get(layer_type, config, mode, reuse, **kwargs):
        """
        Creates latent layer.
        Args:
            layer_type (str): Type of layer.
            config:
            reuse:
            mode:
        Returns:
            An instance of LatentLayer.
        """
        if layer_type == C.LATENT_GAUSSIAN:
            return GaussianLatentLayer(config, mode, reuse, **kwargs)
        elif layer_type == C.LATENT_VARIATIONAL_CODEBOOK:
            return VariationalCodebook(config, mode, reuse, **kwargs)
        elif layer_type == C.LATENT_STOCHASTIC_CODEBOOK:
            return StochasticCodebook(config, mode, reuse, **kwargs)
        elif layer_type == C.LATENT_LADDER_GAUSSIAN:
            return LadderLatentLayer(config, mode, reuse, **kwargs)
        elif layer_type == C.LATENT_CATEGORICAL:
            raise NotImplementedError('Not implemented')
        else:
            raise Exception("Unknown latent layer.")


class VariationalCodebook(LatentLayer):
    """
    Latent space with a codebook of embedding vectors where the encoder maps the inputs to the most relevant
    representation. It is unsupervised. Hence, both the encoder and embedding vectors are learned.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(VariationalCodebook, self).__init__(config, mode, reuse, )

        self.latent_num_components = self.config.get('latent_num_components')
        self.latent_size_components = self.config.get('latent_size')
        self.latent_divisive_normalization = self.config.get('latent_divisive_normalization', False)
        self.use_reinforce = self.config.get('use_reinforce', False)
        self.use_temporal_kld = self.config.get('use_temporal_kld', False)
        self.tkld_weight = self.config.get('tkld_weight', 0.1)
        self.kld_weight = self.config.get('kld_weight', 0.5)
        if not self.is_training:
            self.kld_weight = 1.0
        self.loss_diversity_weight = self.config.get('loss_diversity_weight', 1)
        self.loss_diversity_batch_weight = self.config.get('loss_diversity_batch_weight', 1)

        # Parameters of codebook.
        with tf.variable_scope("latent_codebook", reuse=self.reuse):
            self.codebook_mu = tf.get_variable(name="codebook_mu", dtype=tf.float32, initializer=tf.random_uniform([self.latent_num_components, self.latent_size_components], -1.0, 1.0))

        # Logits of approximate posterior and prior categorical distribution.
        self.q_pi = None
        self.p_pi = None
        self.q_pi_probs = None
        self.p_pi_probs = None
        self.pi_sample = None  # Indices of selected components.

        # Approximate posterior and prior categorical distribution objects.
        self.dist_q_pi = None
        self.dist_p_pi = None

        self.batch_size = None

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions.
        """
        loss_ops_dict = loss_ops_dict or dict()
        self.batch_size = tf.shape(self.p_pi)[0]
        loss_key = "loss_kld"
        with tf.name_scope("kld_loss"):
            temporal_kld_cat_loss = tf_loss.kld_bernoulli(self.q_pi_probs, self.p_pi_probs)
            self.ops_loss[loss_key] = self.kld_weight*reduce_loss_fn(sequence_mask*temporal_kld_cat_loss)
            loss_ops_dict[loss_key] = self.ops_loss[loss_key]

        if self.is_training:
            if self.use_reinforce:
                loss_key = "loss_reinforce"
                # reward = kwargs["reward"] - self.ops_loss["loss_kld"]
                reward = kwargs["reward"]
                logprob_loss = tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(tf.to_int32(self.pi_sample), axis=-1), logits=self.q_pi), axis=-1)
                self.ops_loss[loss_key] = reduce_loss_fn(sequence_mask*logprob_loss*tf.stop_gradient(reward))
                loss_ops_dict[loss_key] = self.ops_loss[loss_key]

            flat_q_pi = tf.reshape(self.q_pi_probs, [-1, self.latent_num_components])
            """
            if self.latent_divisive_normalization and self.loss_diversity_weight > 0:
                loss_key = "loss_diversity"
                with tf.name_scope("diversity_loss"):
                    flat_diversity_loss = tf.reduce_sum(tf.square(self.latent_num_components*flat_q_pi - 1), axis=-1, keepdims=True)
                    temporal_diversity_loss = tf.reshape(flat_diversity_loss, [self.batch_size, -1, 1])
                    self.ops_loss[loss_key] = self.loss_diversity_weight*reduce_loss_fn(sequence_mask*temporal_diversity_loss)
                    loss_ops_dict[loss_key] = self.ops_loss[loss_key]
            """
            if self.loss_diversity_weight > 0:
                loss_key = "loss_entropy_sample"
                with tf.name_scope("sample_entropy_loss"):
                    q_pi_entropy = tf_loss.entropy(flat_q_pi)
                    temporal_entropy_loss = tf.reshape(q_pi_entropy, [self.batch_size, -1, 1])
                    self.ops_loss[loss_key] = self.loss_diversity_weight*reduce_loss_fn(
                        sequence_mask*temporal_entropy_loss)
                    loss_ops_dict[loss_key] = self.ops_loss[loss_key]

            if self.loss_diversity_batch_weight > 0:
                loss_key = "loss_entropy_batch"
                with tf.name_scope("batch_entropy_loss"):
                    num_entries = tf.reduce_sum(sequence_mask)
                    batch_q_pi = tf.reduce_sum(sequence_mask*self.q_pi_probs, axis=[0, 1])/num_entries
                    batch_q_pi_entropy = tf.reduce_sum(tf_loss.entropy(tf.expand_dims(batch_q_pi, axis=0)))
                    self.ops_loss[loss_key] = -self.loss_diversity_batch_weight*batch_q_pi_entropy
                loss_ops_dict[loss_key] = self.ops_loss[loss_key]

            if self.use_temporal_kld:
                prior_step = 1
                loss_key = "loss_temporal_kld"
                with tf.name_scope("temporal_kld_loss"):
                    latent_shape = tf.shape(self.q_pi_probs)
                    p_pi_part = tf.ones([latent_shape[0], prior_step, latent_shape[2]], name="temp_p_pi")/self.latent_num_components
                    q_pi_part = self.q_pi_probs[:, 0:-prior_step, :]
                    temp_p_pi = tf.concat([p_pi_part, q_pi_part], axis=1)

                    temporal_tkld_cat_loss = tf_loss.kld_bernoulli(self.q_pi_probs, tf.stop_gradient(temp_p_pi))
                    self.ops_loss[loss_key] = self.tkld_weight*reduce_loss_fn(sequence_mask*temporal_tkld_cat_loss)
                    loss_ops_dict[loss_key] = self.ops_loss[loss_key]

        return self.ops_loss

    def build_categorical_dist(self, p_input, q_input):
        """
        Auxiliary method to build logit layers.
        Returns:
            Flattened prior and approximate posterior logits.
        """
        flat_p_pi, flat_q_pi = None, None

        if self.latent_divisive_normalization:
            activation_fn = get_activation_fn(C.RELU)
        else:
            activation_fn = None

        with tf.variable_scope('prior', reuse=self.reuse):
            with tf.variable_scope('p_pi', reuse=self.reuse):
                if self.layer_tcn:
                    self.p_pi, flat_p_pi = LatentLayer.build_tcn_layer(input_layer=p_input,
                                                                       num_latent_units=self.latent_num_components,
                                                                       latent_activation_fn=activation_fn,
                                                                       kernel_size=self.config['latent_filter_size'],
                                                                       dilation=self.config['latent_dilation'],
                                                                       zero_padding=True, is_training=self.is_training)
                elif self.layer_fc:
                    self.p_pi, flat_p_pi = LatentLayer.build_fc_layer(input_layer=p_input,
                                                                      num_latent_units=self.latent_num_components,
                                                                      latent_activation_fn=activation_fn,
                                                                      num_hidden_layers=self.config["num_hidden_layers"],
                                                                      num_hidden_units=self.config["num_hidden_units"],
                                                                      hidden_activation_fn=self.config["hidden_activation_fn"],
                                                                      is_training=self.is_training)
                if self.latent_divisive_normalization:
                    self.p_pi_probs = self.p_pi / tf.maximum(tf.reduce_sum(self.p_pi, axis=-1, keepdims=True), 1e-6)
                else:
                    self.p_pi_probs = tf.nn.softmax(self.p_pi)

        with tf.variable_scope('approximate_posterior', reuse=self.reuse):
            with tf.variable_scope('q_pi', reuse=self.reuse):
                if self.layer_tcn:
                    self.q_pi, flat_q_pi = LatentLayer.build_tcn_layer(input_layer=q_input,
                                                                       num_latent_units=self.latent_num_components,
                                                                       latent_activation_fn=activation_fn,
                                                                       kernel_size=self.config['latent_filter_size'],
                                                                       dilation=self.config['latent_dilation'],
                                                                       zero_padding=False, is_training=self.is_training)
                elif self.layer_fc:
                    self.q_pi, flat_q_pi = LatentLayer.build_fc_layer(input_layer=q_input,
                                                                      num_latent_units=self.latent_num_components,
                                                                      latent_activation_fn=activation_fn,
                                                                      num_hidden_layers=self.config["num_hidden_layers"],
                                                                      num_hidden_units=self.config["num_hidden_units"],
                                                                      hidden_activation_fn=self.config["hidden_activation_fn"],
                                                                      is_training=self.is_training)
                if self.latent_divisive_normalization:
                    self.q_pi_probs = self.q_pi / tf.maximum(tf.reduce_sum(self.q_pi, axis=-1, keepdims=True), 1e-6)
                else:
                    self.q_pi_probs = tf.nn.softmax(self.q_pi)
        return flat_p_pi, flat_q_pi

    def build_latent_sample(self, flat_p_pi, flat_q_pi):
        """
        Maps prior or approximate posterior logits to the latent representation.
        Args:
            flat_p_pi:
            flat_q_pi:
        Returns:
            Latent representation and its index.
        """
        with tf.variable_scope('z', reuse=self.reuse):
            if self.is_sampling:
                if self.use_reinforce:
                    flat_latent_sample = VariationalCodebook.draw_deterministic_latent_sample(flat_p_pi, self.codebook_mu)
                    pi_sample = tf.expand_dims(tf.argmax(flat_p_pi, axis=-1), axis=-1)
                else:
                    flat_latent_sample, pi_sample = VariationalCodebook.draw_deterministic_latent_sample_reinforce(flat_p_pi, self.codebook_mu)
            else:
                if self.use_reinforce:
                    flat_latent_sample = VariationalCodebook.draw_deterministic_latent_sample(flat_q_pi, self.codebook_mu)
                    pi_sample = tf.expand_dims(tf.argmax(flat_q_pi, axis=-1), axis=-1)
                else:
                    flat_latent_sample, pi_sample = VariationalCodebook.draw_deterministic_latent_sample_reinforce(flat_q_pi, self.codebook_mu)

            pi_sample = tf.reshape(pi_sample, [self.batch_size, -1, 1])
            latent_sample_z = tf.reshape(flat_latent_sample, [self.batch_size, -1, self.latent_size_components])

        return latent_sample_z, pi_sample

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.

        Note that zero_padding is not applied for approximate posterior (i.e., q). In order to have the same length
        outputs with inputs, zero padding should be applied on q_input beforehand.
        """
        self.batch_size = tf.shape(q_input)[0]
        flat_p_pi, flat_q_pi = self.build_categorical_dist(p_input, q_input)
        latent_sample_z, self.pi_sample = self.build_latent_sample(flat_p_pi, flat_q_pi)

        # Register latent ops and summaries.
        if output_ops_dict is not None:
            output_ops_dict[C.Q_PI] = self.q_pi_probs
            output_ops_dict[C.P_PI] = self.p_pi_probs
        if eval_ops_dict is not None:
            if not self.is_sampling:
                eval_ops_dict[C.Q_PI] = self.q_pi_probs
            eval_ops_dict[C.P_PI] = self.p_pi_probs
        if summary_ops_dict is not None:
            summary_ops_dict["mean_entropy_"+C.Q_PI] = tf.reduce_mean(tf_loss.entropy(tf.reshape(self.q_pi_probs, [-1, self.latent_num_components])))
            summary_ops_dict["mean_entropy_"+C.P_PI] = tf.reduce_mean(tf_loss.entropy(tf.reshape(self.p_pi_probs, [-1, self.latent_num_components])))
            summary_ops_dict["mean_codebook_mu"] = tf.reduce_mean(self.codebook_mu)
            summary_ops_dict["mean_diversity_" + C.Q_PI] = tf.reduce_mean(tf.reduce_sum(tf.square(self.latent_num_components*flat_q_pi - 1), axis=-1))
            tf.summary.histogram("pi_sample", self.pi_sample, collections=[self.mode + '_summary_plot'])

        return latent_sample_z

    @staticmethod
    @tf.custom_gradient
    def draw_deterministic_latent_sample(pi, codebook_mu):
        """
        Draws a latent sample and implements a custom gradient for non-differentiable argmax operation.
        Args:
            pi: logits or probability vector with shape of (batch_size, num_components)
            codebook_mu: components of codebook.
        Returns:
            A latent representation indexed by pi.
        """
        num_components = tf.shape(codebook_mu)[0]
        code_idx = tf.expand_dims(tf.argmax(pi, axis=-1), axis=-1)
        z_sample = tf.gather_nd(codebook_mu, code_idx)

        def grad(z_grad):
            """
            Calculates a custom gradient for the argmax operator, and gradients for latent representations through the
            reparameterization trick.
            """
            reduced_grad = tf.reduce_mean(z_grad, axis=-1, keepdims=True)
            pi_grad = reduced_grad*tf.one_hot(tf.argmax(pi, axis=-1), depth=num_components, axis=-1)

            codebook_mu_grad = tf.IndexedSlices(values=z_grad, indices=code_idx[:, 0])
            return pi_grad, codebook_mu_grad

        return z_sample, grad

    @staticmethod
    def draw_deterministic_latent_sample_reinforce(pi, codebook_mu):
        """
        Draws a latent sample.
        Args:
            pi: categorical distribution.
            codebook_mu: mu components of Gaussian codebook.
        Returns:
            A latent representation indexed by pi.
        """
        code_idx = tf.multinomial(logits=pi, num_samples=1, name=None, output_dtype=tf.int32)
        z_sample = tf.gather_nd(codebook_mu, code_idx)
        return z_sample, code_idx


class StochasticCodebook(VariationalCodebook):
    """
    Latent space consists of embeddings either modeled by probabilistic Gaussian distributions with diagonal covariance
    or deterministic representation vectors.

    The encoder maps the inputs to the most relevant representation component. It is unsupervised. Hence, both the
    encoder and latent representations are learned.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(StochasticCodebook, self).__init__(config, mode, reuse, )

        # sigma parameters of codebook.
        with tf.variable_scope("latent_codebook", reuse=self.reuse):
            self.codebook_sigma = tf.get_variable(name="codebook_sigma", dtype=tf.float32, initializer=tf.constant_initializer(0.1), shape=[self.latent_num_components, self.latent_size_components])

    def build_latent_sample(self, flat_p_pi, flat_q_pi):
        with tf.variable_scope('z', reuse=self.reuse):
            if self.is_sampling:
                if self.use_reinforce:
                    flat_latent_sample, pi_sample = StochasticCodebook.draw_stochastic_latent_sample_reinforce(flat_p_pi, self.codebook_mu, self.codebook_sigma)
                else:
                    flat_latent_sample = StochasticCodebook.draw_stochastic_latent_sample(flat_p_pi, self.codebook_mu, self.codebook_sigma)
                    pi_sample = tf.expand_dims(tf.argmax(flat_p_pi, axis=-1), axis=-1)
            else:
                if self.use_reinforce:
                    flat_latent_sample, pi_sample = StochasticCodebook.draw_stochastic_latent_sample_reinforce(flat_q_pi, self.codebook_mu, self.codebook_sigma)
                else:
                    flat_latent_sample = StochasticCodebook.draw_stochastic_latent_sample(flat_q_pi, self.codebook_mu, self.codebook_sigma)
                    pi_sample = tf.expand_dims(tf.argmax(flat_q_pi, axis=-1), axis=-1)

            pi_sample = tf.reshape(pi_sample, [self.batch_size, -1, 1])
            latent_sample_z = tf.reshape(flat_latent_sample, [self.batch_size, -1, self.latent_size_components])

            return latent_sample_z, pi_sample

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.

        Note that zero_padding is not applied for approximate posterior (i.e., q). In order to have the same length
        outputs with inputs, zero padding should be applied on q_input beforehand.
        """
        latent_sample_z = super(StochasticCodebook, self).build_latent_layer(q_input, p_input, output_ops_dict,
                                                                           eval_ops_dict, summary_ops_dict)
        # Register latent ops and summaries.
        if summary_ops_dict is not None:
            summary_ops_dict["mean_codebook_sigma"] = tf.reduce_mean(self.codebook_sigma)

        return latent_sample_z

    @staticmethod
    @tf.custom_gradient
    def draw_stochastic_latent_sample(pi, codebook_mu, codebook_sigma):
        """
        Draws a latent sample and implements a custom gradient for non-differentiable argmax operation.
        Args:
            pi: logits or probability vector with shape of (batch_size, num_components)
            codebook_mu: mu components of Gaussian codebook.
            codebook_sigma: sigma components of Gaussian codebook.
        Returns:
            A latent sample drawn from the Gaussian distribution selected by pi.
        """
        num_components = tf.shape(codebook_mu)[0]
        component_size = tf.shape(codebook_mu)[1]
        code_idx = tf.expand_dims(tf.argmax(pi, axis=-1), axis=-1)
        mu = tf.gather_nd(codebook_mu, code_idx)
        sigma = tf.gather_nd(codebook_sigma, code_idx)

        eps = tf.random_normal(tf.stack([tf.shape(pi)[0], component_size]), 0.0, 1.0, dtype=tf.float32)
        z_sample = tf.add(mu, tf.multiply(sigma, eps))

        def grad(z_grad):
            """
            Calculates a custom gradient for the argmax operator, and gradients for mu and sigma through the
            reparameterization trick.
            """
            reduced_grad = tf.reduce_mean(z_grad, axis=-1, keepdims=True)
            pi_grad = reduced_grad*tf.one_hot(tf.argmax(pi, axis=-1), depth=num_components, axis=-1)
            # reduced_grad = tf.reduce_sum(z_grad, axis=-1, keepdims=True)
            # mixture_idx_grad = reduced_grad*pi
            # mixture_idx_grad = 1.0*tf.one_hot(tf.argmax(pi, axis=-1), depth=num_components, axis=-1)

            codebook_mu_grad = tf.IndexedSlices(values=z_grad, indices=code_idx[:, 0])
            codebook_sigma_grad = tf.IndexedSlices(values=z_grad*eps, indices=code_idx[:, 0])
            return pi_grad, codebook_mu_grad, codebook_sigma_grad

        return z_sample, grad

    @staticmethod
    def draw_stochastic_latent_sample_reinforce(pi, codebook_mu, codebook_sigma):
        """
        Draws a latent sample and implements a custom gradient for non-differentiable argmax operation.
        Args:
            pi: categorical distribution.
            codebook_mu: mu components of Gaussian codebook.
            codebook_sigma: sigma components of Gaussian codebook.
        Returns:
            A latent sample drawn from the Gaussian distribution selected by pi.
        """
        component_size = tf.shape(codebook_mu)[1]
        code_idx = tf.multinomial(logits=pi, num_samples=1, name=None, output_dtype=tf.int32)

        mu = tf.gather_nd(codebook_mu, code_idx)
        sigma = tf.gather_nd(codebook_sigma, code_idx)

        eps = tf.random_normal(tf.stack([tf.shape(pi)[0], component_size]), 0.0, 1.0, dtype=tf.float32)
        z_sample = tf.add(mu, tf.multiply(sigma, eps))

        return z_sample, code_idx


class GaussianLatentLayer(LatentLayer):
    """
    VAE latent space for time-series data, modeled by a Gaussian distribution with diagonal covariance matrix.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(GaussianLatentLayer, self).__init__(config, mode, reuse, )

        self.use_temporal_kld = self.config.get('use_temporal_kld', False)
        self.tkld_weight = self.config.get('tkld_weight', 0.1)
        self.kld_weight = self.config.get('kld_weight', 0.5)
        if not self.is_training:
            self.kld_weight = 1.0

        # Latent space components.
        self.p_mu = None
        self.q_mu = None
        self.p_sigma = None
        self.q_sigma = None

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions. If use_temporal_kld is True,
        then creates another KL-divergence term between consecutive approximate posteriors in time.
        """
        loss_key = "loss_kld"
        with tf.name_scope("kld_loss"):
            self.ops_loss[loss_key] = self.kld_weight*reduce_loss_fn(
                sequence_mask*tf_loss.kld_normal_isotropic(self.q_mu,
                                                           self.q_sigma,
                                                           self.p_mu,
                                                           self.p_sigma,
                                                           reduce_sum=False))
            loss_ops_dict[loss_key] = self.ops_loss[loss_key]

        if self.is_training and self.use_temporal_kld:
            prior_step = 1
            latent_shape = tf.shape(self.q_sigma)

            p_mu_part = tf.zeros([latent_shape[0], prior_step, latent_shape[2]], name="p_mu")
            p_sigma_part = tf.ones([latent_shape[0], prior_step, latent_shape[2]], name="p_sigma")
            q_mu_part = self.q_mu[:, 0:-prior_step, :]
            q_sigma_part = self.q_sigma[:, 0:-prior_step, :]
            temp_p_mu = tf.concat([p_mu_part, q_mu_part], axis=1)
            temp_p_sigma = tf.concat([p_sigma_part, q_sigma_part], axis=1)

            loss_key = "loss_temporal_kld"
            with tf.name_scope("temporal_kld_loss"):
                self.ops_loss[loss_key] = self.tkld_weight*reduce_loss_fn(
                    sequence_mask*tf_loss.kld_normal_isotropic(self.q_mu,
                                                               self.q_sigma,
                                                               tf.stop_gradient(temp_p_mu),
                                                               tf.stop_gradient(temp_p_sigma),
                                                               reduce_sum=False))
                loss_ops_dict[loss_key] = self.ops_loss[loss_key]

        return self.ops_loss

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.

        Note that zero_padding is not applied for approximate posterior (i.e., q). In order to have the same length
        outputs with inputs, zero padding should be applied on q_input beforehand.
        """
        with tf.variable_scope('prior', reuse=self.reuse):
            with tf.variable_scope('p_mu', reuse=self.reuse):
                if self.layer_tcn:
                    self.p_mu, _ = LatentLayer.build_tcn_layer(input_layer=p_input,
                                                               num_latent_units=self.config['latent_size'],
                                                               latent_activation_fn=None,
                                                               kernel_size=self.config['latent_filter_size'],
                                                               dilation=self.config['latent_dilation'],
                                                               zero_padding=True,
                                                               is_training=self.is_training)
                elif self.layer_fc:
                    self.p_mu, _ = LatentLayer.build_fc_layer(input_layer=p_input,
                                                              num_latent_units=self.config['latent_size'],
                                                              latent_activation_fn=None,
                                                              num_hidden_layers=self.config["num_hidden_layers"],
                                                              num_hidden_units=self.config["num_hidden_units"],
                                                              hidden_activation_fn=self.config["hidden_activation_fn"],
                                                              is_training=self.is_training)
            with tf.variable_scope('p_sigma', reuse=self.reuse):
                if self.layer_tcn:
                    self.p_sigma, _ = LatentLayer.build_tcn_layer(input_layer=p_input,
                                                                  num_latent_units=self.config['latent_size'],
                                                                  latent_activation_fn=tf.exp,
                                                                  kernel_size=self.config['latent_filter_size'],
                                                                  dilation=self.config['latent_dilation'],
                                                                  zero_padding=True, is_training=self.is_training)
                elif self.layer_fc:
                    self.p_sigma, _ = LatentLayer.build_fc_layer(input_layer=p_input,
                                                                 num_latent_units=self.config['latent_size'],
                                                                 latent_activation_fn=tf.exp,
                                                                 num_hidden_layers=self.config["num_hidden_layers"],
                                                                 num_hidden_units=self.config["num_hidden_units"],
                                                                 hidden_activation_fn=self.config["hidden_activation_fn"],
                                                                 is_training=self.is_training)
                if self.config.get('latent_sigma_threshold', 0) > 0:
                    self.p_sigma = tf.clip_by_value(self.p_sigma, 1e-3, self.config.get('latent_sigma_threshold'))

        with tf.variable_scope('approximate_posterior', reuse=self.reuse):
            with tf.variable_scope('q_mu', reuse=self.reuse):
                if self.layer_tcn:
                    self.q_mu, _ = LatentLayer.build_tcn_layer(input_layer=q_input,
                                                               num_latent_units=self.config['latent_size'],
                                                               latent_activation_fn=None,
                                                               kernel_size=self.config['latent_filter_size'],
                                                               dilation=self.config['latent_dilation'],
                                                               zero_padding=False,
                                                               is_training=self.is_training)
                elif self.layer_fc:
                    self.q_mu, _ = LatentLayer.build_fc_layer(input_layer=q_input,
                                                              num_latent_units=self.config['latent_size'],
                                                              latent_activation_fn=None,
                                                              num_hidden_layers=self.config["num_hidden_layers"],
                                                              num_hidden_units=self.config["num_hidden_units"],
                                                              hidden_activation_fn=self.config["hidden_activation_fn"],
                                                              is_training=self.is_training)
            with tf.variable_scope('q_sigma', reuse=self.reuse):
                if self.layer_tcn:
                    self.q_sigma, _ = LatentLayer.build_tcn_layer(input_layer=q_input,
                                                                  num_latent_units=self.config['latent_size'],
                                                                  latent_activation_fn=tf.exp,
                                                                  kernel_size=self.config['latent_filter_size'],
                                                                  dilation=self.config['latent_dilation'],
                                                                  zero_padding=False, is_training=self.is_training)
                elif self.layer_fc:
                    self.q_sigma, _ = LatentLayer.build_fc_layer(input_layer=q_input,
                                                                 num_latent_units=self.config['latent_size'],
                                                                 latent_activation_fn=tf.exp,
                                                                 num_hidden_layers=self.config["num_hidden_layers"],
                                                                 num_hidden_units=self.config["num_hidden_units"],
                                                                 hidden_activation_fn=self.config["hidden_activation_fn"],
                                                                 is_training=self.is_training)
                if self.config.get('latent_sigma_threshold', 0) > 0:
                    self.q_sigma = tf.clip_by_value(self.q_sigma, 1e-3, self.config.get('latent_sigma_threshold'))

        with tf.variable_scope('z', reuse=self.reuse):
            if self.is_sampling:
                eps = tf.random_normal(tf.shape(self.p_sigma), 0.0, 1.0, dtype=tf.float32)
                p_z = tf.add(self.p_mu, tf.multiply(self.p_sigma, eps))
                latent_sample = p_z
            else:
                eps = tf.random_normal(tf.shape(self.q_sigma), 0.0, 1.0, dtype=tf.float32)
                q_z = tf.add(self.q_mu, tf.multiply(self.q_sigma, eps))
                latent_sample = q_z

        # Register latent ops and summaries.
        if output_ops_dict is not None:
            output_ops_dict[C.P_MU] = self.p_mu
            output_ops_dict[C.P_SIGMA] = self.p_sigma
            output_ops_dict[C.Q_MU] = self.q_mu
            output_ops_dict[C.Q_SIGMA] = self.q_sigma
        if eval_ops_dict is not None:
            eval_ops_dict[C.P_MU] = self.p_mu
            eval_ops_dict[C.P_SIGMA] = self.p_sigma
            if not self.is_sampling:
                eval_ops_dict[C.Q_MU] = self.q_mu
                eval_ops_dict[C.Q_SIGMA] = self.q_sigma
        if summary_ops_dict is not None:
            summary_ops_dict["mean_" + C.P_MU] = tf.reduce_mean(self.p_mu)
            summary_ops_dict["mean_" + C.P_SIGMA] = tf.reduce_mean(self.p_sigma)
            summary_ops_dict["mean_" + C.Q_MU] = tf.reduce_mean(self.q_mu)
            summary_ops_dict["mean_" + C.Q_SIGMA] = tf.reduce_mean(self.q_sigma)

        return latent_sample


class LadderLatentLayer(LatentLayer):
    """
    Ladder VAE latent space for time-series data where each step is modeled by a Gaussian distribution with diagonal
    covariance matrix.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(LadderLatentLayer, self).__init__(config, mode, reuse, **kwargs)

        # STCN-dense configuration. Concatenates the samples drawn from all latent variables.
        self.dense_z = self.config.get('dense_z', False)
        # Determines the number of deterministic layers per latent variable.
        self.vertical_dilation = self.config.get('vertical_dilation', 1)
        # Draw a new sample from the approximated posterior whenever needed. Otherwise, draw once and use it every time.
        self.use_same_q_sample = self.config.get('use_same_q_sample', False)
        # Whether the top-most prior is dynamic or not. LadderVAE paper uses standard N(0,I) prior.
        self.use_fixed_pz1 = self.config.get('use_fixed_pz1', False)
        # Prior is calculated by using the deterministic representations at previous step.
        self.dynamic_prior = self.config.get('dynamic_prior', False)
        # Approximate posterior is estimated as a precision weighted update of the prior and initial model predictions.
        self.precision_weighted_update = self.config.get('precision_weighted_update', True)
        # Whether the q distribution is hierarchically updated as in the case of prior or not. In other words, lower
        # q layer uses samples of the upper q layer.
        self.recursive_q = self.config.get('recursive_q', True)
        # Whether we follow top-down or bottom-up hierarchy.
        self.top_down_latents = self.config.get('top_down_latents', True)
        # Network type (i.e., dense, convolutional, etc.) we use to parametrize the latent distributions.
        self.latent_layer_structure = self.config.get('layer_structure', C.LAYER_CONV1)

        # Annealing KL-divergence weight or using fixed weight.
        kld_weight = self.config.get('kld_weight', 1)
        if isinstance(kld_weight, dict) and self.global_step is not None:
            self.kld_weight = get_decay_variable(global_step=self.global_step, config=kld_weight, name="kld_weight")
        else:
            self.kld_weight = kld_weight

        # It is always 1 when we report the loss.
        if not self.is_training:
            self.kld_weight = 1.0

        # Latent space components.
        self.num_d_layers = None  # Total number of deterministic layers.
        self.num_s_layers = None  # Total number of stochastic layers can be different due to the vertical_dilation.
        self.q_approximate = None  # List of approximate q distributions from the recognition network.
        self.q_dists = None  # List of q distributions after updating with p_dists.
        self.p_dists = None  # List of prior distributions.
        self.kld_loss_terms = []  # List of KLD loss term.
        self.latent_samples = []  # List of latent samples.

    def build_latent_dist_conv1(self, input_, idx, scope, reuse):
        with tf.name_scope(scope):
            with tf.variable_scope(scope+'_mu', reuse=reuse):
                mu, flat_mu = LatentLayer.build_conv1_layer(input_layer=input_,
                                                            num_latent_units=self.config['latent_size'][idx],
                                                            latent_activation_fn=None,
                                                            num_hidden_layers=self.config["num_hidden_layers"],
                                                            num_hidden_units=self.config["num_hidden_units"],
                                                            hidden_activation_fn=self.config["hidden_activation_fn"],
                                                            is_training=self.is_training)
            with tf.variable_scope(scope+'_sigma', reuse=reuse):
                sigma, flat_sigma = LatentLayer.build_conv1_layer(input_layer=input_,
                                                                  num_latent_units=self.config['latent_size'][idx],
                                                                  latent_activation_fn=tf.nn.softplus,
                                                                  num_hidden_layers=self.config["num_hidden_layers"],
                                                                  num_hidden_units=self.config["num_hidden_units"],
                                                                  hidden_activation_fn=self.config["hidden_activation_fn"],
                                                                  is_training=self.is_training)
                if self.config.get('latent_sigma_threshold', 0) > 0:
                    sigma = tf.clip_by_value(sigma, 1e-3, self.config.get('latent_sigma_threshold'))
                    flat_sigma = tf.clip_by_value(flat_sigma, 1e-3, self.config.get('latent_sigma_threshold'))

        return (mu, sigma),  (flat_mu, flat_sigma)

    def build_latent_dist_tcn(self, input_, idx, scope, reuse):
        with tf.name_scope(scope):
            with tf.variable_scope(scope + '_mu', reuse=reuse):
                mu, flat_mu = LatentLayer.build_tcn_layer(input_layer=input_,
                                                          num_latent_units=self.config['latent_size'][idx],
                                                          latent_activation_fn=None,
                                                          kernel_size=self.config.get("kernel_size", 1),
                                                          dilation=self.config.get("dilation", 1),
                                                          num_hidden_layers=self.config["num_hidden_layers"],
                                                          num_hidden_units=self.config["num_hidden_units"],
                                                          is_training=self.is_training)
            with tf.variable_scope(scope + '_sigma', reuse=reuse):
                sigma, flat_sigma = LatentLayer.build_tcn_layer(input_layer=input_,
                                                                num_latent_units=self.config['latent_size'][idx],
                                                                latent_activation_fn=tf.nn.softplus,
                                                                kernel_size=self.config.get("kernel_size", 1),
                                                                dilation=self.config.get("dilation", 1),
                                                                num_hidden_layers=self.config["num_hidden_layers"],
                                                                num_hidden_units=self.config["num_hidden_units"],
                                                                is_training=self.is_training)
                if self.config.get('latent_sigma_threshold', 0) > 0:
                    sigma = tf.clip_by_value(sigma, 1e-3, self.config.get('latent_sigma_threshold'))
                    flat_sigma = tf.clip_by_value(flat_sigma, 1e-3, self.config.get('latent_sigma_threshold'))

        return (mu, sigma), (flat_mu, flat_sigma)

    def build_latent_dist_fc(self, input_, idx, scope, reuse):
        with tf.name_scope(scope):
            with tf.variable_scope(scope+'_mu', reuse=reuse):
                mu, flat_mu = LatentLayer.build_fc_layer(input_layer=input_,
                                                         num_latent_units=self.config['latent_size'][idx],
                                                         latent_activation_fn=None,
                                                         num_hidden_layers=self.config["num_hidden_layers"],
                                                         num_hidden_units=self.config["num_hidden_units"],
                                                         hidden_activation_fn=self.config["hidden_activation_fn"],
                                                         is_training=self.is_training)

            with tf.variable_scope(scope+'_sigma', reuse=reuse):
                sigma, flat_sigma = LatentLayer.build_fc_layer(input_layer=input_,
                                                               num_latent_units=self.config['latent_size'][idx],
                                                               latent_activation_fn=tf.nn.softplus,
                                                               num_hidden_layers=self.config["num_hidden_layers"],
                                                               num_hidden_units=self.config["num_hidden_units"],
                                                               hidden_activation_fn=self.config["hidden_activation_fn"],
                                                               is_training=self.is_training)
                if self.config.get('latent_sigma_threshold', 0) > 0:
                    sigma = tf.clip_by_value(sigma, 1e-3, self.config.get('latent_sigma_threshold'))
                    flat_sigma = tf.clip_by_value(flat_sigma, 1e-3, self.config.get('latent_sigma_threshold'))

        return (mu, sigma),  (flat_mu, flat_sigma)

    def build_latent_dist(self, input_, idx, scope, reuse):
        """
        Given the input parametrizes a Normal distribution.
        Args:
            input_:
            idx:
            scope: "approximate_posterior" or "prior".
            reuse:
        Returns:
            mu and sigma tensors.
        """
        if self.latent_layer_structure == C.LAYER_FC:
            return self.build_latent_dist_fc(input_, idx, scope, reuse)
        elif self.latent_layer_structure == C.LAYER_TCN:
            return self.build_latent_dist_tcn(input_, idx, scope, reuse)
        elif self.latent_layer_structure == C.LAYER_CONV1:
            return self.build_latent_dist_conv1(input_, idx, scope, reuse)
        else:
            raise Exception("Unknown latent layer type.")

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Builds stochastic latent variables hierarchically. q_input and p_input consist of outputs of stacked
        deterministic layers. self.vertical_dilation hyper-parameter denotes the size of the deterministic block. For
        example, if it is 5, then every fifth deterministic layer is used to estimate a random variable.

        Args:
            q_input (list): deterministic units to estimate the approximate posterior.
            p_input (list): deterministic units to estimate the prior.
            output_ops_dict (dict):
            eval_ops_dict (dict):
            summary_ops_dict (dict):

        Returns:
            A latent sample.
        """
        p_scope, q_scope = C.LATENT_P, C.LATENT_Q

        self.num_d_layers = len(q_input)
        assert self.num_d_layers % self.vertical_dilation == 0, "# of deterministic layers must be divisible by vertical dilation."
        self.num_s_layers = int(self.num_d_layers / self.vertical_dilation)

        # TODO
        self.config['latent_size'] = self.config['latent_size'] if isinstance(self.config['latent_size'], list) else [self.config['latent_size']]*self.num_s_layers

        self.q_approximate = [0]*self.num_s_layers
        self.q_dists = [0]*self.num_s_layers
        self.p_dists = [0]*self.num_s_layers

        # Indexing latent variables.
        if self.top_down_latents:
            # Build the top most latent layer.
            sl = self.num_s_layers-1  # stochastic layer index.
        else:
            sl = 0  # stochastic layer index.
        dl = (sl + 1)*self.vertical_dilation - 1  # deterministic layer index.

        # Estimate the prior of the first stochastic layer.
        scope = p_scope + "_" + str(sl + 1)
        reuse = self.reuse
        if self.dynamic_prior:
            if self.use_fixed_pz1:
                with tf.name_scope(scope):
                    latent_size = self.config['latent_size'][sl]
                    prior_shape = (tf.shape(p_input[0])[0], tf.shape(p_input[0])[1], latent_size)
                    p_dist = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))
            else:
                p_layer_inputs = [p_input[dl]]
                p_dist, _ = self.build_latent_dist(tf.concat(p_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
        else:
            # Insert N(0,1) as prior.
            with tf.name_scope(scope):
                latent_size = self.config['latent_size'][sl]
                prior_shape = (tf.shape(p_input[0])[0], tf.shape(p_input[0])[1], latent_size)
                p_dist = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))

        self.p_dists[sl] = p_dist
        # If it is not training, then we draw latent samples from the prior distribution.
        if self.is_sampling and self.dynamic_prior:
            posterior = p_dist
        else:
            scope = q_scope + "_" + str(sl + 1)
            reuse = self.reuse

            q_layer_inputs = [q_input[dl]]
            q_dist_approx, q_dist_approx_flat = self.build_latent_dist(tf.concat(q_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
            self.q_approximate[sl] = q_dist_approx

            # Estimate the approximate posterior distribution as a precision-weighted combination.
            if self.precision_weighted_update:
                scope = q_scope + "_pwu_" + str(sl + 1)
                q_dist = self.combine_normal_dist(q_dist_approx, p_dist, scope=scope)
            else:
                q_dist = q_dist_approx
            self.q_dists[sl] = q_dist
            # Set the posterior.
            posterior = q_dist

        posterior_sample_scope = "app_posterior_" + str(sl+1)
        posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], scope=posterior_sample_scope)
        if self.dense_z:
            self.latent_samples.append(posterior_sample)

        # Build hierarchy.
        if self.top_down_latents:
            loop_indices = range(self.num_s_layers-2, -1, -1)
        else:
            loop_indices = range(1, self.num_s_layers, 1)
        for sl in loop_indices:
            dl = (sl + 1)*self.vertical_dilation - 1

            p_dist_preceding = p_dist
            # Estimate the prior distribution.
            scope = p_scope + "_" + str(sl + 1)
            reuse = self.reuse

            # Draw a latent sample from the preceding posterior.
            if not self.use_same_q_sample:
                posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], posterior_sample_scope)

            if self.dynamic_prior:  # Concatenate TCN representation with a sample from the approximated posterior.
                p_layer_inputs = [p_input[dl], posterior_sample]
            else:
                p_layer_inputs = [posterior_sample]

            p_dist, p_dist_flat = self.build_latent_dist(tf.concat(p_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
            self.p_dists[sl] = p_dist

            if self.is_sampling and self.dynamic_prior:
                # Set the posterior.
                posterior = p_dist
            else:
                # Estimate the uncorrected approximate posterior distribution.
                scope = q_scope + "_" + str(sl + 1)
                reuse = self.reuse

                q_layer_inputs = [q_input[dl]]
                if self.recursive_q:
                    # Draw a latent sample from the preceding posterior.
                    if not self.use_same_q_sample:
                        posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], posterior_sample_scope)
                    q_layer_inputs.append(posterior_sample)

                q_dist_approx, q_dist_approx_flat = self.build_latent_dist(tf.concat(q_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
                self.q_approximate[sl] = q_dist_approx

                # Estimate the approximate posterior distribution as a precision-weighted combination.
                if self.precision_weighted_update:
                    scope = q_scope + "_pwu_" + str(sl + 1)
                    q_dist = self.combine_normal_dist(q_dist_approx, p_dist, scope=scope)
                else:
                    q_dist = q_dist_approx
                self.q_dists[sl] = q_dist
                # Set the posterior.
                posterior = q_dist

            # Draw a new sample from the approximated posterior distribution of this layer.
            posterior_sample_scope = "app_posterior_" + str(sl+1)
            posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], posterior_sample_scope)
            if self.dense_z:
                self.latent_samples.append(posterior_sample)

        if self.dense_z:  # Return samples of all stochastic layers.
            return self.latent_samples
        else:  # Use a latent sample from the final stochastic layer.
            return [self.draw_latent_sample(posterior[0], posterior[1], posterior_sample_scope)]

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions. If use_temporal_kld is True,
        then creates another KL-divergence term between consecutive approximate posteriors in time.
        """
        loss_ops_dict = loss_ops_dict or dict()
        # eval_dict contains each KLD term and latent q, p distributions for further analysis.
        eval_dict = kwargs.get("eval_dict", None)
        if eval_dict is not None:
            eval_dict["q_dists"] = self.q_dists
            eval_dict["p_dists"] = self.p_dists
        if not self.is_sampling:
            loss_key = "loss_kld"
            kld_loss = 0.0
            with tf.name_scope("kld_loss"):
                for sl in range(len(self.q_dists)):
                    with tf.name_scope("kld_" + str(sl)):
                        seq_kld_loss = sequence_mask*tf_loss.kld_normal_isotropic(self.q_dists[sl][0],
                                                                                  self.q_dists[sl][1],
                                                                                  self.p_dists[sl][0],
                                                                                  self.p_dists[sl][1],
                                                                                  reduce_sum=False)
                        kld_term = self.kld_weight*reduce_loss_fn(seq_kld_loss)

                        # This is just for monitoring. Only the entries in loss_ops_dict starting with "loss"
                        # contribute to the gradients.
                        if not self.is_training:
                            loss_ops_dict["KL"+str(sl)] = tf.stop_gradient(kld_term)

                        self.kld_loss_terms.append(kld_term)
                        kld_loss += kld_term
                        if eval_dict is not None:
                            eval_dict["summary_kld_" + str(sl)] = kld_term
                            eval_dict["sequence_kld_" + str(sl)] = seq_kld_loss

                # Optimization is done through the accumulated term (i.e., loss_ops_dict[loss_key]).
                self.ops_loss[loss_key] = kld_loss
                loss_ops_dict[loss_key] = kld_loss

        return self.ops_loss

    @classmethod
    def draw_latent_sample(cls, posterior_mu, posterior_sigma, scope):
        """
        Draws a latent sample by using the reparameterization trick.
        Args:
            posterior_mu:
            posterior_sigma:
            scope:
        Returns:
        """
        def normal_sample(mu, sigma):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            return tf.add(mu, tf.multiply(sigma, eps))

        with tf.name_scope(scope+"_z"):
            z = normal_sample(posterior_mu, posterior_sigma)
        return z

    @classmethod
    def combine_normal_dist(cls, dist1, dist2, scope):
        """
        Calculates precision-weighted combination of two Normal distributions.
        Args:
            dist1: (mu, sigma)
            dist2: (mu, sigma)
            scope:
        Returns:
        """
        with tf.name_scope(scope):
            mu1, mu2 = dist1[0], dist2[0]
            precision1, precision2 = tf.pow(dist1[1], -2), tf.pow(dist2[1], -2)

            sigma = 1.0/(precision1 + precision2)
            mu = (mu1*precision1 + mu2*precision2) / (precision1 + precision2)

            return mu, sigma
