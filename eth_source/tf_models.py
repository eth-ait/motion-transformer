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

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict, **kwargs):
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

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions.
        """
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

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict, **kwargs):
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

        # TODO: Hacky
        self.blocked_latent_indices = []

        self.vertical_dilation = self.config.get('vertical_dilation', 1)
        # Draw a new sample from the approximated posterior whenever needed. Otherwise, draw once and use it every time.
        self.use_same_q_sample = self.config.get('use_same_q_sample', False)
        # Feed the top-most latent layers with samples from a Normal(0, I) distribution.
        self.use_z0 = self.config.get('use_z0', True)
        # Apply KLD between the top-most q and N(0, 1)
        self.kld_q0_z0 = self.config.get('kld_q0_z0', False)
        # Apply KLD between the top-most p and N(0, 1)
        self.kld_p0_z0 = self.config.get('kld_p0_z0', False)
        # Apply pwu update at the top-most layer. If False, the approximate posterior q = approximate likelihood q_hat
        self.use_pwu_z0 = self.config.get('use_pwu_z0', False)
        # The top-most prior is dynamic or not. LadderVAE paper uses a fixed prior.
        # "fixed_z0" is deprecated. Keeping for backwards-compatibility. Use use_fixed_pz1 instead.
        self.use_fixed_pz1 = self.config.get('use_fixed_pz1', False) or self.config.get('fixed_z0', False)
        # Prior is calculated by using the deterministic representations at previous step.
        self.dynamic_prior = self.config.get('dynamic_prior', False)
        # Approximate posterior is estimated as a precision weighted update of the prior and initial model predictions.
        self.precision_weighted_update = self.config.get('precision_weighted_update', True)
        # Use the same parameters for q and p of the same stochastic layer.
        self.share_latent_params = self.config.get('share_latent_params', True)
        # Use the same parameters across hierarchical latent layers. Note that the latent sample and deterministic model
        # dimensionality of every layer must be the same.
        self.share_vertical_latent_params = self.config.get('share_vertical_latent_params', False)
        self.use_p_input_in_q = self.config.get('use_p_input_in_q', False)
        # In order to increase robustness of the learned prior use samples from the prior instead of the posterior.
        self.p_q_replacement_ratio = self.config.get('p_q_replacement_ratio', 0)
        if self.is_sampling or self.is_validation or self.is_eval:
            self.p_q_replacement_ratio = 0
        # Whether the q distribution is hierarchically updated as in the case of prior or not. In other words, lower
        # q layer uses samples of the upper q layer.
        self.recursive_q = self.config.get('recursive_q', True)
        self.top_down_latents = self.config.get('top_down_latents', True)
        self.latent_layer_structure = self.config.get('layer_structure', C.LAYER_FC)
        self.use_all_z = self.config.get('use_all_z', False)
        self.use_skip_latent = self.config.get('use_skip_latent', False)
        self.collect_latent_samples = self.use_all_z or self.use_skip_latent

        kld_weight = self.config.get('kld_weight', 1)
        if isinstance(kld_weight, dict) and self.global_step is not None:
            self.kld_weight = get_decay_variable(global_step=self.global_step, config=kld_weight, name="kld_weight")
            if self.is_training:
                self.summary_ops["kld_weight"] = tf.summary.scalar("kld_weight", self.kld_weight, collections=[self.mode+"/model_summary"])
        else:
            self.kld_weight = kld_weight
        if not self.is_training:
            self.kld_weight = 1.0

        self.latent_cell_p, self.latent_cell_state_p = None, None
        self.latent_cell_q, self.latent_cell_state_q = None, None
        if self.latent_layer_structure == C.LAYER_RNN:
            p_scope = "latent_cell_" + C.LATENT_P
            q_scope = "latent_cell_" + C.LATENT_Q
            with tf.variable_scope(p_scope, reuse=self.reuse):
                self.latent_cell_p = get_rnn_cell(cell_type=config["cell_type"],
                                                  size=config["cell_size"],
                                                  num_layers=config["cell_num_layers"])
                self.latent_cell_state_p = None

            with tf.variable_scope(q_scope, reuse=self.reuse):
                self.latent_cell_q = get_rnn_cell(cell_type=config["cell_type"],
                                                  size=config["cell_size"],
                                                  num_layers=config["cell_num_layers"])
                self.latent_cell_state_q = None
        # Latent space components.
        self.p_mu = []
        self.q_mu = []
        self.p_sigma = []
        self.q_sigma = []

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

    def build_latent_dist_rnn(self, input_, idx, scope, reuse):
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
        is_prior = scope.startswith(C.LATENT_P)
        # Flatten the (horizontally) sequence input to feed into an rnn cell (vertically).
        flat_input = tf.reshape(input_, [-1, input_.shape.as_list()[-1]])
        if is_prior and self.latent_cell_state_p is None:
            self.latent_cell_state_p = self.latent_cell_p.zero_state(batch_size=tf.shape(flat_input)[0], dtype=tf.float32)
        elif not is_prior and self.latent_cell_state_q is None:
            self.latent_cell_state_q = self.latent_cell_q.zero_state(batch_size=tf.shape(flat_input)[0], dtype=tf.float32)

        num_latent_units = self.config['latent_size'][idx]
        p_scope = "latent_cell_" + C.LATENT_P
        q_scope = "latent_cell_" + C.LATENT_Q
        with tf.name_scope(scope):
            if is_prior:
                with tf.variable_scope(p_scope):
                    flat_cell_output, self.latent_cell_state_p = self.latent_cell_p(inputs=flat_input, state=self.latent_cell_state_p)
            else:
                with tf.variable_scope(q_scope):
                    flat_cell_output, self.latent_cell_state_q = self.latent_cell_q(inputs=flat_input, state=self.latent_cell_state_q)
            with tf.variable_scope(scope + '_mu', reuse=reuse):
                flat_mu = linear(input_layer=flat_cell_output,
                                 output_size=num_latent_units,
                                 activation_fn=None,
                                 is_training=self.is_training)
                mu = tf.reshape(flat_mu, [tf.shape(input_)[0], -1, num_latent_units])

            with tf.variable_scope(scope + '_sigma', reuse=reuse):
                flat_sigma = linear(input_layer=flat_cell_output,
                                    output_size=num_latent_units,
                                    activation_fn=tf.exp,
                                    is_training=self.is_training)
                sigma = tf.reshape(flat_sigma, [tf.shape(input_)[0], -1, num_latent_units])

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
        elif self.latent_layer_structure == C.LAYER_RNN:
            return self.build_latent_dist_rnn(input_, idx, scope, reuse)
        elif self.latent_layer_structure == C.LAYER_TCN:
            return self.build_latent_dist_tcn(input_, idx, scope, reuse)
        elif self.latent_layer_structure == C.LAYER_CONV1:
            return self.build_latent_dist_conv1(input_, idx, scope, reuse)
        else:
            raise Exception("Unknown latent layer type.")

    def build_latent_layer(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        p_scope = C.LATENT_P if not self.share_latent_params else "latent"
        q_scope = C.LATENT_Q if not self.share_latent_params else "latent"

        self.num_d_layers = len(q_input)
        assert self.num_d_layers % self.vertical_dilation == 0, "# of deterministic layers must be divisible by vertical dilation."
        self.num_s_layers = int(self.num_d_layers / self.vertical_dilation)

        # TODO
        self.config['latent_size'] = self.config['latent_size'] if isinstance(self.config['latent_size'], list) else [self.config['latent_size']]*self.num_s_layers

        self.q_approximate = [0]*self.num_s_layers
        self.q_dists = [0]*self.num_s_layers
        self.p_dists = [0]*self.num_s_layers

        if self.top_down_latents:
            # Build the top most latent layer.
            sl = self.num_s_layers-1  # stochastic layer index.
        else:
            sl = 0  # stochastic layer index.
        dl = (sl + 1)*self.vertical_dilation - 1  # deterministic layer index.

        # Estimate the prior of the first stochastic layer.
        scope = p_scope if self.share_vertical_latent_params else p_scope + "_" + str(sl + 1)
        reuse = self.reuse
        p_dense_created = False
        if self.dynamic_prior:
            if self.use_fixed_pz1:
                with tf.name_scope(scope):
                    latent_size = self.config['latent_size'][sl]
                    prior_shape = (tf.shape(p_input[0])[0], tf.shape(p_input[0])[1], latent_size)
                    p_dist = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))
            else:
                if self.use_z0:
                    z0_sample = tf.random_normal((tf.shape(p_input[0])[0], tf.shape(p_input[0])[1], self.config['latent_size'][sl]), 0.0, 1.0, dtype=tf.float32)
                    p_layer_inputs = [p_input[dl], z0_sample]
                else:
                    p_layer_inputs = [p_input[dl]]
                p_dist, _ = self.build_latent_dist(tf.concat(p_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
                p_dense_created = True
        else:
            # Insert N(0,1) as prior.
            with tf.name_scope(scope):
                latent_size = self.config['latent_size'][sl]
                prior_shape = (tf.shape(p_input[0])[0], tf.shape(p_input[0])[1], latent_size)
                p_dist = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))

        self.p_dists[sl] = p_dist
        if self.is_sampling and self.dynamic_prior:
            posterior = p_dist
        else:
            scope = q_scope if self.share_vertical_latent_params else q_scope + "_" + str(sl + 1)
            reuse = True if (self.share_latent_params and p_dense_created) else self.reuse

            q_layer_inputs = [q_input[dl]]
            if self.recursive_q and self.use_z0 and not self.use_fixed_pz1:
                with tf.name_scope(scope + "_z"):
                    z0_sample = tf.random_normal((tf.shape(q_input[dl])[0], tf.shape(q_input[dl])[1], self.config['latent_size'][sl]), 0.0, 1.0, dtype=tf.float32)
                q_layer_inputs.append(z0_sample)

            if self.use_p_input_in_q and not self.share_latent_params:
                # If we are sharing the parameters between q and p, then this is not possible.
                q_layer_inputs.append(p_input[dl])

            q_dist_approx, q_dist_approx_flat = self.build_latent_dist(tf.concat(q_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
            self.q_approximate[sl] = q_dist_approx

            if self.use_pwu_z0:
                # Estimate the approximate posterior distribution as a precision-weighted combination.
                # For the first latent layer, apply PWU only if z0 is used.
                if self.precision_weighted_update:
                    scope = q_scope + "_pwu_" + str(sl + 1)
                    q_dist = self.combine_normal_dist(q_dist_approx, p_dist, scope=scope)
                else:
                    q_dist = q_dist_approx
                self.q_dists[sl] = q_dist
            else:
                # For the initial layer, posterior is equal to its approximation.
                q_dist = q_dist_approx
                self.q_dists[sl] = q_dist

            # Set the posterior.
            posterior = q_dist

        posterior_sample_scope = "app_posterior_" + str(sl+1)
        posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], scope=posterior_sample_scope, idx=sl)
        if self.collect_latent_samples:
            self.latent_samples.append(posterior_sample)

        if self.top_down_latents:
            loop_indices = range(self.num_s_layers-2, -1, -1)
        else:
            loop_indices = range(1, self.num_s_layers, 1)

        for sl in loop_indices:
            dl = (sl + 1)*self.vertical_dilation - 1

            p_dist_preceding = p_dist
            # Estimate the prior distribution.
            scope = p_scope if self.share_vertical_latent_params else p_scope + "_" + str(sl + 1)
            reuse = True if (self.share_vertical_latent_params and p_dense_created) else self.reuse

            # Draw a latent sample from the preceding posterior.
            if not self.use_same_q_sample:
                posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], posterior_sample_scope, sl)

            if self.dynamic_prior:  # Concatenate TCN representation with a sample from the approximated posterior.
                p_layer_inputs = [p_input[dl], posterior_sample]
            else:
                p_layer_inputs = [posterior_sample]

            p_dist, p_dist_flat = self.build_latent_dist(tf.concat(p_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
            self.p_dists[sl] = p_dist
            p_dense_created = True

            if self.is_sampling and self.dynamic_prior:
                # Set the posterior.
                posterior = p_dist
            else:
                # Estimate the uncorrected approximate posterior distribution.
                scope = q_scope if self.share_vertical_latent_params else q_scope + "_" + str(sl + 1)
                reuse = True if (self.share_latent_params or self.share_vertical_latent_params and p_dense_created) else self.reuse

                q_layer_inputs = [q_input[dl]]
                if self.recursive_q:
                    # Draw a latent sample from the preceding posterior.
                    if not self.use_same_q_sample:
                        posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist_preceding[0], p_dist_preceding[1], posterior_sample_scope, sl)
                    q_layer_inputs.append(posterior_sample)
                if self.use_p_input_in_q and not self.share_latent_params:
                    # If we are sharing the parameters between q and p, then this is not possible.
                    q_layer_inputs.append(p_input[dl])

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
            posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], posterior_sample_scope, sl)
            if self.collect_latent_samples:
                self.latent_samples.append(posterior_sample)

        # TODO Missing an activation function. Do we need one here?
        if self.use_all_z:
            # Concatenate the latent samples of all stochastic layers.
            return tf.concat(self.latent_samples, axis=-1)
        elif self.use_skip_latent:
            # Get the summation of all latent samples.
            return sum(self.latent_samples)
        else:
            # Use a latent sample from the final stochastic layer.
            return self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], posterior_sample_scope, sl)

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions. If use_temporal_kld is True,
        then creates another KL-divergence term between consecutive approximate posteriors in time.
        """
        loss_ops_dict = loss_ops_dict or {}
        # eval_dict contains each KLD term and latent q, p distributions for further analysis.
        eval_dict = kwargs.get("eval_dict", None)
        if eval_dict is not None:
            eval_dict["q_dists"] = self.q_dists
            eval_dict["p_dists"] = self.p_dists
        if not self.is_sampling:
            loss_key = "loss_kld"
            kld_loss = 0.0
            with tf.name_scope("kld_loss"):
                for sl in range(self.num_s_layers-1, -1, -1):
                    with tf.name_scope("kld_" + str(sl)):
                        seq_kld_loss = sequence_mask*tf_loss.kld_normal_isotropic(self.q_dists[sl][0],
                                                                                  self.q_dists[sl][1],
                                                                                  self.p_dists[sl][0],
                                                                                  self.p_dists[sl][1],
                                                                                  reduce_sum=False)
                        kld_term = self.kld_weight*reduce_loss_fn(seq_kld_loss)

                        # This is just for reporting. Only the entries in loss_ops_dict starting with "loss"
                        # contribute to the gradients.
                        if not self.is_training:
                            loss_ops_dict["KL"+str(sl)] = tf.stop_gradient(kld_term)

                        self.kld_loss_terms.append(kld_term)
                        kld_loss += kld_term
                        if eval_dict is not None:
                            eval_dict["summary_kld_" + str(sl)] = kld_term
                            eval_dict["sequence_kld_" + str(sl)] = seq_kld_loss

                # TODO Assuming top-down hierarchy.
                z0_shape = tf.shape(self.q_dists[-1][0])
                if self.kld_q0_z0:
                    with tf.name_scope("kld_q0_z0"):
                        seq_kld_loss = sequence_mask*tf_loss.kld_normal_isotropic(self.q_dists[-1][0],
                                                                                  self.q_dists[-1][1],
                                                                                  tf.zeros(z0_shape, dtype=tf.float32),
                                                                                  tf.ones(z0_shape, dtype=tf.float32),
                                                                                  reduce_sum=False)
                        kld_term = self.kld_weight*reduce_loss_fn(seq_kld_loss)

                    self.kld_loss_terms.append(kld_term)
                    kld_loss += kld_term
                    if eval_dict is not None:
                        eval_dict["summary_kld_" + "q0_z0"] = kld_term
                        eval_dict["sequence_kld_" + "q0_z0"] = seq_kld_loss

                if self.kld_p0_z0:
                    with tf.name_scope("kld_p0_z0"):
                        seq_kld_loss = sequence_mask*tf_loss.kld_normal_isotropic(self.p_dists[-1][0],
                                                                                  self.p_dists[-1][1],
                                                                                  tf.zeros(z0_shape, dtype=tf.float32),
                                                                                  tf.ones(z0_shape, dtype=tf.float32),
                                                                                  reduce_sum=False)
                        kld_term = self.kld_weight*reduce_loss_fn(seq_kld_loss)

                    self.kld_loss_terms.append(kld_term)
                    kld_loss += kld_term
                    if eval_dict is not None:
                        eval_dict["summary_kld_" + "p0_z0"] = kld_term
                        eval_dict["sequence_kld_" + "p0_z0"] = seq_kld_loss

                # Optimization is done through the accumulated term (i.e., loss_ops_dict[loss_key]).
                self.ops_loss[loss_key] = kld_loss
                loss_ops_dict[loss_key] = kld_loss
        return loss_ops_dict

    def draw_latent_sample(self, posterior_mu, posterior_sigma, prior_mu, prior_sigma, scope, idx):
        """
        Draws a latent sample by using the reparameterization trick.
        Args:
            prior_mu:
            prior_sigma:
            posterior_mu:
            posterior_sigma:
            scope
        Returns:
        """
        def normal_sample(mu, sigma):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            return tf.add(mu, tf.multiply(sigma, eps))

        if idx in self.blocked_latent_indices:
            z = tf.zeros(tf.shape(posterior_mu))
            # eps = tf.random_normal(tf.shape(posterior_sigma), 0.0, 1.0, dtype=tf.float32)
            # z = tf.add(posterior_mu, tf.multiply(posterior_sigma*0.1, eps))
        else:
            if self.p_q_replacement_ratio > 0 and prior_mu is not None:
                with tf.name_scope(scope+"_z"):
                    z = tf.cond(pred=tf.random_uniform([1])[0] < self.p_q_replacement_ratio,
                                true_fn=lambda: normal_sample(prior_mu, prior_sigma),
                                false_fn=lambda: normal_sample(posterior_mu, posterior_sigma))
            else:
                with tf.name_scope(scope+"_z"):
                    z = normal_sample(posterior_mu, posterior_sigma)
        return z

    def combine_normal_dist(self, dist1, dist2, scope):
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


class BaseTemporalModel(object):
    """
    Model class for modeling of temporal data, providing auxiliary functions implementing tensorflow routines and
    abstract functions to build model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        self.config = config
        self.global_step = kwargs.get("global_step", None)

        # "sampling" is only valid for generative models.
        assert mode in [C.TRAIN, C.VALID, C.EVAL, C.SAMPLE]
        self.mode = mode
        self.is_sampling = mode == C.SAMPLE
        self.is_validation = mode in [C.VALID, C.EVAL]
        self.is_training = mode == C.TRAIN
        self.is_eval = mode == C.EVAL  # Similar to the validation mode, returns some details for analysis.
        self.print_every_step = self.config.get('print_every_step')

        self.reuse = reuse
        self.session = session

        self.placeholders = placeholders
        self.pl_inputs = placeholders[C.PL_INPUT]
        self.pl_targets = placeholders[C.PL_TARGET]
        self.pl_seq_length = placeholders[C.PL_SEQ_LEN]
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.pl_seq_length, dtype=tf.float32), -1)

        # Create an activation function for std predictions. Bias helps the model to make smoother predictions.
        # output_sigma_bias = config.get("output_sigma_bias", 0)
        sigma_threshold = config.get("sigma_threshold", 50.0)
        # self.sigma_activation_fn = lambda x: tf.clip_by_value(tf.exp(x - output_sigma_bias), 1e-6, sigma_threshold) if output_sigma_bias > 0 else tf.clip_by_value(tf.exp(x), 1e-6, sigma_threshold)
        # self.sigma_activation_fn = tf.nn.softplus
        self.sigma_activation_fn = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-3, sigma_threshold)

        # Creates a sample by using model outputs.
        self.sample_fn_tf, self.sample_fn_np = config.get_sample_function()

        self.input_dims = input_dims.copy()
        self.target_dims = target_dims.copy()
        self.target_pieces = tf.split(self.pl_targets, target_dims, axis=2)

        input_shape = self.pl_inputs.shape.as_list()  # Check if input shape is defined.
        self.batch_size = tf.shape(self.pl_inputs)[0] if input_shape[0] is None else input_shape[0]
        self.sequence_length = tf.shape(self.pl_inputs)[1] if input_shape[1] is None else input_shape[1]

        self.output_layer_config = copy.deepcopy(config.get('output_layer'))
        # Update output ops.
        self.loss_config = copy.deepcopy(self.config.get('loss', None))

        # Function to calculate final loss value (i.e., average or sum). See get_reduce_loss_func in tf_model_utils.py
        self.reduce_loss_fn = None
        # Loss op to be used during training.
        self.loss = None
        # Tensorflow summary object for loss plots.
        self.loss_summary = None
        # Accumulating likelihood loss terms.
        self.likelihood = 0

        # Model's output.
        self.output_sample = None
        # Model's raw input
        self.input_sample = None
        # Output of initial input layer.
        self.inputs_hidden = None

        # In validation/evaluation mode we first accumulate losses and then plot.
        # At the end of validation loop, we calculate average performance on the whole validation dataset and create
        # corresponding summary entries. See build_summary_plots method and `Summary methods for validation mode`
        # section.
        # Create containers and placeholders for every loss term. After each validation step, keep adding losses.
        if not self.is_training:
            self.container_loss = dict()
            self.container_loss_placeholders = dict()
            self.container_loss_summaries = dict()
            self.container_validation_feed_dict = dict()
            self.validation_summary_num_runs = 0

        # Ops to be evaluated by training loop function. It is a dictionary containing <key, value> pairs where the
        # `value` is tensorflow graph op. For example, summary, loss, training operations. Note that different modes
        # (i.e., training, sampling, validation) may have different set of ops.
        self.ops_run_loop = dict()
        # `summary` ops are kept in a list.
        self.ops_run_loop['summary'] = []

        # Dictionary of model outputs such as logits or mean and sigma of Gaussian distribution modeling outputs.
        # They are used in making predictions and creating loss terms.
        self.ops_model_output = dict()

        # To keep track of loss ops. List of loss terms that must be evaluated by session.run during training.
        self.ops_loss = dict()

        # (Default) graph ops to be fed into session.run while evaluating the model. Note that tf_evaluate* codes expect
        # to get these op results.
        self.ops_evaluation = dict()

        # Graph ops for scalar summaries such as average predicted variance.
        self.ops_scalar_summary = dict()

        # Auxiliary ops to be used in analysis of the model. It is used only in the evaluation mode.
        self.ops_for_eval_mode = dict()

        # Total number of trainable parameters.
        self.num_parameters = None

        for loss_name, loss_entry in self.loss_config.items():
            self.define_loss(loss_entry)

    def define_loss(self, loss_config):
        if loss_config['type'] in [C.NLL_NORMAL, C.NLL_BINORMAL]:
            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_MU)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(None)

            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_SIGMA)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(self.sigma_activation_fn)

        if loss_config['type'] in [C.NLL_BINORMAL]:
            self.output_layer_config['out_keys'].append(loss_config['out_key']+C.SUF_RHO)
            self.output_layer_config['out_dims'].append(1)
            self.output_layer_config['out_activation_fn'].append(C.TANH)

        if loss_config['type'] in [C.NLL_GMM, C.NLL_BIGMM]:
            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_MU)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']] * loss_config['num_components'])
            self.output_layer_config['out_activation_fn'].append(None)

            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_SIGMA)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']] * loss_config['num_components'])
            self.output_layer_config['out_activation_fn'].append(self.sigma_activation_fn)

            self.output_layer_config['out_keys'].append(loss_config['out_key']+C.SUF_COEFFICIENT)
            self.output_layer_config['out_dims'].append(loss_config['num_components'])
            self.output_layer_config['out_activation_fn'].append(C.SOFTMAX)

        if loss_config['type'] == C.NLL_BIGMM:
            self.output_layer_config['out_keys'].append(loss_config['out_key']+C.SUF_RHO)
            self.output_layer_config['out_dims'].append(loss_config['num_components'])
            self.output_layer_config['out_activation_fn'].append(C.TANH)

        if loss_config['type'] in [C.NLL_BERNOULLI]:
            self.output_layer_config['out_keys'].append(loss_config['out_key']+C.SUF_BINARY)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(C.SIGMOID)

        if loss_config['type'] == C.NLL_CENT:
            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_MU)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(None)

        if loss_config['type'] == C.NLL_CENT_BINARY:
            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_MU)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(None)

    def build_graph(self):
        """
        Called by TrainingEngine. Assembles modules of tensorflow computational graph by creating model, loss terms and
        summaries for tensorboard. Applies preprocessing on the inputs and postprocessing on model outputs if necessary.
        """
        raise NotImplementedError('subclasses must override build_graph method')

    def build_network(self):
        """
        Builds internal dynamics of the model. Sets
        """
        raise NotImplementedError('subclasses must override build_network method')

    def sample(self, **kwargs):
        """
        Draws samples from model.
        """
        raise NotImplementedError('subclasses must override sample method')

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        raise NotImplementedError('subclasses must override reconstruct method')

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        # Function to get final loss value, i.e., average or sum.
        self.reduce_loss_fn = get_reduce_loss_func(self.config.get('reduce_loss'), tf.reduce_sum(self.seq_loss_mask, axis=[1, 2]))
        for loss_name, loss_entry in self.loss_config.items():
            loss_type = loss_entry['type']
            out_key = loss_entry['out_key']
            target_idx = loss_entry['target_idx']
            loss_key = "loss_" + loss_name
            op_loss_key = loss_name + "_loss"
            if loss_key not in self.ops_loss:
                with tf.name_scope(op_loss_key):
                    # Negative log likelihood loss.
                    if loss_type == C.NLL_NORMAL:
                        logli_term = tf_loss.logli_normal_isotropic(self.target_pieces[target_idx],
                                                                    self.ops_model_output[out_key + C.SUF_MU],
                                                                    self.ops_model_output[out_key + C.SUF_SIGMA],
                                                                    reduce_sum=False)
                    elif loss_type == C.NLL_BINORMAL:
                        logli_term = tf_loss.logli_normal_bivariate(self.target_pieces[target_idx],
                                                                    self.ops_model_output[out_key + C.SUF_MU],
                                                                    self.ops_model_output[out_key + C.SUF_SIGMA],
                                                                    self.ops_model_output[out_key + C.SUF_RHO],
                                                                    reduce_sum=False)
                    elif loss_type == C.NLL_GMM:
                        logli_term = tf_loss.logli_gmm_logsumexp(self.target_pieces[target_idx],
                                                                 self.ops_model_output[out_key + C.SUF_MU],
                                                                 self.ops_model_output[out_key + C.SUF_SIGMA],
                                                                 self.ops_model_output[out_key + C.SUF_COEFFICIENT])
                    elif loss_type == C.NLL_BERNOULLI:
                        logli_term = tf_loss.logli_bernoulli(self.target_pieces[target_idx],
                                                             self.ops_model_output[out_key + C.SUF_BINARY])
                    elif loss_type == C.MSE:
                        logli_term = -tf.reduce_sum(tf.square((self.target_pieces[target_idx] - self.ops_model_output[out_key + C.SUF_MU])), axis=2, keepdims=True)
                    elif loss_type == C.NLL_CENT:
                        labels = self.target_pieces[target_idx]
                        logits = self.ops_model_output[out_key + C.SUF_MU]
                        logli_term = tf.expand_dims(-tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), axis=-1)
                    elif loss_type == C.NLL_CENT_BINARY:
                        labels = self.target_pieces[target_idx]
                        logits = self.ops_model_output[out_key + C.SUF_MU]
                        logli_term = -tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                        # Get model's predicted probabilities for binary outputs.
                        self.ops_evaluation["out_probability"] = tf.nn.sigmoid(logits)
                    else:
                        raise Exception(loss_type + " is not implemented.")

                    self.likelihood += logli_term
                    loss_term = -loss_entry['weight']*self.reduce_loss_fn(self.seq_loss_mask*logli_term)
                    self.ops_loss[loss_key] = loss_term

    def build_total_loss(self):
        """
        Accumulate losses to create training optimization. Model.loss is used by the optimization function.
        """
        self.loss = 0
        for loss_key, loss_op in self.ops_loss.items():
            # Optimization is done by only using "loss*" terms.
            if loss_key.startswith("loss"):
                self.loss += loss_op
        self.ops_loss['total_loss'] = self.loss

    def build_summary_plots(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch
        and then report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create
        placeholders per loss to pass the average loss.
        """
        if self.is_training:
            # For each loss term, create a tensorboard plot.
            for loss_name, loss_op in self.ops_loss.items():
                tf.summary.scalar(loss_name, loss_op, collections=[self.mode + '_summary_plot', self.mode + '_loss'])

        else:
            # Validation: first accumulate losses and then plot.
            # Create containers and placeholders for every loss term. After each validation step, keeps summing losses.
            # At the end of validation loop, calculates average performance on the whole validation dataset and creates
            # summary entries.
            for loss_name, _ in self.ops_loss.items():
                self.container_loss[loss_name] = 0
                self.container_loss_placeholders[loss_name] = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar(loss_name, self.container_loss_placeholders[loss_name], collections=[self.mode + '_summary_plot', self.mode + '_loss'])
                self.container_validation_feed_dict[self.container_loss_placeholders[loss_name]] = 0.0

        # for summary_name, scalar_summary_op in self.ops_scalar_summary.items():
        #    tf.summary.scalar(summary_name, scalar_summary_op, collections=[self.mode + '_summary_plot', self.mode + '_scalar_summary'])

    def finalise_graph(self):
        """
        Finalises graph building. It is useful if child classes must create some ops first.
        """
        self.loss_summary = tf.summary.merge_all(self.mode + '_summary_plot')
        if self.is_training:
            self.register_run_ops('summary', self.loss_summary)

        self.register_run_ops('loss', self.ops_loss)
        self.register_run_ops('batch_size', tf.shape(self.pl_seq_length)[0])

    def training_step(self, step, epoch, feed_dict=None):
        """
        Training loop function. Takes a batch of samples, evaluates graph ops and parameter_update model parameters.

        Args:
            step: current step.
            epoch: current epoch.
            feed_dict (dict): feed dictionary.

        Returns (dict): evaluation results.
        """
        start_time = time.perf_counter()
        ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)

        if math.isnan(ops_run_loop_results['loss']['total_loss']):
            raise Exception("NaN values.")

        if step % self.print_every_step == 0:
            time_elapsed = (time.perf_counter() - start_time)
            self.log_loss(ops_run_loop_results['loss'], step, epoch, time_elapsed, prefix=self.mode + ": ")

        return ops_run_loop_results

    def evaluation_step(self, step, epoch, num_iterations, feed_dict=None):
        """
        Evaluation loop function. Evaluates the whole validation/test dataset and logs performance.

        Args:
            step: current step.
            epoch: current epoch.
            num_iterations: number of steps.
            feed_dict (dict): feed dictionary.

        Returns: summary object.
        """
        self.reset_validation_loss()
        start_time = time.perf_counter()
        for i in range(num_iterations):
            ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)
            self.update_validation_loss(ops_run_loop_results)

        summary, total_loss = self.get_validation_summary()

        time_elapsed = (time.perf_counter() - start_time)
        self.log_loss(total_loss, step, epoch, time_elapsed, prefix=self.mode + ": ")

        return summary, total_loss

    def evaluation_step_test_time(self, coord, threads, step, epoch, num_iterations, feed_dict=None):
        self.reset_validation_loss()
        start_time = time.perf_counter()
        for i in range(num_iterations-1):
            ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)
            self.update_validation_loss(ops_run_loop_results)
        try:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=0.5)
        except:
            pass
        ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)
        self.update_validation_loss(ops_run_loop_results)

        summary, total_loss = self.get_validation_summary()
        time_elapsed = (time.perf_counter() - start_time)
        self.log_loss(total_loss, step, epoch, time_elapsed, prefix=self.mode + ": ")
        return summary, total_loss

    def log_loss(self, eval_loss, step=0, epoch=0, time_elapsed=None, prefix=""):
        """
        Prints status messages during training. It is called in the main training loop.
        Args:
            eval_loss (dict): evaluated results of `ops_loss` dictionary.
            step (int): current step.
            epoch (int): current epoch.
            time_elapsed (float): elapsed time.
            prefix (str): some informative text. For example, "training" or "validation".
        """
        loss_format = prefix + "{}/{} \t Total: {:.4f} \t"
        loss_entries = [step, epoch, eval_loss['total_loss']]

        for loss_key in sorted(eval_loss.keys()):
            if loss_key != 'total_loss':
                loss_format += "{}: {:.4f} \t"
                loss_entries.append(loss_key)
                loss_entries.append(eval_loss[loss_key])

        if time_elapsed is not None:
            print(loss_format.format(*loss_entries) + "time/batch = {:.3f}".format(time_elapsed))
        else:
            print(loss_format.format(*loss_entries))

    def register_run_ops(self, op_key, op):
        """
        Adds a new graph op into `self.ops_run_loop`.

        Args:
            op_key (str): dictionary key.
            op: tensorflow op

        Returns:
        """
        if op_key in self.ops_run_loop and isinstance(self.ops_run_loop[op_key], list):
            self.ops_run_loop[op_key].append(op)
        else:
            self.ops_run_loop[op_key] = op

        for key, op in self.ops_model_output.items():
            self.ops_run_loop[key] = op
        self.ops_run_loop["inputs"] = self.pl_inputs
        self.ops_run_loop["targets"] = self.pl_targets

    def flat_tensor(self, tensor, dim=-1):
        """
        Reshapes a tensor such that it has 2 dimensions. The dimension specified by `dim` is kept.
        """
        keep_dim_size = tensor.shape.as_list()[dim]
        return tf.reshape(tensor, [-1, keep_dim_size])

    def temporal_tensor(self, flat_tensor):
        """
        Reshapes a flat tensor (2-dimensional) to a tensor with shape (batch_size, seq_len, feature_size). Assuming
        that the flat tensor has shape of (batch_size*seq_len, feature_size).
        """
        feature_size = flat_tensor.shape.as_list()[-1]
        return tf.reshape(flat_tensor, [self.batch_size, -1, feature_size])

    def log_num_parameters(self):
        """
        Prints total number of parameters.
        """
        num_param = 0
        for v in tf.global_variables():
            num_param += np.prod(v.shape.as_list())

        self.num_parameters = num_param
        print("# of parameters: " + str(num_param))
        self.config.set('total_parameters', int(self.num_parameters), override=True)

    ########################################
    # Summary methods for validation mode.
    ########################################
    def update_validation_loss(self, loss_evaluated):
        """
        Updates validation losses. Note that this method is called after every validation step.

        Args:
            loss_evaluated: valuated results of `ops_loss` dictionary.
        """
        batch_size = loss_evaluated["batch_size"]
        self.validation_summary_num_runs += batch_size
        for loss_name, loss_value in loss_evaluated["loss"].items():
            self.container_loss[loss_name] += (loss_value*batch_size)

    def reset_validation_loss(self):
        """
        Resets validation loss containers.
        """
        self.validation_summary_num_runs = 0
        for loss_name, loss_value in self.container_loss.items():
            self.container_loss[loss_name] = 0

    def get_validation_summary(self):
        """
        Creates a feed dictionary of validation losses for validation summary. Note that this method is called after
        validation loops is over.

        Returns (dict, dict):
            feed_dict for validation summary.
            average `ops_loss` results for `log_loss` method.
        """
        for loss_name, loss_pl in self.container_loss_placeholders.items():
            self.container_loss[loss_name] /= self.validation_summary_num_runs
            self.container_validation_feed_dict[loss_pl] = self.container_loss[loss_name]

        self.validation_summary_num_runs = 0
        valid_summary = self.session.run(self.loss_summary, self.container_validation_feed_dict)
        return valid_summary, self.container_loss


class CCN(BaseTemporalModel):
    """
    Causal convolutional network from `Wavenet: A Generative Model for Raw Audio` (https://arxiv.org/abs/1609.03499)
    paper.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(CCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.input_layer_config = config.get('input_layer', None)
        self.cnn_layer_config = config.get('cnn_layer')
        self.use_gate = self.cnn_layer_config.get('use_gating', False)
        self.use_residual = self.cnn_layer_config.get('use_residual', False)
        self.use_skip = self.cnn_layer_config.get('use_skip', False)
        # Concatenates representations of these layers for the outputs.
        self.tcn_output_layer_idx = self.cnn_layer_config.get('tcn_output_layer_idx', [-1])
        # If True, at every layer the input sequence is padded with zeros at the beginning such that the output length
        # becomes equal to the input length.
        self.zero_padding = self.cnn_layer_config.get('zero_padding', False)
        self.activation_fn = get_activation_fn(self.cnn_layer_config['activation_fn'])

        # Output of temporal convolutional layers.
        self.temporal_block_outputs = None

        # Model's receptive field length.
        self.receptive_field_width = None
        # Model's output length. If self.zero_padding is True, then it is the same as self.pl_seq_length.
        self.output_width = None

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
            filter_op = CCN.causal_conv_layer(input_layer=input_layer,
                                              num_filters=num_filters,
                                              kernel_size=kernel_size,
                                              dilation=dilation,
                                              zero_padding=zero_padding,
                                              activation_fn=tf.nn.tanh)
        with tf.name_scope('gate_conv'):
            gate_op = CCN.causal_conv_layer(input_layer=input_layer,
                                            num_filters=num_filters,
                                            kernel_size=kernel_size,
                                            dilation=dilation,
                                            zero_padding=zero_padding,
                                            activation_fn=tf.nn.sigmoid)
        with tf.name_scope('gating'):
            gated_dilation = gate_op*filter_op

        return gated_dilation

    @staticmethod
    def temporal_block_ccn(input_layer, num_filters, kernel_size, dilation, activation_fn, num_extra_conv=0, use_gate=True, use_residual=True, zero_padding=False):
        if use_gate:
            with tf.name_scope('gated_causal_layer'):
                temp_out = CCN.causal_gated_layer(input_layer=input_layer,
                                                  kernel_size=kernel_size,
                                                  num_filters=num_filters,
                                                  dilation=dilation,
                                                  zero_padding=zero_padding)
        else:
            with tf.name_scope('causal_layer'):
                temp_out = CCN.causal_conv_layer(input_layer=input_layer,
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

    def build_graph(self):
        """
        Builds model and creates plots for tensorboard. Decomposes model building into sub-modules and makes inheritance
        is easier.
        """
        self.build_network()
        self.build_loss_terms()
        self.build_total_loss()
        self.build_summary_plots()
        self.finalise_graph()
        if self.reuse is False:
            self.log_num_parameters()

    def build_network(self):
        self.build_input_layer()
        self.build_causal_convolutions()
        self.build_output_layer()

    def build_input_layer(self):
        """
        Builds a number fully connected layers projecting the inputs into an intermediate representation  space.
        """
        self.inputs_hidden = self.pl_inputs
        if self.input_layer_config is not None:
            if self.input_layer_config.get("dropout_rate", 0) > 0:
                with tf.variable_scope('input_layer', reuse=self.reuse):
                    self.inputs_hidden = tf.layers.dropout(self.pl_inputs,
                                                           rate=self.input_layer_config.get("dropout_rate"),
                                                           noise_shape=None,
                                                           seed=self.config.seed,
                                                           training=self.is_training)

    def build_causal_convolutions(self):
        current_layer = self.inputs_hidden

        self.receptive_field_width = CCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        if self.zero_padding is True:
            self.output_width = tf.shape(current_layer)[1]
        else:
            self.output_width = tf.shape(current_layer)[1] - self.receptive_field_width + 1

        # Initial causal convolution layer mapping inputs to a space with number of dilation filters dimensions.
        with tf.variable_scope('causal_conv_layer_0', reuse=self.reuse):
            current_layer = CCN.causal_conv_layer(input_layer=current_layer,
                                                  num_filters=self.cnn_layer_config['num_filters'],
                                                  kernel_size=self.cnn_layer_config['filter_size'],
                                                  dilation=1,
                                                  zero_padding=self.zero_padding,
                                                  activation_fn=None)
        skip_layers = []
        out_layers = []
        for idx in range(self.cnn_layer_config['num_layers']):
            with tf.variable_scope('temporal_block_' + str(idx + 1), reuse=self.reuse):
                temp_block, skip_out = CCN.temporal_block_ccn(input_layer=current_layer,
                                                              num_filters=self.cnn_layer_config['num_filters'],
                                                              kernel_size=self.cnn_layer_config['filter_size'],
                                                              dilation=self.cnn_layer_config['dilation_size'][idx],
                                                              activation_fn=self.activation_fn,
                                                              num_extra_conv=self.cnn_layer_config['num_conv_layers_per_block'],
                                                              use_gate=self.use_gate,
                                                              use_residual=self.use_residual,
                                                              zero_padding=self.zero_padding)
                if self.use_skip:
                    skip_layers.append(tf.slice(skip_out,
                                                [0, tf.shape(skip_out)[1] - self.output_width, 0],
                                                [-1, -1, -1]))
                current_layer = temp_block
                out_layers.append(temp_block)

        if self.use_skip:
            # Sum skip connections from the outputs of each layer.
            self.temporal_block_outputs = self.activation_fn(sum(skip_layers))
        else:
            tcn_output_layers = []
            for idx in self.tcn_output_layer_idx:
                tcn_output_layers.append(out_layers[idx])

            self.temporal_block_outputs = self.activation_fn(tf.concat(tcn_output_layers, axis=-1))

    def build_output_layer(self):
        """
        Builds a number fully connected layers projecting CNN representations onto output space. Then, outputs are
        predicted by linear layers.

        Returns:
        """
        if self.cnn_layer_config.get('use_dense_output_layer', False):
            flat_outputs_hidden = self.flat_tensor(self.temporal_block_outputs)
            with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
                flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training,
                                                            **self.output_layer_config)

            for idx in range(len(self.output_layer_config['out_keys'])):
                key = self.output_layer_config['out_keys'][idx]
                with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                    flat_out = linear(input_layer=flat_outputs_hidden,
                                      output_size=self.output_layer_config['out_dims'][idx],
                                      activation_fn=self.output_layer_config['out_activation_fn'][idx],
                                      is_training=self.is_training)

                    self.ops_model_output[key] = self.temporal_tensor(flat_out)
        else:
            with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
                current_layer = self.temporal_block_outputs
                for idx in range(self.output_layer_config.get('num_layers', 1)):
                    with tf.variable_scope('conv1d_' + str(idx + 1), reuse=self.reuse):
                        current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                         filters=self.cnn_layer_config['num_filters'], dilation_rate=1,
                                                         activation=self.activation_fn)
                outputs_hidden = current_layer
            for idx in range(len(self.output_layer_config['out_keys'])):
                key = self.output_layer_config['out_keys'][idx]
                with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                    output = tf.layers.conv1d(inputs=outputs_hidden,
                                              filters=self.output_layer_config['out_dims'][idx],
                                              kernel_size=1,
                                              padding='valid',
                                              activation=get_activation_fn(self.output_layer_config['out_activation_fn'][idx]))
                    self.ops_model_output[key] = output

        # Trim initial steps corresponding to the receptive field.
        self.seq_loss_mask = tf.slice(self.seq_loss_mask, [0, tf.shape(self.seq_loss_mask)[1] - self.output_width, 0], [-1, -1, -1])
        for idx, target in enumerate(self.target_pieces):
            self.target_pieces[idx] = tf.slice(target, [0, tf.shape(target)[1] - self.output_width, 0], [-1, -1, -1])

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.OUT_MU])[-1], tf.float32)
        if C.OUT_MU in self.ops_model_output:
            self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU]*self.seq_loss_mask)/num_entries
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.ops_model_output[C.OUT_SIGMA]*self.seq_loss_mask)/num_entries

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:

        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size). Due to causality constraint, number of
            prediction steps is input_seq_len-receptive_field_width. We simply take the first <receptive_field_width>
            many steps from the input sequence to pad reconstructed sequence.
        """
        input_sequence = kwargs.get('input_sequence', None)
        target_sequence = kwargs.get('target_sequence', None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)
        input_seq_len = input_sequence.shape[1]
        if self.zero_padding is False:
            assert input_seq_len >= self.receptive_field_width, "Input sequence should have at least " + str(self.receptive_field_width) + " steps."

        feed_dict = {self.pl_inputs: input_sequence}

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation['loss'] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence
            feed_dict[self.pl_seq_length] = np.array([target_sequence.shape[1]]*target_sequence.shape[0])

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs['loss'])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample(self, **kwargs):
        """
        Sampling function.
        Args:
            **kwargs:
        """
        seed_sequence = kwargs.get('seed_sequence', None)
        sample_length = kwargs.get('sample_length', 100)

        assert seed_sequence is not None, "Need a seed sample."
        batch_dimension = seed_sequence.ndim == 3
        if batch_dimension is False:
            seed_sequence = np.expand_dims(seed_sequence, axis=0)
        seed_len = seed_sequence.shape[1]
        if self.zero_padding is False:
            assert seed_len >= self.receptive_field_width, "Seed sequence should have at least " + str(self.receptive_field_width) + " steps."

        model_input = seed_sequence[:, -self.receptive_field_width:]
        model_outputs = self.sample_function(model_input, sample_length)

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample_function(self, model_input, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
            model_input (batch_size, seq_len, feature_size): seed sequence which must have at least
                self.receptive_field_width many steps.
            sample_length (int): number of sample steps.

        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        sequence = model_input.copy()
        for step in range(sample_length):
            model_input = sequence[:, -self.receptive_field_width:]
            model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: model_input})

            next_step = model_outputs['sample'][:, -1:]
            sequence = np.concatenate([sequence, next_step], axis=1)
        return {"sample": sequence[:, -sample_length:]}


class TCN(CCN):
    """
    Temporal convolutional network from `An Empirical Evaluation of Generic Convolutional and Recurrent Networks
    for Sequence Modeling` paper (https://arxiv.org/abs/1803.01271).

    Ensures that the number of output steps is equal to the number of input steps by applying padding at the start
    of input sequence.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(TCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

    @staticmethod
    def receptive_field_size(filter_size, dilation_size_list):
        # 2* is due to the second causal convolution layer in a temporal block.
        # TODO Not sure if this is correct. For now we don't need it.
        return 2*(filter_size - 1)*sum(dilation_size_list) + 1

    @staticmethod
    def temporal_block_tcn(input_layer, num_filters, kernel_size, dilation, activation_fn, dropout_rate, use_layer_norm=False, use_residual=True, is_training=True):
        with tf.name_scope('causal_layer_1'):
            temp_out = CCN.causal_conv_layer(input_layer, num_filters, kernel_size, dilation, zero_padding=True, activation_fn=activation_fn)

        if use_layer_norm:
            with tf.name_scope('layer_norm_1'):
                temp_out = tf.contrib.layers.layer_norm(temp_out)

        if dropout_rate > 0:
            with tf.name_scope('dropout_layer_1'):
                temp_out = tf.layers.dropout(temp_out, dropout_rate, noise_shape=[tf.shape(temp_out)[0], 1, tf.shape(temp_out)[2]], seed=17, training=is_training)

        with tf.name_scope('causal_layer_2'):
            temp_out = CCN.causal_conv_layer(temp_out, num_filters, kernel_size, dilation, zero_padding=True, activation_fn=activation_fn)

        if use_layer_norm:
            with tf.name_scope('layer_norm_2'):
                temp_out = tf.contrib.layers.layer_norm(temp_out)

        if dropout_rate > 0:
            with tf.name_scope('dropout_layer_2'):
                temp_out = tf.layers.dropout(temp_out, dropout_rate, noise_shape=[tf.shape(temp_out)[0], 1, tf.shape(temp_out)[2]], seed=17, training=is_training)

        skip_out = temp_out
        if use_residual is True:
            with tf.name_scope('residual_layer'):
                res_layer = input_layer
                if input_layer.shape[2] != num_filters:
                    res_layer = tf.layers.conv1d(inputs=input_layer,
                                                 filters=num_filters,
                                                 kernel_size=1,
                                                 padding='valid',
                                                 dilation_rate=1,
                                                 activation=None)
                temp_out = activation_fn(temp_out + res_layer)

        return temp_out, skip_out

    def build_causal_convolutions(self):
        current_layer = self.inputs_hidden
        self.receptive_field_width = TCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        # We always pad input sequences such that the output sequence has the same length with input sequence.
        self.output_width = tf.shape(current_layer)[1]

        skip_layers = []
        out_layers = []
        for idx in range(self.cnn_layer_config['num_layers']):
            with tf.variable_scope('temporal_block_'+str(idx+1), reuse=self.reuse):
                temp_block, skip_out = TCN.temporal_block_tcn(input_layer=current_layer,
                                                              num_filters=self.cnn_layer_config['num_filters'],
                                                              kernel_size=self.cnn_layer_config['filter_size'],
                                                              dilation=self.cnn_layer_config['dilation_size'][idx],
                                                              activation_fn=self.activation_fn,
                                                              dropout_rate=self.cnn_layer_config.get('dropout_rate', 0),
                                                              use_layer_norm=self.cnn_layer_config.get('use_layer_norm', False),
                                                              use_residual=self.use_residual,
                                                              is_training=self.is_training)
                if self.use_skip:
                    skip_layers.append(temp_block)

                current_layer = temp_block
                out_layers.append(temp_block)

        if self.use_skip:
            # Sum skip connections from the outputs of each layer.
            self.temporal_block_outputs = self.activation_fn(sum(skip_layers))
        else:
            tcn_output_layers = []
            for idx in self.tcn_output_layer_idx:
                tcn_output_layers.append(out_layers[idx])

            self.temporal_block_outputs = self.activation_fn(tf.concat(tcn_output_layers, axis=-1))

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:

        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size).
        """
        input_sequence = kwargs.get('input_sequence', None)
        target_sequence = kwargs.get('target_sequence', None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = {self.pl_inputs: input_sequence}

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation['loss'] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence
            feed_dict[self.pl_seq_length] = np.array([target_sequence.shape[1]]*target_sequence.shape[0])

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs['loss'])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs['sample'][0]

        return model_outputs

    def sample(self, **kwargs):
        """
        Sampling function.
        Args:
            **kwargs:
        """
        seed_sequence = kwargs.get('seed_sequence', None)
        sample_length = kwargs.get('sample_length', 100)

        assert seed_sequence is not None, "Need a seed sample."
        batch_dimension = seed_sequence.ndim == 3
        if batch_dimension is False:
            seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # model_input = seed_sequence[:, -self.receptive_field_width:]
        model_input = seed_sequence
        model_outputs = self.sample_function(model_input, sample_length)

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample_function(self, model_input, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
            model_input (batch_size, seq_len, feature_size): seed sequence which must have at least
                self.receptive_field_width many steps.
            sample_length (int): number of sample steps.

        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        sequence = model_input.copy()
        for step in range(sample_length):
            model_input = sequence
            model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: model_input})

            next_step = model_outputs['sample'][:, -1:]
            sequence = np.concatenate([sequence, next_step], axis=1)
        return {"sample": sequence[:, -sample_length:]}


class BiTCN(TCN):
    """
    Bidirectional temporal convolutional model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(BiTCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.stacked_forward_backward_blocks = self.config.get('stacked_forward_backward_blocks', False)
        self.temporal_block_type = self.config.get("temporal_block_type", C.MODEL_TCN)

        self.receptive_field_width = TCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.output_width = tf.shape(self.pl_inputs)[1]

        # Raw model inputs in reverse order.
        with tf.variable_scope("reverted_input", reuse=self.reuse):
            self.pl_inputs_reverse = tf.reverse_sequence(self.pl_inputs, self.pl_seq_length, seq_dim=1, batch_dim=0)

    def build_network(self):
        f_current_layer = self.pl_inputs
        b_current_layer = self.pl_inputs_reverse

        for idx in range(self.cnn_layer_config['num_layers']):
            with tf.variable_scope("f_temporal_block_" + str(idx + 1), reuse=self.reuse):
                f_temp_block, f_temp_wo_res = self.build_temporal_block(f_current_layer, idx)

            with tf.variable_scope("b_temporal_block_" + str(idx + 1), reuse=self.reuse):
                b_temp_block_reverse, b_temp_wo_res_reverse = self.build_temporal_block(b_current_layer, idx)

            with tf.variable_scope("reverted_input_" + str(idx + 1), reuse=self.reuse):
                if self.stacked_forward_backward_blocks:
                    b_temp_block = tf.reverse_sequence(b_temp_block_reverse, self.pl_seq_length, seq_dim=1, batch_dim=0)
                    f_temp_block_reverse = tf.reverse_sequence(f_temp_block, self.pl_seq_length, seq_dim=1, batch_dim=0)
                    f_current_layer = tf.concat([f_temp_block, b_temp_block], axis=-1)
                    b_current_layer = tf.concat([f_temp_block_reverse, b_temp_block_reverse], axis=-1)
                else:
                    f_current_layer = f_temp_block
                    b_current_layer = b_temp_block_reverse

        with tf.name_scope("decoder"):
            if self.stacked_forward_backward_blocks:
                self.temporal_block_outputs = f_current_layer
            else:
                b_current_layer = tf.reverse_sequence(b_current_layer, self.pl_seq_length, seq_dim=1, batch_dim=0)
                self.temporal_block_outputs = tf.concat([f_current_layer, b_current_layer], axis=-1)

        self.output_width = tf.shape(self.temporal_block_outputs)[1]
        self.build_output_layer()

    def build_temporal_block(self, input_layer, layer_idx):
        if self.temporal_block_type == C.MODEL_TCN:
            temp_block, temp_wo_res = TCN.temporal_block_tcn(input_layer=input_layer,
                                                             num_filters=self.cnn_layer_config['num_filters'],
                                                             kernel_size=self.cnn_layer_config['filter_size'],
                                                             dilation=self.cnn_layer_config['dilation_size'][layer_idx],
                                                             activation_fn=self.activation_fn,
                                                             dropout_rate=self.cnn_layer_config.get('dropout_rate', 0),
                                                             use_layer_norm=self.cnn_layer_config.get('use_layer_norm', False),
                                                             use_residual=self.use_residual,
                                                             is_training=self.is_training)
        elif self.temporal_block_type == C.MODEL_CCN:
            temp_block, temp_wo_res = CCN.temporal_block_ccn(input_layer=input_layer,
                                                             num_filters=self.cnn_layer_config['num_filters'],
                                                             kernel_size=self.cnn_layer_config['filter_size'],
                                                             dilation=self.cnn_layer_config['dilation_size'][layer_idx],
                                                             activation_fn=self.activation_fn,
                                                             num_extra_conv=0,
                                                             use_gate=self.use_gate,
                                                             use_residual=self.use_residual,
                                                             zero_padding=self.zero_padding)
        else:
            raise Exception("Unknown model type.")

        return temp_block, temp_wo_res


class EmbeddedSTCN(TCN):
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(EmbeddedSTCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.temporal_block_type = self.config.get("temporal_block_type", C.MODEL_TCN)
        self.decoder_use_enc_prev = self.config.get('decoder_use_enc_prev', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.num_encoder_blocks = self.cnn_layer_config.get('num_encoder_layers', self.cnn_layer_config['num_layers'])
        self.num_decoder_blocks = self.cnn_layer_config.get('num_decoder_layers', self.cnn_layer_config['num_layers'])

        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")

        # List of temporal convolution layers that are used in encoder.
        self.encoder_blocks = []
        # List of temporal convolution layers for skip connection.
        self.encoder_skip_blocks = []
        # List of temporal convolution layers that are used in decoder.
        self.decoder_blocks = []
        # List of temporal convolution layers for skip connection.
        self.decoder_skip_blocks = []

        #################
        #  Ladder latent
        #################
        self.latent_dims = self.latent_layer_config['latent_size']
        self.vertical_dilation = self.latent_layer_config.get('vertical_dilation', 1)
        self.dynamic_prior = self.latent_layer_config.get('dynamic_prior', False)  # Prior is estimated by using the past inputs.
        # Approximate posterior is estimated as a precision weighted update of the prior and initial model predictions.
        self.precision_weighted_update = self.latent_layer_config.get('precision_weighted_update', True)
        # Use the same parameters across hierarchical latent layers. Note that the latent sample and deterministic model
        # dimensionality of every layer must be the same.
        self.share_latent_params = self.latent_layer_config.get('share_latent_params', True)
        self.use_p_input_in_q = self.latent_layer_config.get('use_p_input_in_q', False)
        # In order to increase robustness of the learned prior use samples from the prior instead of the posterior.
        self.p_q_replacement_ratio = self.latent_layer_config.get('p_q_replacement_ratio', 0)
        # Whether the q distribution is hierarchically updated as in the case of prior or not. In other words, lower
        # q layer uses samples of the upper q layer.
        self.recursive_q = self.latent_layer_config.get('recursive_q', True)
        self.latent_layer_structure = self.latent_layer_config.get('layer_structure', C.LAYER_FC)
        self.use_all_z = self.latent_layer_config.get('use_all_z', False)
        self.use_skip_latent = self.latent_layer_config.get('use_skip_latent', False)
        self.collect_latent_samples = self.use_all_z or self.use_skip_latent

        kld_weight = self.latent_layer_config.get('kld_weight', 0.5)
        if isinstance(kld_weight, dict) and self.global_step:
            self.kld_weight = get_decay_variable(global_step=self.global_step, config=kld_weight, name="kld_weight")
        else:
            self.kld_weight = kld_weight
        if not self.is_training:
            self.kld_weight = 1.0

        # Latent space components.
        self.p_mu = []
        self.q_mu = []
        self.p_sigma = []
        self.q_sigma = []

        self.num_d_layers = None  # Total number of deterministic layers.
        self.num_s_layers = None  # Total number of stochastic layers can be different due to the vertical_dilation.
        self.q_approximate = None  # List of approximate q distributions from the recognition network.
        self.q_dists = None  # List of q distributions after updating with p_dists.
        self.p_dists = None  # List of prior distributions.
        self.kld_loss_terms = []  # List of KLD loss term.
        self.latent_samples = []  # List of latent samples.

    def build_network(self):
        self.build_input_layer()

        # We always pad input sequences such that the output sequence has the same length with input sequence.
        self.receptive_field_width = TCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        with tf.name_scope("input_padding"):
            shifted_inputs = tf.pad(self.inputs_hidden, tf.constant([(0, 0,), (1, 0), (0, 0)]), mode='CONSTANT')

        self.num_d_layers = self.num_encoder_blocks
        assert self.num_d_layers % self.vertical_dilation == 0, "# of deterministic layers must be divisible by vertical dilation."
        self.num_s_layers = int(self.num_d_layers/self.vertical_dilation)
        self.latent_dims = self.latent_dims if isinstance(self.latent_dims, list) else [self.latent_dims]*self.num_s_layers
        self.q_approximate = [0]*self.num_s_layers
        self.q_dists = [0]*self.num_s_layers
        self.p_dists = [0]*self.num_s_layers

        with tf.variable_scope("encoder", reuse=self.reuse):
            # Initialize the prior and posterior with zero-mean, unit-variance Gaussian.
            with tf.name_scope("stochastic_layer_0"):
                prior_shape = (tf.shape(shifted_inputs)[0], tf.shape(shifted_inputs)[1], self.latent_dims[0])
                posterior = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))
                p_dist = (tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32))

            loop_indices = range(0, self.num_d_layers, 1)
            current_layer = shifted_inputs
            latent_layer_created = False
            for dl in loop_indices:
                # Build deterministic layer.
                temp_out, skip_out = self.build_temporal_block(current_layer, dl, self.reuse, kernel_size=self.cnn_layer_config['filter_size'])
                self.encoder_blocks.append(temp_out)
                self.encoder_skip_blocks.append(skip_out)

                # Build stochastic layer.
                if (dl+1) % self.vertical_dilation == 0:
                    sl = int((dl+1)/self.vertical_dilation) - 1
                    with tf.name_scope("stochastic_layer_" + str(sl+1)):
                        p_input = skip_out  # [:, 0:-1]
                        q_input = skip_out  # [:, 1:]

                        # Estimate the prior distribution.
                        scope = C.LATENT_P if self.share_latent_params else C.LATENT_P + "_" + str(sl)
                        reuse = True if (self.share_latent_params and latent_layer_created) else self.reuse
                        # Draw a latent sample from the preceding posterior.
                        p_layer_inputs = [self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], scope)]
                        if self.dynamic_prior:
                            p_layer_inputs.append(p_input)
                        p_dist, _ = self.build_latent_dist(tf.concat(p_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
                        self.p_dists[sl] = [p_dist[0][:, 0:-1], p_dist[1][:, 0:-1]]

                        if self.is_sampling:
                            # Set the posterior.
                            posterior = p_dist
                        else:
                            # Estimate the uncorrected approximate posterior distribution.
                            scope = C.LATENT_Q if self.share_latent_params else C.LATENT_Q + "_app" + "_" + str(sl)
                            reuse = True if (self.share_latent_params and latent_layer_created) else self.reuse

                            q_layer_inputs = [q_input]
                            if self.recursive_q:
                                # Draw a latent sample from the preceding posterior.
                                q_layer_inputs.append(self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], scope))
                            if self.use_p_input_in_q:
                                q_layer_inputs.append(p_input)
                            q_dist_approx, _ = self.build_latent_dist(tf.concat(q_layer_inputs, axis=-1), idx=sl, scope=scope, reuse=reuse)
                            self.q_approximate[sl] = q_dist_approx

                            # Estimate the approximate posterior distribution as a precision-weighted combination.
                            if self.precision_weighted_update:
                                scope = C.LATENT_Q + "_pwu_" + str(sl)
                                posterior = self.combine_normal_dist(q_dist_approx, p_dist, scope=scope)
                            else:
                                posterior = q_dist_approx
                            self.q_dists[sl] = [posterior[0][:, 1:], posterior[1][:, 1:]]

                        posterior_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], scope)
                        if self.collect_latent_samples:
                            self.latent_samples.append(posterior_sample)
                        current_layer = tf.concat([temp_out, posterior_sample], axis=-1)
                        latent_layer_created = True
                else:
                    current_layer = temp_out

            if self.use_all_z:
                # Concatenate the latent samples of all stochastic layers.
                latent_sample = tf.concat(self.latent_samples, axis=-1)
            elif self.use_skip_latent:
                # Get the summation of all latent samples.
                latent_sample = sum(self.latent_samples)
            else:
                # Use a latent sample from the final stochastic layer.
                latent_sample = self.draw_latent_sample(posterior[0], posterior[1], p_dist[0], p_dist[1], "latent_sample")
            if self.is_sampling:
                latent_sample = latent_sample[:, 0:-1]
            else:
                latent_sample = latent_sample[:, 1:]

        with tf.variable_scope("decoder", reuse=self.reuse):
            decoder_inputs = [latent_sample]
            if self.use_skip is True:
                encoder_skip_out = [enc_layer[:, 0:-1] for enc_layer in self.encoder_skip_blocks]
                decoder_inputs.append(self.activation_fn(sum(encoder_skip_out)))
            elif self.decoder_use_enc_prev:
                decoder_inputs.append(self.encoder_blocks[-1][:, 0:-1])

            if self.decoder_use_raw_inputs:
                decoder_inputs.append(shifted_inputs[:, 0:-1])

            decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
            decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
            current_layer = decoder_input_layer
            for idx in range(self.num_decoder_blocks):
                temp_out, skip_out = self.build_temporal_block(current_layer, idx, self.reuse, kernel_size=decoder_filter_size)
                self.decoder_blocks.append(temp_out)
                self.decoder_skip_blocks.append(skip_out)
                current_layer = temp_out
            decoder = current_layer

        self.output_width = tf.shape(decoder_input_layer)[1]
        self.temporal_block_outputs = decoder
        self.build_output_layer()

    def build_temporal_block(self, input_layer, idx, reuse, kernel_size=2):
        with tf.variable_scope('temporal_block_' + str(idx+1), reuse=reuse):
            if self.temporal_block_type == C.MODEL_TCN:
                temp_block, temp_wo_res = TCN.temporal_block_tcn(input_layer=input_layer,
                                                                 num_filters=self.cnn_layer_config['num_filters'],
                                                                 kernel_size=kernel_size,
                                                                 dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                 activation_fn=self.activation_fn,
                                                                 dropout_rate=self.cnn_layer_config.get('dropout_rate', 0),
                                                                 use_layer_norm=self.cnn_layer_config.get('use_layer_norm', False),
                                                                 use_residual=self.use_residual,
                                                                 is_training=self.is_training)
            elif self.temporal_block_type == C.MODEL_CCN:
                temp_block, temp_wo_res = CCN.temporal_block_ccn(input_layer=input_layer,
                                                                 num_filters=self.cnn_layer_config['num_filters'],
                                                                 kernel_size=kernel_size,
                                                                 dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                 activation_fn=self.activation_fn,
                                                                 num_extra_conv=0,
                                                                 use_gate=self.use_gate,
                                                                 use_residual=self.use_residual,
                                                                 zero_padding=self.zero_padding)
            else:
                raise Exception("Unknown model type.")
        return temp_block, temp_wo_res

    def build_latent_dist_fc(self, input_, idx, scope, reuse):
        with tf.name_scope(scope):
            with tf.variable_scope(scope+'_mu', reuse=reuse):
                mu, flat_mu = LatentLayer.build_fc_layer(input_layer=input_,
                                                         num_latent_units=self.latent_dims[idx],
                                                         latent_activation_fn=None,
                                                         num_hidden_layers=self.latent_layer_config["num_hidden_layers"],
                                                         num_hidden_units=self.latent_layer_config["num_hidden_units"],
                                                         hidden_activation_fn=self.latent_layer_config["hidden_activation_fn"],
                                                         is_training=self.is_training)
            with tf.variable_scope(scope+'_sigma', reuse=reuse):
                sigma, flat_sigma = LatentLayer.build_fc_layer(input_layer=input_,
                                                               num_latent_units=self.latent_dims[idx],
                                                               latent_activation_fn=tf.nn.softplus,
                                                               num_hidden_layers=self.latent_layer_config["num_hidden_layers"],
                                                               num_hidden_units=self.latent_layer_config["num_hidden_units"],
                                                               hidden_activation_fn=self.latent_layer_config["hidden_activation_fn"],
                                                               is_training=self.is_training)
                if self.latent_layer_config.get('latent_sigma_threshold', 0) > 0:
                    sigma = tf.clip_by_value(sigma, 1e-3, self.latent_layer_config.get('latent_sigma_threshold'))
                    flat_sigma = tf.clip_by_value(flat_sigma, 1e-3, self.latent_layer_config.get('latent_sigma_threshold'))

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
            raise Exception("Not implemented.")
        else:
            raise Exception("Unknown layer.")

    def draw_latent_sample(self, posterior_mu, posterior_sigma, prior_mu, prior_sigma, scope):
        """
        Draws a latent sample by using the reparameterization trick.
        Args:
            prior_mu:
            prior_sigma:
            posterior_mu:
            posterior_sigma:
            scope
        Returns:
        """
        def normal_sample(mu, sigma):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            return tf.add(mu, tf.multiply(sigma, eps))

        if self.p_q_replacement_ratio > 0 and prior_mu is not None:
            with tf.name_scope(scope+"_z"):
                z = tf.cond(pred=tf.random_uniform([1])[0] < self.p_q_replacement_ratio,
                            true_fn=lambda: normal_sample(prior_mu, prior_sigma),
                            false_fn=lambda: normal_sample(posterior_mu, posterior_sigma))
        else:
            with tf.name_scope(scope+"_z"):
                z = normal_sample(posterior_mu, posterior_sigma)
        return z

    def combine_normal_dist(self, dist1, dist2, scope):
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

    def build_loss_terms(self):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions. If use_temporal_kld is True,
        then creates another KL-divergence term between consecutive approximate posteriors in time.
        """
        super(EmbeddedSTCN, self).build_loss_terms()

        if not self.is_sampling:
            loss_key = "loss_kld"
            kld_loss = 0.0
            with tf.name_scope("kld_loss"):
                for sl in range(self.num_s_layers-1, -1, -1):
                    with tf.name_scope("kld_" + str(sl)):
                        kld_term = self.kld_weight*self.reduce_loss_fn(
                            self.seq_loss_mask*tf_loss.kld_normal_isotropic(self.q_dists[sl][0],
                                                                            self.q_dists[sl][1],
                                                                            self.p_dists[sl][0],
                                                                            self.p_dists[sl][1],
                                                                            reduce_sum=False))
                        # This is just for reporting. Any entry in loss_ops_dict is used in optimization by default.
                        # Optimization is done through the accumulated term (i.e., loss_ops_dict[loss_key]).
                        # loss_ops_dict[loss_key+"_"+str(sl)] = tf.stop_gradient(kld_term)
                        self.kld_loss_terms.append(kld_term)
                        kld_loss += kld_term
                self.ops_loss[loss_key] = kld_loss

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        if self.cnn_layer_config.get('use_dense_output_layer', False):
            flat_outputs_hidden = self.flat_tensor(self.temporal_block_outputs)
            with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
                flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

            for idx in range(len(self.output_layer_config['out_keys'])):
                key = self.output_layer_config['out_keys'][idx]
                with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                    flat_out = linear(input_layer=flat_outputs_hidden,
                                      output_size=self.output_layer_config['out_dims'][idx],
                                      activation_fn=self.output_layer_config['out_activation_fn'][idx],
                                      is_training=self.is_training)

                    self.ops_model_output[key] = self.temporal_tensor(flat_out)
        else:
            with tf.variable_scope('output_layer', reuse=self.reuse):
                current_layer = self.temporal_block_outputs
                for idx in range(1):
                    with tf.variable_scope('out_conv1d_' + str(idx + 1), reuse=self.reuse):
                        current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                         filters=self.cnn_layer_config['num_filters'], dilation_rate=1,
                                                         activation=self.activation_fn)

                for idx in range(len(self.output_layer_config['out_keys'])):
                    key = self.output_layer_config['out_keys'][idx]
                    with tf.variable_scope('out_' + key, reuse=self.reuse):
                        out_activation = get_activation_fn(self.output_layer_config['out_activation_fn'][idx])
                        output = tf.layers.conv1d(inputs=current_layer,
                                                  filters=self.output_layer_config['out_dims'][idx],
                                                  kernel_size=1,
                                                  padding='valid',
                                                  activation=out_activation)
                        self.ops_model_output[key] = output

        self.seq_loss_mask = tf.slice(self.seq_loss_mask, [0, tf.shape(self.seq_loss_mask)[1] - self.output_width, 0], [-1, -1, -1])
        for idx, target in enumerate(self.target_pieces):
            self.target_pieces[idx] = tf.slice(target, [0, tf.shape(target)[1] - self.output_width, 0], [-1, -1, -1])

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.OUT_MU])[-1], tf.float32)
        if C.OUT_MU in self.ops_model_output:
            self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU]*self.seq_loss_mask)/num_entries
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.ops_model_output[C.OUT_SIGMA]*self.seq_loss_mask)/num_entries

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def sample_function(self, model_input, sample_length):
        """
        Update: From now on we assume that the causal relationship between the inputs and targets are handled by dataset.
        Hence, we don't need to insert a dummy step.

        Auxiliary method to draw sequence of samples in auto-regressive fashion. We use prior distribution to sample
        next step.
        Args:
            model_input (batch_size, seq_len, feature_size): seed sequence which must have at least
                self.receptive_field_width many steps.
            sample_length (int): number of sample steps.

        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        assert self.is_sampling, "The model must be in sampling mode."

        # For each evaluation op, create a dummy output.
        output_dict = dict()
        for key, op in self.ops_evaluation.items():
            output_dict[key] = np.zeros((model_input.shape[0], 0, op.shape[2]))
        output_dict["sample"] = model_input.copy()

        dummy_x = np.zeros([model_input.shape[0], 1, model_input.shape[2]])
        for step in range(sample_length):
            model_inputs = np.concatenate([output_dict["sample"], dummy_x], axis=1)
            feed_dict = dict()
            feed_dict[self.pl_inputs] = model_inputs
            feed_dict[self.pl_seq_length] = np.array([model_inputs.shape[1]]*model_inputs.shape[0])
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)

            for key, val in model_outputs.items():
                output_dict[key] = np.concatenate([output_dict[key], val[:, -1:]], axis=1)

        output_dict["sample"] = output_dict["sample"][:, -sample_length:]
        return output_dict


class StochasticTCN(TCN):
    """
    Temporal convolutional model with stochastic latent space.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(StochasticTCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.temporal_block_type = self.config.get("temporal_block_type", C.MODEL_TCN)
        self.decoder_use_enc_prev = self.config.get('decoder_use_enc_prev', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.num_encoder_blocks = self.cnn_layer_config.get('num_encoder_layers', self.cnn_layer_config['num_layers'])
        self.num_decoder_blocks = self.cnn_layer_config.get('num_decoder_layers', self.cnn_layer_config['num_layers'])

        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")
        self.latent_layer = LatentLayer.get(self.latent_layer_config["type"], self.latent_layer_config, mode, reuse, global_step=self.global_step)

        # List of temporal convolution layers that are used in encoder.
        self.encoder_blocks = []
        self.encoder_blocks_no_res = []
        # List of temporal convolution layers that are used in decoder.
        self.decoder_blocks = []
        self.decoder_blocks_no_res = []

        self.use_future_steps_in_q = self.config.get('use_future_steps_in_q', False)
        self.bw_encoder_blocks = []
        self.bw_encoder_blocks_no_res = []

    def build_network(self):
        # We always pad the input sequences such that the output sequence has the same length with input sequence.
        if self.temporal_block_type == C.MODEL_TCN:
            self.receptive_field_width = TCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        elif self.temporal_block_type == C.MODEL_CCN:
            self.receptive_field_width = CCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])

        # Shift the input sequence by one step so that the task is prediction of the next step.
        with tf.name_scope("input_padding"):
            shifted_inputs = tf.pad(self.pl_inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]), mode='CONSTANT')

        self.inputs_hidden = shifted_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(shifted_inputs, rate=self.input_layer_config.get("dropout_rate"), seed=self.config.seed, training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

        if self.use_future_steps_in_q:
            reuse_params_in_bw = True
            reversed_inputs = tf.manip.reverse(shifted_inputs, axis=[1])
            if reuse_params_in_bw:
                with tf.variable_scope("encoder", reuse=True):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, True, self.cnn_layer_config['filter_size'])
            else:
                with tf.variable_scope("bw_encoder", reuse=self.reuse):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

            self.bw_encoder_blocks = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks]
            self.bw_encoder_blocks_no_res = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks_no_res]

        with tf.variable_scope("latent", reuse=self.reuse):
            if self.latent_layer_config["type"] == C.LATENT_LADDER_GAUSSIAN:
                prev_enc_output = self.encoder_blocks[-1][:, 0:-1]  # Top-most convolutional layer (prior inputs).
                p_input = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks]
                if self.latent_layer_config.get('dynamic_prior', False):
                    if self.use_future_steps_in_q:
                        q_input = [tf.concat([fw_enc[:, 1:], bw_enc], axis=-1) for fw_enc, bw_enc in zip(self.encoder_blocks, self.bw_encoder_blocks)]
                    else:
                        q_input = [enc_layer[:, 1:] for enc_layer in self.encoder_blocks]
                else:
                    q_input = p_input
                latent_sample = self.latent_layer.build_latent_layer(q_input=q_input,
                                                                     p_input=p_input,
                                                                     output_ops_dict=self.ops_model_output,
                                                                     eval_ops_dict=self.ops_evaluation,
                                                                     summary_ops_dict=self.ops_scalar_summary)
            else:
                """
                The task is prediction of the next step by using the information provided until the current step.
                In other words, it is p_dists(x_{t+1} | x_{0:t}).

                Prior distribution is estimated by using information until the current time-step t. On the other hand,
                approximate-posterior distribution is estimated by using some future steps. In order to model the
                current step t, we use ceil(<latent_filter_size>/2)-1 many future steps. For example, it 1d filter
                width is 5, then the current step is modeled by using 2 future steps (i.e., x_{t-2}, x_{t-1}, x_{t},
                x_{t+1}, x_{t+2}).
                """
                # TODO
                raise Exception("Not adapted for non-shifted inputs/targets.")
                num_zero_padding_end = math.ceil(self.latent_layer_config['latent_filter_size']/2) - 1
                num_zero_padding_begin = math.floor(self.latent_layer_config['latent_filter_size']/2)
                q_input = tf.pad(encoder, tf.constant([(0, 0,), (num_zero_padding_begin, num_zero_padding_end), (0, 0)]), mode='CONSTANT')
                prev_enc_output = None
                latent_sample = self.latent_layer.build_latent_layer(q_input=q_input,
                                                                     p_input=encoder,
                                                                     output_ops_dict=self.ops_model_output,
                                                                     eval_ops_dict=self.ops_evaluation,
                                                                     summary_ops_dict=self.ops_scalar_summary)
        # Build causal decoder blocks if we have any. Otherwise, we just use a number of 1x1 convolutions in
        # build_output_layer. Note that there are several input options.
        if self.num_decoder_blocks > 0:
            with tf.variable_scope("decoder", reuse=self.reuse):
                decoder_inputs = [latent_sample]
                if self.use_skip is True:
                    skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks_no_res]
                    decoder_inputs.append(self.activation_fn(sum(skip_connections)))
                elif self.decoder_use_enc_prev:
                    decoder_inputs.append(prev_enc_output)

                if self.decoder_use_raw_inputs:
                    decoder_inputs.append(shifted_inputs[:, 0:-1])

                decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer, self.num_decoder_blocks, self.reuse, kernel_size=decoder_filter_size)

                if self.use_skip is True and len(self.decoder_blocks) > 0:
                    decoder = self.activation_fn(sum(self.decoder_blocks))

                # self.temporal_block_outputs = tf.concat([latent_sample, decoder], axis=-1)
                self.temporal_block_outputs = self.decoder_blocks[-1]
        else:
            output_layer_inps = [latent_sample]
            if self.use_skip is True:
                skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks_no_res]
                output_layer_inps.append(self.activation_fn(sum(skip_connections)))
            elif self.decoder_use_enc_prev:
                output_layer_inps.append(prev_enc_output)
            if self.decoder_use_raw_inputs:
                output_layer_inps.append(shifted_inputs[:, 0:-1])

            self.temporal_block_outputs = tf.concat(output_layer_inps, axis=-1)

        self.output_width = tf.shape(latent_sample)[1]
        self.build_output_layer()

    def build_temporal_block(self, input_layer, num_layers, reuse, kernel_size=2):
        current_layer = input_layer
        temporal_blocks = []
        temporal_blocks_no_res = []
        for idx in range(num_layers):
            with tf.variable_scope('temporal_block_' + str(idx + 1), reuse=reuse):
                if self.temporal_block_type == C.MODEL_TCN:
                    temp_block, temp_wo_res = TCN.temporal_block_tcn(input_layer=current_layer,
                                                                     num_filters=self.cnn_layer_config['num_filters'],
                                                                     kernel_size=kernel_size,
                                                                     dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                     activation_fn=self.activation_fn,
                                                                     dropout_rate=self.cnn_layer_config.get('dropout_rate', 0),
                                                                     use_layer_norm=self.cnn_layer_config.get('use_layer_norm', False),
                                                                     use_residual=self.use_residual,
                                                                     is_training=self.is_training)
                elif self.temporal_block_type == C.MODEL_CCN:
                    temp_block, temp_wo_res = CCN.temporal_block_ccn(input_layer=current_layer,
                                                                     num_filters=self.cnn_layer_config['num_filters'],
                                                                     kernel_size=kernel_size,
                                                                     dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                     activation_fn=self.activation_fn,
                                                                     num_extra_conv=0,
                                                                     use_gate=self.use_gate,
                                                                     use_residual=self.use_residual,
                                                                     zero_padding=self.zero_padding)
                else:
                    raise Exception("Unknown model type.")

                temporal_blocks_no_res.append(temp_wo_res)
                temporal_blocks.append(temp_block)
                current_layer = temp_block

        return temporal_blocks, temporal_blocks_no_res

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        out_layer_type = self.output_layer_config.get('type', None)
        if out_layer_type is None:
            if self.cnn_layer_config.get('use_dense_output_layer', False):
                out_layer_type = C.LAYER_FC
            else:
                out_layer_type = C.LAYER_TCN

        with tf.variable_scope('output_layer', reuse=self.reuse):
            if out_layer_type == C.LAYER_FC:
                flat_outputs_hidden = self.flat_tensor(self.temporal_block_outputs)
                with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
                    flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

                for idx in range(len(self.output_layer_config['out_keys'])):
                    key = self.output_layer_config['out_keys'][idx]
                    with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                        flat_out = linear(input_layer=flat_outputs_hidden,
                                          output_size=self.output_layer_config['out_dims'][idx],
                                          activation_fn=self.output_layer_config['out_activation_fn'][idx],
                                          is_training=self.is_training)
                        self.ops_model_output[key] = self.temporal_tensor(flat_out)

            elif out_layer_type == C.LAYER_CONV1 or out_layer_type == C.LAYER_TCN:
                current_layer = self.temporal_block_outputs
                num_filters = self.cnn_layer_config['num_filters'] if self.output_layer_config.get('size', 0) < 1 else self.output_layer_config.get('size')

                if out_layer_type == C.LAYER_CONV1:
                    for idx in range(self.output_layer_config.get('num_layers', 1)):
                        with tf.variable_scope('out_conv1d_' + str(idx + 1), reuse=self.reuse):
                            current_layer = tf.layers.conv1d(inputs=current_layer, kernel_size=1, padding='valid',
                                                             filters=num_filters, dilation_rate=1,
                                                             activation=self.activation_fn)
                if out_layer_type == C.LAYER_TCN:
                    kernel_size = self.cnn_layer_config['filter_size'] if self.output_layer_config.get('filter_size', 0) < 1 else self.output_layer_config.get('filter_size', 0)
                    for idx in range(self.output_layer_config.get('num_layers', 1)):
                        with tf.variable_scope('out_convCCN_' + str(idx + 1), reuse=self.reuse):
                            current_layer, _ = CCN.temporal_block_ccn(input_layer=current_layer,
                                                                      num_filters=num_filters,
                                                                      kernel_size=kernel_size,
                                                                      dilation=1,
                                                                      activation_fn=self.activation_fn,
                                                                      num_extra_conv=0,
                                                                      use_gate=self.use_gate,
                                                                      use_residual=self.use_residual,
                                                                      zero_padding=True)
                for idx in range(len(self.output_layer_config['out_keys'])):
                    key = self.output_layer_config['out_keys'][idx]
                    with tf.variable_scope('out_' + key, reuse=self.reuse):
                        out_activation = get_activation_fn(self.output_layer_config['out_activation_fn'][idx])
                        output = tf.layers.conv1d(inputs=current_layer,
                                                  filters=self.output_layer_config['out_dims'][idx],
                                                  kernel_size=1,
                                                  padding='valid',
                                                  activation=out_activation)
                        self.ops_model_output[key] = output

        self.seq_loss_mask = tf.slice(self.seq_loss_mask, [0, tf.shape(self.seq_loss_mask)[1] - self.output_width, 0], [-1, -1, -1])
        # for idx, target in enumerate(self.target_pieces):
        #    self.target_pieces[idx] = tf.slice(target, [0, tf.shape(target)[1] - self.output_width, 0], [-1, -1, -1])

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.OUT_MU])[-1], tf.float32)
        if C.OUT_MU in self.ops_model_output:
            self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU]*self.seq_loss_mask)/num_entries
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.ops_model_output[C.OUT_SIGMA]*self.seq_loss_mask)/num_entries

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        super(StochasticTCN, self).build_loss_terms()

        # Get latent layer loss terms, apply mask and reduce function, and insert into our loss container.
        if self.is_eval:
            self.latent_layer.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood, eval_dict=self.ops_for_eval_mode)
            self.ops_evaluation["eval_dict"] = self.ops_for_eval_mode
        else:
            self.latent_layer.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood)

    def build_summary_plots(self):
        super(StochasticTCN, self).build_summary_plots()

        # Create summaries to visualize distribution of latent variables.
        if self.config.get('tensorboard_verbose', 0) > 1:
            for idx, encoder_block in enumerate(self.encoder_blocks):
                plot_key = "encoder_block_" + str(idx + 1)
                tf.summary.histogram(plot_key, encoder_block, collections=[self.mode + '_summary_plot', self.mode + '_temporal_block_activations'])

            for idx, decoder_block in enumerate(self.decoder_blocks):
                plot_key = "decoder_block_" + str(idx + 1)
                tf.summary.histogram(plot_key, decoder_block, collections=[self.mode + '_summary_plot', self.mode + '_temporal_block_activations'])

    def sample_function(self, model_input, sample_length):
        """
        Update: From now on we assume that the causal relationship between the inputs and targets are handled by dataset.
        Hence, we don't need to insert a dummy step.

        Auxiliary method to draw sequence of samples in auto-regressive fashion. We use prior distribution to sample
        next step.
        Args:
            model_input (batch_size, seq_len, feature_size): seed sequence which must have at least
                self.receptive_field_width many steps.
            sample_length (int): number of sample steps.

        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """

        assert self.is_sampling, "The model must be in sampling mode."
        # For each evaluation op, create a dummy output.
        output_dict = dict()
        for key, op in self.ops_evaluation.items():
            output_dict[key] = np.zeros((model_input.shape[0], 0, op.shape[2]))
        output_dict["sample"] = model_input.copy()

        dummy_x = np.zeros([model_input.shape[0], 1, model_input.shape[2]])
        for step in range(sample_length):
            model_inputs = np.concatenate([output_dict["sample"], dummy_x], axis=1)
            end_idx = min(self.receptive_field_width, model_inputs.shape[1])
            model_inputs = model_inputs[:, -end_idx:]
            feed_dict = dict()
            feed_dict[self.pl_inputs] = model_inputs
            feed_dict[self.pl_seq_length] = np.array([model_inputs.shape[1]]*model_inputs.shape[0])
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)

            for key, val in model_outputs.items():
                output_dict[key] = np.concatenate([output_dict[key], val[:, -1:]], axis=1)

        output_dict["sample"] = output_dict["sample"][:, -sample_length:]
        return output_dict


class BiStochasticTCN(StochasticTCN):
    """
    Bidirectional temporal convolutional model with stochastic latent variables.
    """

    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(BiStochasticTCN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.tied_forward_backward_blocks = self.config.get('tied_forward_backward_blocks', False)

        self.receptive_field_width = TCN.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.output_width = tf.shape(self.pl_inputs)[1]

        # Raw model inputs in reverse order.
        with tf.variable_scope("reverted_input", reuse=self.reuse):
            self.pl_inputs_reverse = tf.reverse_sequence(self.pl_inputs, self.pl_seq_length, seq_dim=1, batch_dim=0)

        # List of temporal convolutional layers that are used in forward and backward encoder.
        self.forward_blocks = None
        self.backward_blocks = None
        # List of temporal convolutional layers that are used in decoder.
        self.decoder_blocks = None

    def build_network(self):
        f_variable_scope = "encoder" if self.tied_forward_backward_blocks else "f_encoder"
        b_variable_scope = "encoder" if self.tied_forward_backward_blocks else "b_encoder"

        with tf.name_scope("forward_encoder"):
            with tf.variable_scope(f_variable_scope, reuse=self.reuse):
                forward_encoder, self.forward_blocks = self.build_temporal_block(self.pl_inputs, self.num_encoder_blocks, self.reuse)

        reuse_backward = self.reuse or self.tied_forward_backward_blocks
        with tf.name_scope("backward_encoder"):
            with tf.variable_scope(b_variable_scope, reuse=reuse_backward):
                backward_encoder_reverse, self.backward_blocks = self.build_temporal_block(self.pl_inputs_reverse, self.num_encoder_blocks, reuse_backward)
                backward_encoder = tf.reverse_sequence(backward_encoder_reverse, self.pl_seq_length, seq_dim=1, batch_dim=0)

        with tf.variable_scope("latent", reuse=self.reuse):
            """
            The task is prediction of the next step by using the information provided until the current step.
            In other words, it is p_dists(x_{t+1} | x_{0:t}).

            Prior distribution is estimated by using information until the current time-step t. On the other hand,
            approximate-posterior distribution is estimated by using past and future steps. In order to model the
            current step t, we use both forward and backward encoder outputs.
            """
            q_input = tf.concat([forward_encoder, backward_encoder], axis=2)
            q_input_shape = q_input.shape.as_list()
            padding_steps = (self.latent_layer_config['latent_filter_size'] - 1)*self.latent_layer_config['latent_dilation']
            q_input = tf.pad(q_input, tf.constant([(0, 0,), (1, 0), (0, 0)])*padding_steps, mode='CONSTANT')
            if q_input_shape[1] is not None:
                q_input_shape[1] += padding_steps
            q_input.set_shape(q_input_shape)
            latent_sample = self.latent_layer.build_latent_layer(q_input=q_input,
                                                                 p_input=forward_encoder,
                                                                 output_ops_dict=self.ops_model_output,
                                                                 eval_ops_dict=self.ops_evaluation,
                                                                 summary_ops_dict=self.ops_scalar_summary)

        with tf.name_scope("decoder"):
            if self.decoder_use_enc_prev:
                decoder_input_layer = tf.concat([forward_encoder, latent_sample], axis=-1)
            elif self.decoder_use_raw_inputs:
                decoder_input_layer = tf.concat([self.pl_inputs, latent_sample], axis=-1)
            else:
                decoder_input_layer = latent_sample

            self.output_width = tf.shape(decoder_input_layer)[1]
            decoder, self.decoder_blocks = self.build_temporal_block(decoder_input_layer, self.num_decoder_blocks, self.reuse)

        self.temporal_block_outputs = decoder
        self.build_output_layer()


class BaseRNN(BaseTemporalModel):
    """
    Implements abstract build_graph and build_network methods to build an RNN model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(BaseRNN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.input_layer_config = config.get('input_layer')
        self.cell_config = config.get('rnn_layer')

        self.cell = None  # RNN cell.
        self.initial_states = None  # Initial cell state.
        self.rnn_outputs = None  # Output of RNN layer.
        self.rnn_output_state = None  # Final state of RNN layer.
        self.output_layer_inputs = None  # Input to output layer.

    def build_graph(self):
        """
        Builds model and creates plots for tensorboard. Decomposes model building into sub-modules and makes inheritance
        is easier.
        """
        self.build_network()
        self.build_loss_terms()
        self.build_total_loss()
        self.build_summary_plots()
        self.finalise_graph()
        if self.reuse is False:
            self.log_num_parameters()

    def build_network(self):
        self.build_cell()
        self.build_input_layer()
        self.build_rnn_layer()
        self.build_output_layer()

    def build_cell(self):
        """
        Builds a Tensorflow RNN cell object by using the given configuration `self.cell_config`.
        """
        self.cell = get_rnn_cell(scope='rnn_cell', reuse=self.reuse, **self.cell_config)
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_input_layer(self):
        """
        Builds a number fully connected layers projecting the inputs into an intermediate representation  space.
        """
        if self.input_layer_config is not None:
            with tf.variable_scope('input_layer', reuse=self.reuse):
                if self.input_layer_config.get("dropout_rate", 0) > 0:
                    self.inputs_hidden = tf.layers.dropout(self.pl_inputs,
                                                           rate=self.input_layer_config.get("dropout_rate"),
                                                           noise_shape=None,
                                                           seed=17,
                                                           training=self.is_training)
                else:
                    self.inputs_hidden = self.pl_inputs

                if self.input_layer_config.get("num_layers", 0) > 0:
                    flat_inputs_hidden = self.flat_tensor(self.inputs_hidden)
                    flat_inputs_hidden = fully_connected_layer(flat_inputs_hidden, **self.input_layer_config)
                    self.inputs_hidden = self.temporal_tensor(flat_inputs_hidden)
        else:
            self.inputs_hidden = self.pl_inputs

    def build_rnn_layer(self):
        """
        Builds RNN layer by using dynamic_rnn wrapper of Tensorflow.
        """
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_output_state = tf.nn.dynamic_rnn(self.cell,
                                                                        self.inputs_hidden,
                                                                        sequence_length=self.pl_seq_length,
                                                                        initial_state=self.initial_states,
                                                                        dtype=tf.float32)
            self.output_layer_inputs = self.rnn_outputs
            self.ops_evaluation['state'] = self.rnn_output_state

    def build_output_layer(self):
        """
        Builds a number fully connected layers projecting RNN predictions into an embedding space. Then, for each model
        output is predicted by a linear layer.
        """
        flat_outputs_hidden = self.flat_tensor(self.output_layer_inputs)
        with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
            flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

        for idx in range(len(self.output_layer_config['out_keys'])):
            key = self.output_layer_config['out_keys'][idx]

            with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                flat_out = linear(input_layer=flat_outputs_hidden,
                                  output_size=self.output_layer_config['out_dims'][idx],
                                  activation_fn=self.output_layer_config['out_activation_fn'][idx],
                                  is_training=self.is_training)

                self.ops_model_output[key] = self.temporal_tensor(flat_out)

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample


class RNNAutoRegressive(BaseRNN):
    """
    Auto-regressive RNN model. Predicts next step (t+1) given the current step (t). Note that here we assume targets are
    equivalent to inputs shifted by one step in time.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(RNNAutoRegressive, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

    def build_output_layer(self):
        # Prediction layer.
        BaseRNN.build_output_layer(self)

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.OUT_MU])[-1], tf.float32)
        if C.OUT_MU in self.ops_model_output:
            self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU]*self.seq_loss_mask)/num_entries
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.ops_model_output[C.OUT_SIGMA]*self.seq_loss_mask)/num_entries

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get('input_sequence', None)
        target_sequence = kwargs.get('target_sequence', None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = dict()
        feed_dict[self.pl_inputs] = input_sequence
        feed_dict[self.pl_seq_length] = np.array([input_sequence.shape[1]]*input_sequence.shape[0])

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation['loss'] = self.ops_loss
            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs['loss'])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample(self, **kwargs):
        """
        Sampling function.
        Args:
            **kwargs:
        """
        seed_sequence = kwargs.get('seed_sequence', None)
        sample_length = kwargs.get('sample_length', 100)

        assert seed_sequence is not None, "Need a seed sample."
        batch_dimension = seed_sequence.ndim == 3
        if batch_dimension is False:
            seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Feed seed sequence and update RNN state.
        if not("state" in self.ops_model_output):
            self.ops_evaluation["state"] = self.rnn_output_state
        model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: seed_sequence, self.pl_seq_length:np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]})

        # Get the last step.
        last_step = model_outputs['sample'][:, -1:, :]
        model_outputs = self.sample_function(last_step, model_outputs['state'], sample_length)

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample_function(self, current_input, previous_state, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        # TODO accumulate other evaluation results.
        sequence = current_input.copy()
        num_samples = sequence.shape[0]
        for step in range(sample_length):
            feed_dict = {self.pl_inputs     : sequence[:, -1:, :],
                         self.initial_states: previous_state,
                         self.pl_seq_length : np.ones(num_samples)}
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)
            previous_state = model_outputs['state']

            sequence = np.concatenate([sequence, model_outputs['sample']], axis=1)
        return {"sample": sequence[:, -sample_length:]}


class RNNLadder(RNNAutoRegressive):
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(RNNLadder, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.latent_layer_config = self.config.get("latent_layer")
        self.latent_layer = LatentLayer.get(self.latent_layer_config["type"], self.latent_layer_config, mode, reuse, global_step=self.global_step)

        self.input_layer_config = config.get('input_layer', None)
        self.rnn_layer_config = config.get('rnn_layer')

        self.activation_fn = get_activation_fn(self.rnn_layer_config['activation_fn'])
        self.num_units_per_cell = self.rnn_layer_config['size']
        self.num_layers = self.rnn_layer_config['num_layers']
        self.cell_type = self.rnn_layer_config['cell_type']

        self.output_layer_enc_prev = self.output_layer_config.get('use_enc_prev', False)
        self.output_layer_raw_inputs = self.output_layer_config.get('use_raw_inputs', False)
        self.output_layer_skip = self.output_layer_config.get('use_skip', False)

        self.cells = None
        self.initial_states = None
        self.rnn_layers = None

    def build_network(self):
        # Shift the input sequence by one step so that the task is prediction of the next step.
        with tf.name_scope("input_padding"):
            shifted_inputs = tf.pad(self.pl_inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]), mode='CONSTANT')

        self.inputs_hidden = shifted_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(shifted_inputs, rate=self.input_layer_config.get("dropout_rate"), seed=self.config.seed, training=self.is_training)

        self.cells = get_rnn_cell(cell_type=self.cell_type, size=self.num_units_per_cell, num_layers=self.num_layers, intermediate_outputs=True)
        # self.cells = get_rnn_cell(cell_type=self.cell_type, size=self.num_units_per_cell, num_layers=1)
        self.initial_states = self.cells.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.rnn_layers, self.rnn_output_state = tf.nn.dynamic_rnn(self.cells,
                                                                       self.inputs_hidden,
                                                                       sequence_length=self.pl_seq_length,
                                                                       initial_state=self.initial_states,
                                                                       dtype=tf.float32)
            if not isinstance(self.rnn_layers, list):  # 1 layer only.
                self.rnn_layers = [self.rnn_layers]

        with tf.variable_scope("latent", reuse=self.reuse):
            assert self.latent_layer_config["type"] == C.LATENT_LADDER_GAUSSIAN, "Unknown latent structure."
            p_input = [enc_layer[:, 0:-1] for enc_layer in self.rnn_layers]
            if self.latent_layer_config.get('dynamic_prior', False):
                q_input = [enc_layer[:, 1:] for enc_layer in self.rnn_layers]
            else:
                q_input = p_input

            latent_sample = self.latent_layer.build_latent_layer(q_input=q_input,
                                                                 p_input=p_input,
                                                                 output_ops_dict=self.ops_model_output,
                                                                 eval_ops_dict=self.ops_evaluation,
                                                                 summary_ops_dict=self.ops_scalar_summary)

        output_layer_inputs = [latent_sample]
        if self.output_layer_skip is True:
            skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.rnn_layers]
            output_layer_inputs.append(self.activation_fn(sum(skip_connections)))
        elif self.output_layer_enc_prev:
            output_layer_inputs.append(self.rnn_layers[-1][:, 0:-1])
        if self.output_layer_raw_inputs:
            output_layer_inputs.append(shifted_inputs[:, 0:-1])

        self.output_layer_inputs = tf.concat(output_layer_inputs, axis=-1)
        self.build_output_layer()

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        super(RNNLadder, self).build_loss_terms()

        # Get latent layer loss terms, apply mask and reduce function, and insert into our loss container.
        if self.is_eval:
            self.latent_layer.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood, eval_dict=self.ops_for_eval_mode)
            self.ops_evaluation["eval_dict"] = self.ops_for_eval_mode
        else:
            self.latent_layer.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood)

    def sample(self, **kwargs):
        """
        Sampling function.
        Args:
            **kwargs:
        """
        seed_sequence = kwargs.get('seed_sequence', None)
        sample_length = kwargs.get('sample_length', 100)

        assert seed_sequence is not None, "Need a seed sample."
        batch_dimension = seed_sequence.ndim == 3
        if batch_dimension is False:
            seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Feed seed sequence and update RNN state.
        if not("state" in self.ops_model_output):
            self.ops_evaluation["state"] = self.rnn_output_state

        dummy_x = np.zeros([seed_sequence.shape[0], 1, seed_sequence.shape[2]])
        model_inputs = np.concatenate([seed_sequence[:, -1:], dummy_x], axis=1)
        model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: model_inputs, self.pl_seq_length:np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]})

        # Get the last step.
        last_step = model_outputs['sample'][:, -1:, :]
        model_outputs = self.sample_function(last_step, model_outputs['state'], sample_length)

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample_function(self, current_input, previous_state, sample_length):
        assert self.is_sampling, "The model must be in sampling mode."
        # For each evaluation op, create a dummy output.
        output_dict = dict()
        # State is a problem when using multirnncell.
        # for key, op in self.ops_evaluation.items():
        #     output_dict[key] = np.zeros((current_input.shape[0], 0, op.shape[2]))
        output_dict["sample"] = current_input.copy()

        dummy_x = np.zeros([current_input.shape[0], 1, current_input.shape[2]])
        for step in range(sample_length):
            model_inputs = np.concatenate([output_dict["sample"][:, -1:], dummy_x], axis=1)

            feed_dict = dict()
            feed_dict[self.initial_states] = previous_state
            feed_dict[self.pl_inputs] = model_inputs
            feed_dict[self.pl_seq_length] = np.array([model_inputs.shape[1]]*model_inputs.shape[0])
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)
            previous_state = model_outputs['state']

            # for key, val in model_outputs.items():
            #    output_dict[key] = np.concatenate([output_dict[key], val[:, -1:]], axis=1)
            output_dict["sample"] = np.concatenate([output_dict["sample"], model_outputs["sample"][:, -1:]], axis=1)

        output_dict["sample"] = output_dict["sample"][:, -sample_length:]
        return output_dict


class RNNLatentAE(RNNAutoRegressive):
    """
    RNN based sequence auto-encoder with latent space.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(RNNLatentAE, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.decoder_use_enc_prev = self.config.get('decoder_use_enc_prev', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.encoder_cell = None  # RNN cell.
        self.encoder_initial_state = None  # Initial cell state.
        self.encoder_state = None
        self.decoder_cell = None  # RNN cell.
        self.decoder_initial_state = None  # Initial cell state.
        self.decoder_state = None

        self.encoder_rnn_config = self.config.get("encoder_rnn_layer")
        self.decoder_rnn_config = self.config.get("decoder_rnn_layer")

        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")
        self.latent_layer = LatentLayer.get(self.latent_layer_config["type"], self.latent_layer_config, mode, reuse)

    def build_network(self):
        self.build_input_layer()

        with tf.variable_scope("encoder_rnn_cell", reuse=self.reuse):
            self.encoder_cell = get_rnn_cell(reuse=self.reuse, **self.encoder_rnn_config)
            self.encoder_initial_state = self.encoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        with tf.variable_scope("decoder_rnn_cell", reuse=self.reuse):
            self.decoder_cell = get_rnn_cell(reuse=self.reuse, **self.decoder_rnn_config)
            self.decoder_initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        with tf.variable_scope("encoder", reuse=self.reuse):
            encoder, self.encoder_state = self.build_rnn_block(self.inputs_hidden, self.encoder_cell, self.encoder_initial_state)

        with tf.variable_scope("latent", reuse=self.reuse):
            if self.latent_layer_config["layer_structure"] == C.LAYER_TCN:
                num_zero_padding_end = math.ceil(self.latent_layer_config['latent_filter_size']/2) - 1
                num_zero_padding_begin = math.floor(self.latent_layer_config['latent_filter_size']/2)
                q_input = tf.pad(encoder, tf.constant([(0, 0,), (num_zero_padding_begin, num_zero_padding_end), (0, 0)]), mode='CONSTANT')
            elif self.latent_layer_config["layer_structure"] == C.LAYER_FC:
                num_future_steps = 1
                q_input = tf.concat([encoder[:, num_future_steps:], encoder[:, -num_future_steps:]], axis=1)

            latent_sample = self.latent_layer.build_latent_layer(q_input=q_input,
                                                                 p_input=encoder,
                                                                 output_ops_dict=self.ops_model_output,
                                                                 eval_ops_dict=self.ops_evaluation,
                                                                 summary_ops_dict=self.ops_scalar_summary)
        with tf.variable_scope("decoder", reuse=self.reuse):
            if self.decoder_use_enc_prev:
                decoder_input_layer = tf.concat([encoder, latent_sample], axis=-1)
            elif self.decoder_use_raw_inputs:
                decoder_input_layer = tf.concat([self.inputs_hidden, latent_sample], axis=-1)
            else:
                decoder_input_layer = latent_sample

            decoder, self.decoder_state = self.build_rnn_block(decoder_input_layer, self.decoder_cell, self.decoder_initial_state)

        self.ops_evaluation["state"] = {"encoder_state": self.encoder_state, "decoder_state": self.decoder_state}
        self.rnn_output_state = self.ops_evaluation["state"]
        self.output_layer_inputs = decoder
        self.build_output_layer()

    def build_rnn_block(self, input_layer, cell, initial_state):
        """
        Builds RNN layer by using dynamic_rnn wrapper of Tensorflow.
        """
        rnn_outputs, rnn_output_state = tf.nn.dynamic_rnn(cell,
                                                          input_layer,
                                                          sequence_length=self.pl_seq_length,
                                                          initial_state=initial_state,
                                                          dtype=tf.float32)
        return rnn_outputs, rnn_output_state

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        super(RNNLatentAE, self).build_loss_terms()

        # Get latent layer loss terms, apply mask and reduce function, and insert into our loss container.
        self.latent_layer.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood)

    def sample_function(self, current_input, previous_state, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        # For each evaluation op, create a dummy output.
        output_dict = dict()
        for key, op in self.ops_evaluation.items():
            if key is not "state":
                output_dict[key] = np.zeros((current_input.shape[0], 0, op.shape[2]))
        output_dict["sample"] = current_input.copy()

        num_samples = current_input.shape[0]
        prev_encoder_state = previous_state["encoder_state"]
        prev_decoder_state = previous_state["decoder_state"]
        for step in range(sample_length):
            model_inputs = output_dict["sample"][:, -1:, :]
            feed_dict = {self.pl_inputs     : model_inputs,
                         self.encoder_initial_state: prev_encoder_state,
                         self.decoder_initial_state: prev_decoder_state,
                         self.pl_seq_length : np.ones(num_samples)}
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)

            prev_encoder_state = model_outputs["state"]["encoder_state"]
            prev_decoder_state = model_outputs["state"]["decoder_state"]
            for key in output_dict.keys():
                output_dict[key] = np.concatenate([output_dict[key], model_outputs[key]], axis=1)

        output_dict["sample"] = output_dict["sample"][:, -sample_length:]
        return output_dict


class BiRNN(BaseRNN):
    """
    Bi-directional RNN.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(BiRNN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.cells_fw = []
        self.cells_bw = []

        self.initial_states_fw = []
        self.initial_states_bw = []

        self.output_state_fw = None
        self.output_state_bw = None

        # See https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
        self.stack_fw_bw_cells = self.cell_config.get('stack_fw_bw_cells', True)

    def build_cell(self):
        """
        Builds a Tensorflow RNN cell object by using the given configuration `self.cell_config`.
        """
        if self.stack_fw_bw_cells:
            single_cell_config = self.cell_config.copy()
            single_cell_config['num_layers'] = 1
            for i in range(self.cell_config['num_layers']):
                cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **single_cell_config)
                self.cells_fw.append(cell_fw)
                self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

                cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **single_cell_config)
                self.cells_bw.append(cell_bw)
                self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))
        else:
            cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **self.cell_config)
            self.cells_fw.append(cell_fw)
            self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

            cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **self.cell_config)
            self.cells_bw.append(cell_bw)
            self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

    def build_rnn_layer(self):
        with tf.variable_scope("bidirectional_rnn_layer", reuse=self.reuse):
            if self.stack_fw_bw_cells:
                self.rnn_outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                            cells_fw=self.cells_fw,
                                                                            cells_bw=self.cells_bw,
                                                                            inputs=self.inputs_hidden,
                                                                            initial_states_fw=self.initial_states_fw,
                                                                            initial_states_bw=self.initial_states_bw,
                                                                            dtype=tf.float32,
                                                                            sequence_length=self.pl_seq_length)
            else:
                outputs_tuple, output_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cells_fw[0],
                    cell_bw=self.cells_bw[0],
                    inputs=self.inputs_hidden,
                    sequence_length=self.pl_seq_length,
                    initial_state_fw=self.initial_states_fw[0],
                    initial_state_bw=self.initial_states_bw[0],
                    dtype=tf.float32)

                self.rnn_outputs = tf.concat(outputs_tuple, 2)
                self.output_state_fw, self.output_state_bw = output_states

            self.output_layer_inputs = self.rnn_outputs
            self.ops_evaluation["state"] = [self.output_state_fw, self.output_state_bw]

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get("input_sequence", None)
        target_sequence = kwargs.get("target_sequence", None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = dict()
        feed_dict[self.pl_inputs] = input_sequence
        feed_dict[self.pl_seq_length] = np.array([input_sequence.shape[1]]*input_sequence.shape[0])

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation["loss"] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs["loss"])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def reconstruct_chunks(self, **kwargs):
        """
        """
        input_sequence = kwargs.get("input_sequence", None)
        target_sequence = kwargs.get("target_sequence", None)
        len_past = kwargs.get("len_past", None)
        len_future = kwargs.get("len_future", None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation["loss"] = self.ops_loss

        batch_size, seq_len, input_size = input_sequence.shape

        predictions = []
        loss = 0.0
        feed_dict = {}

        for step in range(seq_len):
            start_idx = max(step-len_past, 0)
            end_idx = min(step+len_future+1, seq_len)

            feed_dict[self.pl_inputs] = input_sequence[:, start_idx:end_idx]
            feed_dict[self.pl_seq_length] = np.array([end_idx-start_idx]*batch_size)
            if target_sequence is not None:
                feed_dict[self.pl_targets] = target_sequence[:, start_idx:end_idx]

            model_outputs = self.session.run(self.ops_evaluation, feed_dict)
            prediction_step = min(step, len_past)
            predictions.append(model_outputs["sample"][:, prediction_step:prediction_step+1])
            if "loss" in model_outputs:
                loss += model_outputs["loss"]["total_loss"]*batch_size

        model_outputs = dict()
        model_outputs["sample"] = np.concatenate(predictions, axis=1)
        model_outputs["loss"] = {"total_loss":loss/(batch_size*seq_len)}
        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs


class ZForce(BaseTemporalModel):
    """
    Z-forcing Network.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(ZForce, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.input_layer_config = config.get("input_layer")
        self.f_cell_config = config.get("f_cell")
        self.b_cell_config = config.get("b_cell")
        self.z_cell_config = copy.deepcopy(config.get("z_cell"))
        self.z_cell_config["output_layer"] = copy.deepcopy(self.output_layer_config)

        self.use_x_backward_pred = config.get("use_x_backward_pred", False)
        self.use_backward_z_pred = config.get("use_backward_z_pred", False)
        self.use_z_force = config.get("use_z_force", True)
        self.anneal_aux_weights = config.get("anneal_aux_weights", False)

        # Raw model inputs in reverse order.
        with tf.variable_scope("reverted_input", reuse=self.reuse):
            self.pl_inputs_reverse = tf.reverse_sequence(self.pl_inputs, self.pl_seq_length, seq_dim=1, batch_dim=0)

        # Forward RNN cell and its outputs.
        self.f_cell = None
        self.f_outputs = None
        # Backward RNN cell, its initial state and final states and output.
        self.b_cell = None
        self.b_cell_initial_states = None
        self.b_state = None
        self.b_outputs = None
        # Zforcing cell, its initial and final state.
        self.z_cell = None
        self.z_cell_initial_states = None
        self.z_state = None
        # List of RNN layer outputs. It is accessed by other methods.
        self.rnn_outputs = None
        # List of RNN layer state. It is accessed by other methods.
        self.rnn_output_state = None

        self.aux_x_weight = 1.0
        self.aux_b_weight = 1.0
        self.kld_weight = 1.0
        self.global_step = None

    def build_graph(self):
        if self.anneal_aux_weights and self.global_step is not None:
            max_steps = 10000
            num_steps = 10
            values = np.linspace(0.1, 1.0, num_steps + 1).tolist()
            boundaries = np.int32(np.linspace(0, max_steps, num_steps)).tolist()

            self.aux_x_weight = tf.train.piecewise_constant(self.global_step, boundaries, values)
            self.ops_scalar_summary["aux_x_weight"] = self.aux_x_weight
            self.aux_b_weight = tf.train.piecewise_constant(self.global_step, boundaries, values)
            self.ops_scalar_summary["aux_b_weight"] = self.aux_b_weight

        self.build_network()
        self.build_loss_terms()
        self.build_total_loss()
        self.build_summary_plots()
        self.finalise_graph()
        if self.reuse is False:
            self.log_num_parameters()

    def build_network(self):
        self.build_input_layer()
        self.build_cell()
        self.build_backward_layer()
        self.build_z_forward_layer()
        self.build_output_layer()

    def build_input_layer(self):
        """
        Builds a number fully connected layers projecting the inputs into an intermediate representation  space.
        """
        if self.input_layer_config is not None:
            with tf.variable_scope('input_layer', reuse=self.reuse):
                if self.input_layer_config.get("dropout_rate", 0) > 0:
                    self.inputs_hidden = tf.layers.dropout(self.pl_inputs,
                                                           rate=self.input_layer_config.get("dropout_rate"),
                                                           noise_shape=None,
                                                           seed=17,
                                                           training=self.is_training)
                else:
                    self.inputs_hidden = self.pl_inputs

                if self.input_layer_config.get("num_layers", 0) > 0:
                    flat_inputs_hidden = self.flat_tensor(self.inputs_hidden)
                    flat_inputs_hidden = fully_connected_layer(flat_inputs_hidden, **self.input_layer_config)
                    self.inputs_hidden = self.temporal_tensor(flat_inputs_hidden)
        else:
            self.inputs_hidden = self.pl_inputs

    def build_cell(self):
            self.f_cell = get_rnn_cell(scope='f_cell', reuse=self.reuse, **self.f_cell_config)
            self.b_cell = get_rnn_cell(scope='b_cell', reuse=self.reuse, **self.b_cell_config)
            self.z_cell = ZForcingCell(forward_cell=self.f_cell, reuse=self.reuse, mode=self.mode, **self.z_cell_config)

            self.z_cell_initial_states = self.f_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.b_cell_initial_states = self.b_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_backward_layer(self):
        with tf.variable_scope("backward_rnn", reuse=self.reuse):
            b_outputs_reverse, self.b_state = tf.nn.dynamic_rnn(self.b_cell,
                                                                self.pl_inputs_reverse,
                                                                sequence_length=self.pl_seq_length,
                                                                initial_state=self.b_cell_initial_states,
                                                                dtype=tf.float32)
            # TODO Do we need to revert backward state as well?
            self.b_outputs = tf.reverse_sequence(b_outputs_reverse, self.pl_seq_length, seq_dim=1, batch_dim=0)

    def build_z_forward_layer(self):
        with tf.variable_scope("z_forward_rnn", reuse=self.reuse):
            if self.is_sampling:
                forward_input = self.inputs_hidden
            else:
                forward_input = tf.concat([self.inputs_hidden, self.b_outputs], axis=-1)
            self.f_outputs, self.z_state = tf.nn.dynamic_rnn(self.z_cell,
                                                             forward_input,
                                                             sequence_length=self.pl_seq_length,
                                                             initial_state=self.z_cell_initial_states,
                                                             dtype=tf.float32)
            self.rnn_outputs = self.f_outputs
            self.rnn_output_state = self.z_state

    def build_output_layer(self):
        # These are the predefined cell outputs.
        model_out_keys = [C.Q_MU, C.Q_SIGMA, C.P_MU, C.P_SIGMA, C.Z_LATENT]
        model_out_keys.extend(self.output_layer_config['out_keys'])
        # Assign model outputs.
        for out_key, out_op in zip(model_out_keys, self.rnn_outputs):
            self.ops_model_output[out_key] = out_op

        if self.is_training:
            if self.use_x_backward_pred:
                flat_outputs_hidden = self.flat_tensor(self.b_outputs)
                with tf.variable_scope('out_x_backward_pred', reuse=self.reuse):
                    flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

                    out_key = "x_backward_pred" + C.SUF_MU
                    with tf.variable_scope(out_key, reuse=self.reuse):
                        flat_out = linear(input_layer=flat_outputs_hidden,
                                          output_size=self.target_dims[0],
                                          activation_fn=None,
                                          is_training=self.is_training)
                        self.ops_model_output[out_key] = self.temporal_tensor(flat_out)

                    out_key = "x_backward_pred" + C.SUF_SIGMA
                    with tf.variable_scope(out_key, reuse=self.reuse):
                        flat_out = linear(input_layer=flat_outputs_hidden,
                                          output_size=self.target_dims[0],
                                          activation_fn=C.SOFTPLUS,
                                          is_training=self.is_training)
                        self.ops_model_output[out_key] = self.temporal_tensor(flat_out)

            if self.use_backward_z_pred:
                if self.use_z_force:
                    inputs = self.ops_model_output[C.Z_LATENT]
                else:
                    inputs = tf.concat([self.z_state, self.ops_model_output[C.Z_LATENT]], axis=-1)

                flat_outputs_hidden = self.flat_tensor(inputs)
                with tf.variable_scope('out_backward_z_pred', reuse=self.reuse):
                    flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

                    out_key = "backward_z_pred" + C.SUF_MU
                    with tf.variable_scope(out_key, reuse=self.reuse):
                        flat_out = linear(input_layer=flat_outputs_hidden,
                                          output_size=self.config.get("backward_h_size"),
                                          activation_fn=C.TANH,
                                          is_training=self.is_training)
                        self.ops_model_output[out_key] = self.temporal_tensor(flat_out)

                    out_key = "backward_z_pred" + C.SUF_SIGMA
                    with tf.variable_scope(out_key, reuse=self.reuse):
                        flat_out = linear(input_layer=flat_outputs_hidden,
                                          output_size=self.config.get("backward_h_size"),
                                          activation_fn=C.SOFTPLUS,
                                          is_training=self.is_training)
                        self.ops_model_output[out_key] = self.temporal_tensor(flat_out)

        self.ops_evaluation[C.P_MU] = self.ops_model_output[C.P_MU]
        self.ops_evaluation[C.P_SIGMA] = self.ops_model_output[C.P_SIGMA]
        self.ops_evaluation[C.Q_MU] = self.ops_model_output[C.Q_MU]
        self.ops_evaluation[C.Q_SIGMA] = self.ops_model_output[C.Q_SIGMA]
        self.ops_evaluation['state'] = self.rnn_output_state

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.P_MU])[-1], tf.float32)
        self.ops_scalar_summary["mean_p_sigma"] = tf.reduce_sum(self.ops_model_output[C.P_SIGMA]*self.seq_loss_mask) / num_entries
        self.ops_scalar_summary["mean_q_sigma"] = tf.reduce_sum(self.ops_model_output[C.Q_SIGMA]*self.seq_loss_mask) / num_entries
        self.ops_scalar_summary["mean_q_mu"] = tf.reduce_sum(self.ops_model_output[C.Q_MU]*self.seq_loss_mask)/num_entries
        self.ops_scalar_summary["mean_p_mu"] = tf.reduce_sum(self.ops_model_output[C.P_MU]*self.seq_loss_mask)/num_entries

    def build_loss_terms(self):
        super(ZForce, self).build_loss_terms()

        loss_key = 'loss_kld'
        if loss_key not in self.ops_loss:
            with tf.name_scope('kld_loss'):
                # KL-Divergence.
                self.ops_loss['loss_kld'] = self.kld_weight * self.reduce_loss_fn(
                    self.seq_loss_mask*tf_loss.kld_normal_isotropic(self.ops_model_output[C.Q_MU],
                                                                    self.ops_model_output[C.Q_SIGMA],
                                                                    self.ops_model_output[C.P_MU],
                                                                    self.ops_model_output[C.P_SIGMA], reduce_sum=False))
        if self.is_training:
            if self.use_x_backward_pred:
                logli_term = tf_loss.logli_normal_isotropic(self.target_pieces[0],
                                                            self.ops_model_output["x_backward_pred" + C.SUF_MU],
                                                            self.ops_model_output["x_backward_pred" + C.SUF_SIGMA],
                                                            reduce_sum=False)
                self.ops_loss["aux_x"] = -self.aux_x_weight * self.reduce_loss_fn(self.seq_loss_mask*logli_term)

            if self.use_backward_z_pred:
                logli_term = tf_loss.logli_normal_isotropic(tf.stop_gradient(self.b_outputs),
                                                            self.ops_model_output["backward_z_pred" + C.SUF_MU],
                                                            self.ops_model_output["backward_z_pred" + C.SUF_SIGMA],
                                                            reduce_sum=False)
                self.ops_loss["aux_b"] = -self.aux_b_weight * self.reduce_loss_fn(self.seq_loss_mask*logli_term)

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get("input_sequence", None)
        target_sequence = kwargs.get("target_sequence", None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = dict()
        feed_dict[self.pl_inputs] = input_sequence
        feed_dict[self.pl_seq_length] = np.array([input_sequence.shape[1]]*input_sequence.shape[0])

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation["loss"] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs["loss"])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs


class RNNLatentCellModel(RNNAutoRegressive):
    """
    Variational RNN model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(RNNLatentCellModel, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, )

        self.decoder_use_enc_prev = self.config.get('decoder_use_enc_prev', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.cell_config = self.config.get("latent_layer")

    def build_cell(self):
        self.cell = LatentCell.get(self.cell_config["type"], self.cell_config, self.mode, self.reuse)
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_output_layer(self):
        self.cell.register_sequence_components(self.rnn_outputs, self.ops_model_output, self.ops_evaluation, self.ops_scalar_summary)
        self.ops_evaluation['state'] = self.rnn_output_state

        decoder_inputs = [self.rnn_outputs[-1]]  # Latent sample.
        if self.decoder_use_raw_inputs:
            decoder_inputs.append(self.pl_inputs)
        elif self.decoder_use_enc_prev:
            decoder_inputs.append(self.rnn_outputs[-2])

        self.output_layer_inputs = tf.concat(decoder_inputs, axis=-1)
        super(RNNLatentCellModel, self).build_output_layer()

    def build_loss_terms(self):
        super(RNNLatentCellModel, self).build_loss_terms()
        # Get latent cell loss terms, apply mask and reduce function, and insert into our loss container.
        self.cell.build_loss(self.seq_loss_mask, self.reduce_loss_fn, self.ops_loss, reward=self.likelihood)

    def sample_function(self, current_input, previous_state, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        # For each evaluation op, create a dummy output.
        output_dict = dict()
        for key, op in self.ops_evaluation.items():
            if key is not "state":
                output_dict[key] = np.zeros((current_input.shape[0], 0, op.shape[2]))
        output_dict["sample"] = current_input.copy()

        num_samples = current_input.shape[0]
        current_state = previous_state
        for step in range(sample_length):
            model_inputs = output_dict["sample"][:, -1:, :]
            feed_dict = {self.pl_inputs     : model_inputs,
                         self.initial_states: current_state,
                         self.pl_seq_length : np.ones(num_samples)}
            model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed_dict)
            current_state = model_outputs["state"]
            for key in output_dict.keys():
                output_dict[key] = np.concatenate([output_dict[key], model_outputs[key]], axis=1)

        output_dict["sample"] = output_dict["sample"][:, -sample_length:]
        return output_dict


class VRNN(BaseRNN):
    """
    Variational RNN model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs):
        super(VRNN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, **kwargs)

        self.latent_size = self.config.get('latent_size')

        self.vrnn_cell_constructor = getattr(sys.modules[__name__], self.config.get('vrnn_cell_cls'))
        # TODO: Create a dictionary just for cell arguments.
        self.vrnn_cell_args = copy.deepcopy(config.config)
        self.vrnn_cell_args['input_dims'] = self.input_dims
        self.vrnn_cell_args['output_layer'] = self.output_layer_config

        kld_weight = self.config.get('kld_weight', 0.5)
        if isinstance(kld_weight, dict) and self.global_step:
            self.kld_weight = get_decay_variable(global_step=self.global_step, config=kld_weight, name="kld_weight")
        else:
            self.kld_weight = kld_weight
        if not self.is_training:
            self.kld_weight = 1.0

    def build_cell(self):
        self.cell = self.vrnn_cell_constructor(reuse=self.reuse, mode=self.mode, config=self.vrnn_cell_args, sample_fn=self.sample_fn_tf)

        assert isinstance(self.cell, VRNNCell), "Cell object must be an instance of VRNNCell for VRNN model."
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_output_layer(self):
        # These are the predefined vrnn cell outputs.
        vrnn_model_out_keys = [C.Q_MU, C.Q_SIGMA, C.P_MU, C.P_SIGMA]
        vrnn_model_out_keys.extend(self.output_layer_config['out_keys'])

        # Assign model outputs.
        for out_key, out_op in zip(vrnn_model_out_keys, self.rnn_outputs):
            self.ops_model_output[out_key] = out_op

        self.ops_evaluation[C.P_MU] = self.ops_model_output[C.P_MU]
        self.ops_evaluation[C.P_SIGMA] = self.ops_model_output[C.P_SIGMA]
        self.ops_evaluation[C.Q_MU] = self.ops_model_output[C.Q_MU]
        self.ops_evaluation[C.Q_SIGMA] = self.ops_model_output[C.Q_SIGMA]
        self.ops_evaluation['state'] = self.rnn_output_state

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.OUT_MU])[-1], tf.float32)
        if C.OUT_MU in self.ops_model_output:
            self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU]*self.seq_loss_mask)/num_entries
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(self.ops_model_output[C.OUT_SIGMA]*self.seq_loss_mask) / num_entries

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

        num_entries = tf.cast(tf.reduce_sum(self.seq_loss_mask), tf.float32)*tf.cast(tf.shape(self.ops_model_output[C.P_MU])[-1], tf.float32)
        self.ops_scalar_summary["mean_p_sigma"] = tf.reduce_sum(self.ops_model_output[C.P_SIGMA]*self.seq_loss_mask) / num_entries
        self.ops_scalar_summary["mean_q_sigma"] = tf.reduce_sum(self.ops_model_output[C.Q_SIGMA]*self.seq_loss_mask) / num_entries
        self.ops_scalar_summary["mean_q_mu"] = tf.reduce_sum(self.ops_model_output[C.Q_MU]*self.seq_loss_mask)/num_entries
        self.ops_scalar_summary["mean_p_mu"] = tf.reduce_sum(self.ops_model_output[C.P_MU]*self.seq_loss_mask)/num_entries

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        super(VRNN, self).build_loss_terms()

        loss_key = 'loss_kld'
        if loss_key not in self.ops_loss:
            with tf.name_scope('kld_loss'):
                # KL-Divergence.
                self.ops_loss['loss_kld'] = self.kld_weight*self.reduce_loss_fn(
                    self.seq_loss_mask*tf_loss.kld_normal_isotropic(self.ops_model_output[C.Q_MU],
                                                                    self.ops_model_output[C.Q_SIGMA],
                                                                    self.ops_model_output[C.P_MU],
                                                                    self.ops_model_output[C.P_SIGMA], reduce_sum=False))

    def build_summary_plots(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch
        and then report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create
        placeholders per loss to pass the average loss.
        """
        super(VRNN, self).build_summary_plots()

        # Create summaries to visualize distribution of latent variables.
        if self.config.get('tensorboard_verbose', 0) > 1:
            set_of_graph_nodes = [C.Q_MU, C.Q_SIGMA, C.P_MU, C.P_SIGMA, C.OUT_MU, C.OUT_SIGMA]
            for out_key in set_of_graph_nodes:
                tf.summary.histogram(out_key, self.ops_model_output[out_key], collections=[self.mode+'_summary_plot', self.mode+'_stochastic_variables'])

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get('input_sequence', None)
        target_sequence = kwargs.get('target_sequence', None)

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        if not("state" in self.ops_evaluation):
            self.ops_evaluation["state"] = self.rnn_output_state

        sample_length = input_sequence.shape[1]
        feed_dict = {self.pl_inputs: input_sequence, self.pl_seq_length:np.ones(1)*sample_length}

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation['loss'] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs['loss'])

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample(self, **kwargs):
        """
        Sampling function. Since model has different graphs for sampling and evaluation modes, a seed state must be
        given in order to predict future steps. Otherwise, a sample will be synthesized randomly.

        Args:
            **kwargs:
        """
        assert self.is_sampling, "The model must be in sampling mode."

        seed_state = kwargs.get('seed_state', None)
        sample_length = kwargs.get('sample_length', 100)
        batch_dimension = False

        if not("state" in self.ops_evaluation):
            self.ops_evaluation["state"] = self.rnn_output_state

        dummy_x = np.zeros((1, sample_length, sum(self.input_dims)))

        # Feed seed sequence and update RNN state.
        feed = {self.pl_inputs    : dummy_x,
                self.pl_seq_length: np.ones(1)*sample_length}
        if seed_state is not None:
            feed[self.initial_states] = seed_state

        model_outputs = self.session.run(self.ops_evaluation, feed_dict=feed)

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs
