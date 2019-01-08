import tensorflow as tf

from tf_model_utils import linear, get_activation_fn, get_rnn_cell, fully_connected_layer
import tf_loss
from constants import Constants
C = Constants()


class LatentCell(tf.contrib.rnn.RNNCell):
    """
    Base class for latent cells.

    Instances of this cell is used by rnn wrappers (e.g., dynamic_rnn, static_rnn). Hence, for a given input step t, all
    calculations are made for this single step. The latent components at step t are returned as part of the output.

    The rnn wrapper accumulates outputs of all steps. In order to access latent components of the whole sequence
    register_sequence_components method should be called first. Then the cell can operate on the entire sequence (i.e.,
    implementing latent loss terms, etc.)
    """
    def __init__(self, config, mode, reuse):
        super(LatentCell, self).__init__()

        self.config = config
        self.reuse = reuse
        assert mode in ["training", "validation", "test", "sampling"]
        self.mode = mode
        self.is_sampling = mode == "sampling"
        self.is_validation = mode == "validation" or mode == "test"
        self.is_training = mode == "training"
        self.layer_structure = config.get("layer_structure")

        self.ops_loss = dict()
        self.state_size_ = None
        self.output_size_ = None

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_size_)

    def register_sequence_components(self, cell_ops_list, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Fetches rnn wrapper outputs. Inserts latent ops into main model's containers. See BaseTemporalModel for details.
        Args:
            cell_ops_list: rnn wrapper (e.g., dynamic_rnn, static_rnn) outputs.
            output_ops_dict:
            eval_ops_dict:
            summary_ops_dict:
        Returns:
        """
        raise NotImplementedError('subclasses must override sample method')

    def build_latent_cell(self, q_input, p_input, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
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

    def get_cell_state(self, state):
        """
        Fetches hidden state of the internal cell and latent sample.
        Args:
            state: state output of GaussianLatentCell, i.e., output of __call__.
        Returns:
            state of the internal cell, state of the internal cell, latent sample
        """
        internal_cell_state = state[0]
        latent_sample = state[-1]  # Latent sample is always the last element.
        # If the internal cell is stacking multiple rnn cells, then the output is the hidden state of the top most cell.
        if self.internal_cell_type == C.GRU:
            state_h = internal_cell_state[-1] if type(internal_cell_state) == tuple else internal_cell_state
        else:
            state_h = internal_cell_state[-1].h if type(internal_cell_state) == tuple else internal_cell_state.h

        return internal_cell_state, state_h, latent_sample

    def phi(self, input_, scope=None, reuse=None):
        """
        A fully connected layer to increase model capacity and learn and intermediate representation. It is reported to
        be useful in https://arxiv.org/pdf/1506.02216.pdf
        Args:
            input_:
            scope:
            reuse:
        Returns:
        """
        if scope is not None:
            with tf.variable_scope(scope, reuse=reuse):
                return fully_connected_layer(input_layer=input_,
                                             is_training=self.is_training,
                                             activation_fn=self.config["hidden_activation_fn"],
                                             num_layers=self.config["num_hidden_layers"],
                                             size=self.config["num_hidden_units"])
        else:
            return fully_connected_layer(input_layer=input_,
                                         is_training=self.is_training,
                                         activation_fn=self.config["hidden_activation_fn"],
                                         num_layers=self.config["num_hidden_layers"],
                                         size=self.config["num_hidden_units"])

    @staticmethod
    def get(layer_type, config, mode, reuse):
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
            return GaussianLatentCell(config, mode, reuse)
        elif layer_type == C.LATENT_VARIATIONAL_CODEBOOK:
            return VariationalCodebookCell(config, mode, reuse)
        elif layer_type == C.LATENT_STOCHASTIC_CODEBOOK:
            return StochasticCodebookCell(config, mode, reuse)
        else:
            raise Exception("Unknown latent cell.")


class GaussianLatentCell(LatentCell):
    """
    VAE latent space for time-series data, modeled by a Gaussian distribution with diagonal covariance matrix.
    """
    def __init__(self, config, mode, reuse):
        super(GaussianLatentCell, self).__init__(config, mode, reuse)

        self.use_temporal_kld = self.config.get('use_temporal_kld', False)
        self.tkld_weight = self.config.get('tkld_weight', 0.1)
        self.kld_weight = self.config.get('kld_weight', 0.1)
        if not self.is_training:
            self.kld_weight = 1.0

        self.latent_size = self.config.get('latent_size')
        self.latent_phi_size = self.config["num_hidden_units"] if self.config["num_hidden_units"] > 0 and self.config["num_hidden_layers"] > 0 else self.config.get('latent_size')
        self.latent_sigma_threshold = self.config.get('latent_sigma_threshold', 0)

        # Latent space components in single step level.
        self.p_mu_t = None
        self.q_mu_t = None
        self.p_sigma_t = None
        self.q_sigma_t = None
        self.z_t = None  # Latent sample.
        self.z_t_phi = None  # An intermediate representation of the latent sample z_t.

        # This cell is used by dynamic_rnn which accumulates states and outputs at every step.
        # dynamic_rnn outputs should be fed back to the cell in order to create loss internally.
        # Latent space components in sequence level.
        self.p_mu = None
        self.q_mu = None
        self.p_sigma = None
        self.q_sigma = None

        self.internal_cell_config = dict(cell_type=config['cell_type'],
                                         num_layers=config['cell_num_layers'],
                                         size=config['cell_size'])
        self.internal_cell_type = config['cell_type']
        self.internal_cell = get_rnn_cell(scope='internal_cell', **self.internal_cell_config)
        self.internal_cell_state = None
        self.internal_cell_output = None  # Cell output
        self.state_size_ = [self.internal_cell.state_size, self.latent_phi_size]
        # q_mu, q_sigma, p_mu, p_sigma, internal_cell_output, latent_sample
        self.output_size_ = [self.latent_size]*4 + [config['cell_size'], self.latent_phi_size]

    @staticmethod
    def reparameterization(mu_t, sigma_t):
        """
        Given an isotropic normal distribution (mu and sigma), draws a sample by using reparameterization trick:
        z = mu + sigma*epsilon
        Args:
            mu_t: mean of Gaussian distribution with diagonal covariance matrix.
            sigma_t: standard deviation of Gaussian distribution with a diagonal covariance matrix.
        Returns:
        """
        with tf.variable_scope('z'):
            eps = tf.random_normal(tf.shape(sigma_t), 0.0, 1.0, dtype=tf.float32)
            z_t = tf.add(mu_t, tf.multiply(sigma_t, eps))
            return z_t

    def __call__(self, inputs, state, scope=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            cell_state_prev, cell_output_prev, z_sample_prev = self.get_cell_state(state)

            if self.is_sampling:
                # The input is prediction from the previous step.
                # First update the internal cell by using cell state and latent sample from previous steps.
                input_cell = tf.concat((inputs, z_sample_prev), axis=1)
                self.internal_cell_output, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                # Draw a latent sample from prior distribution.
                self.p_mu_t, self.p_sigma_t = self.build_latent_dist(self.internal_cell_output, C.LATENT_P, self.reuse)
                self.z_t = GaussianLatentCell.reparameterization(self.p_mu_t, self.p_sigma_t)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)
                # Although we don't need q_mu_t, q_sigma_t in the sampling mode, __call__ method still requires them.
                # Make an arbitrary assignment.
                self.q_mu_t, self.q_sigma_t = self.p_mu_t, self.p_sigma_t

            elif self.is_training or self.is_validation:
                # First draw a latent sample from approx. posterior distribution.
                input_z = tf.concat((inputs, cell_output_prev), axis=1)
                self.q_mu_t, self.q_sigma_t = self.build_latent_dist(input_z, C.LATENT_Q, self.reuse)
                self.z_t = GaussianLatentCell.reparameterization(self.q_mu_t, self.q_sigma_t)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)

                # Estimate prior.
                self.p_mu_t, self.p_sigma_t = self.build_latent_dist(cell_output_prev, C.LATENT_P, self.reuse)

                # Update the internal cell by using cell state from the previous step and latent sample.
                input_cell = tf.concat((inputs, self.z_t_phi), axis=1)
                _, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                self.internal_cell_output = cell_output_prev

            # Prepare cell output.
            cell_output = [self.q_mu_t, self.q_sigma_t, self.p_mu_t, self.p_sigma_t, self.internal_cell_output, self.z_t_phi]
            # Prepare cell state.
            cell_state = [self.internal_cell_state, self.z_t_phi]
            return tuple(cell_output), tuple(cell_state)

    def build_latent_dist(self, input_, scope, reuse):
        """
        Given the input parametrizes a Normal distribution.
        Args:
            input_:
            scope: "approximate_posterior" or "prior".
            reuse:
        Returns:
            mu and sigma tensors.
        """
        name = 'q' if scope == C.LATENT_Q else 'p_dists'
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope(name+'_mu', reuse=reuse):
                mu_t = linear(input_layer=self.phi(input_),
                              output_size=self.config['latent_size'],
                              activation_fn=None,
                              is_training=self.is_training)
            with tf.variable_scope(name+'_sigma', reuse=reuse):
                sigma_t = linear(input_layer=self.phi(input_),
                                 output_size=self.config['latent_size'],
                                 activation_fn=tf.nn.softplus,
                                 is_training=self.is_training)
                if self.latent_sigma_threshold > 0:
                    sigma_t = tf.clip_by_value(sigma_t, -self.latent_sigma_threshold, self.latent_sigma_threshold)
        return mu_t, sigma_t

    def register_sequence_components(self, cell_ops_list, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Gets dynamic_rnn outputs and fetches latent cell components.
        Args:
            cell_ops_list: list containing the approximate posterior and prior component ops.
            output_ops_dict:
            eval_ops_dict:
            summary_ops_dict:
        Returns:
        """
        # Follows the order in __call__ method.
        self.q_mu = cell_ops_list[0]
        self.q_sigma = cell_ops_list[1]
        self.p_mu = cell_ops_list[2]
        self.p_sigma = cell_ops_list[3]

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

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict, **kwargs):
        """
        Creates KL-divergence loss between prior and approximate posterior distributions. If use_temporal_kld is True,
        then creates another KL-divergence term between consecutive approximate posteriors in time.
        """
        assert self.p_mu is not None, "Sequence components must be registered first."

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


class VariationalCodebookCell(LatentCell):
    """
    Latent space with a codebook of embedding vectors where the encoder maps the inputs to the most relevant
    representation. It is unsupervised. Hence, both the encoder and embedding vectors are learned.
    """
    def __init__(self, config, mode, reuse):
        super(VariationalCodebookCell, self).__init__(config, mode, reuse)

        self.latent_num_components = config.get('latent_num_components')
        self.latent_size_components = config.get('latent_size')
        self.latent_phi_size = self.config["num_hidden_units"] if self.config["num_hidden_units"] > 0 and self.config["num_hidden_layers"] > 0 else self.config.get('latent_size')
        self.latent_divisive_normalization = config.get('latent_divisive_normalization', False)
        self.use_temporal_kld = config.get('use_temporal_kld', False)
        self.use_reinforce = config.get('use_reinforce', False)
        self.tkld_weight = config.get('tkld_weight', 0.1)
        self.kld_weight = config.get('kld_weight', 0.5)
        if not self.is_training:
            self.kld_weight = 1.0
        self.loss_diversity_weight = config.get('loss_diversity_weight', 1)
        self.loss_diversity_batch_weight = config.get('loss_diversity_batch_weight', 1)
        self.batch_size = None

        # Parameters of codebook.
        with tf.variable_scope("latent_codebook", reuse=self.reuse):
            self.codebook_mu = tf.get_variable(name="codebook_mu", dtype=tf.float32, initializer=tf.random_uniform([self.latent_num_components, self.latent_size_components], -1.0, 1.0))

        # Logits of approximate posterior and prior categorical distributions in single step level.
        self.q_pi_t = None
        self.p_pi_t = None
        self.pi_sample_t = None  # Indices of selected components.

        # This cell is used by dynamic_rnn which accumulates states and outputs at every step.
        # dynamic_rnn outputs should be fed back to the cell in order to create loss internally.
        # Latent space components in sequence level.
        self.q_pi = None
        self.p_pi = None
        self.pi_sample = None  # Indices of selected components.
        self.q_pi_probs = None
        self.p_pi_probs = None

        self.internal_cell_config = dict(cell_type=config['cell_type'],
                                         num_layers=config['cell_num_layers'],
                                         size=config['cell_size'])
        self.internal_cell_type = config['cell_type']
        self.internal_cell = get_rnn_cell(scope='internal_cell', **self.internal_cell_config)
        self.internal_cell_state = None
        self.internal_cell_output = None  # Cell output
        self.state_size_ = [self.internal_cell.state_size, self.latent_phi_size]
        # q_pi, p_pi, pi_sample, internal_cell_output, latent_sample
        self.output_size_ = [self.latent_num_components]*2 + [1, config['cell_size'], self.latent_phi_size]

    @staticmethod
    @tf.custom_gradient
    def draw_deterministic_latent_sample(pi, codebook):
        """
        Draws a latent sample and implements a custom gradient for non-differentiable argmax operation.
        Args:
            pi: logits or probability vector with shape of (batch_size, num_components)
            codebook: codebook.
        Returns:
            A latent representation indexed by pi.
        """
        num_components = tf.shape(codebook)[0]
        code_idx = tf.expand_dims(tf.argmax(pi, axis=-1), axis=-1)
        z_sample = tf.gather_nd(codebook, code_idx)

        def grad(z_grad):
            """
            Calculates a custom gradient for the argmax operator, and gradients for latent representations through the
            reparameterization trick.
            """
            reduced_grad = tf.reduce_mean(z_grad, axis=-1, keepdims=True)
            pi_grad = reduced_grad*tf.one_hot(tf.argmax(pi, axis=-1), depth=num_components, axis=-1)

            codebook_grad = tf.IndexedSlices(values=z_grad, indices=code_idx[:,0])
            return pi_grad, codebook_grad

        return z_sample, grad

    @staticmethod
    def draw_deterministic_latent_sample_reinforce(pi, codebook):
        """
        Draws a latent sample operated by the reinforce algorithm.
        Args:
            pi: categorical distribution.
            codebook: codebook
        Returns:
            A latent representation indexed by pi.
        """
        # dist_pi = tf.contrib.distributions.Categorical(logits=pi, name="dist_pi")
        # code_idx = tf.expand_dims(dist_pi.sample(), axis=-1)
        # code_idx = tf.expand_dims(tf.argmax(pi, axis=-1), axis=-1)
        code_idx = tf.multinomial(logits=pi, num_samples=1, name=None, output_dtype=tf.int32)
        z_sample = tf.gather_nd(codebook, code_idx)
        return z_sample, code_idx

    def __call__(self, inputs, state, scope=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            cell_state_prev, cell_output_prev, z_sample_prev = self.get_cell_state(state)

            if self.is_sampling:
                # The input is prediction from the previous step.
                # First update the internal cell by using cell state and latent sample from previous steps.
                input_cell = tf.concat((inputs, z_sample_prev), axis=1)
                self.internal_cell_output, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                # Draw a latent sample from prior distribution.
                self.p_pi_t = self.build_latent_dist(self.internal_cell_output, C.LATENT_P, self.reuse)
                if self.use_reinforce:
                    self.z_t, self.pi_sample_t = VariationalCodebookCell.draw_deterministic_latent_sample_reinforce(self.p_pi_t, self.codebook_mu)
                else:
                    self.z_t = VariationalCodebookCell.draw_deterministic_latent_sample(self.p_pi_t, self.codebook_mu)
                    self.pi_sample_t = tf.expand_dims(tf.argmax(self.p_pi_t, axis=-1), axis=-1)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)

                # Although we don't need q_mu_t, q_sigma_t in the sampling mode, __call__ method still requires them.
                # Make an arbitrary assignment.
                self.q_pi_t = self.p_pi_t

            elif self.is_training or self.is_validation:
                # First draw a latent sample from approx. posterior distribution.
                input_z = tf.concat((inputs, cell_output_prev), axis=1)
                self.q_pi_t = self.build_latent_dist(input_z, C.LATENT_Q, self.reuse)
                if self.use_reinforce:
                    self.z_t, self.pi_sample_t = VariationalCodebookCell.draw_deterministic_latent_sample_reinforce(self.q_pi_t, self.codebook_mu)
                else:
                    self.z_t = VariationalCodebookCell.draw_deterministic_latent_sample(self.q_pi_t, self.codebook_mu)
                    self.pi_sample_t = tf.expand_dims(tf.argmax(self.q_pi_t, axis=-1), axis=-1)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)

                # Estimate prior.
                self.p_pi_t = self.build_latent_dist(cell_output_prev, C.LATENT_P, self.reuse)

                # Update the internal cell by using cell state from the previous step and latent sample.
                input_cell = tf.concat((inputs, self.z_t_phi), axis=1)
                _, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                self.internal_cell_output = cell_output_prev

            # Prepare cell output.
            cell_output = [self.q_pi_t, self.p_pi_t, tf.to_float(self.pi_sample_t), self.internal_cell_output, self.z_t_phi]
            # Prepare cell state.
            cell_state = [self.internal_cell_state, self.z_t_phi]
            return tuple(cell_output), tuple(cell_state)

    def build_latent_dist(self, input_, scope, reuse):
        """
        Given the input parametrizes a Normal distribution.
        Args:
            input_:
            scope: "approximate_posterior" or "prior".
            reuse:
        Returns:
            mu and sigma tensors.
        """
        if self.latent_divisive_normalization:
            activation_fn = get_activation_fn(C.RELU)
        else:
            activation_fn = None  # tf.nn.softmax

        name = 'q' if scope == C.LATENT_Q else 'p_dists'
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope(name+'_pi', reuse=reuse):
                pi_t = linear(input_layer=self.phi(input_),
                              output_size=self.latent_num_components,
                              activation_fn=activation_fn,
                              is_training=self.is_training)
        return pi_t

    def register_sequence_components(self, cell_ops_list, output_ops_dict=None, eval_ops_dict=None, summary_ops_dict=None):
        """
        Gets dynamic_rnn outputs and fetches latent cell components.
        Args:
            cell_ops_list: list containing the approximate posterior and prior component ops.
            output_ops_dict:
            eval_ops_dict:
            summary_ops_dict:
        Returns:
        """
        # Follows the order in __call__ method.
        self.q_pi = cell_ops_list[0]
        self.p_pi = cell_ops_list[1]
        self.pi_sample = cell_ops_list[2]

        if self.latent_divisive_normalization:
            self.q_pi_probs = self.q_pi / tf.maximum(tf.reduce_sum(self.q_pi, axis=-1, keepdims=True), 1e-6)
            self.p_pi_probs = self.p_pi / tf.maximum(tf.reduce_sum(self.p_pi, axis=-1, keepdims=True), 1e-6)
        else:
            self.q_pi_probs = tf.nn.softmax(self.q_pi)
            self.p_pi_probs = tf.nn.softmax(self.p_pi)

        output_ops_dict["q_probs"] = self.q_pi_probs
        output_ops_dict["p_probs"] = self.p_pi_probs

        # Register latent ops and summaries.
        if output_ops_dict is not None:
            output_ops_dict[C.Q_PI] = self.q_pi_probs
            output_ops_dict[C.P_PI] = self.p_pi_probs
        if eval_ops_dict is not None:
            if not self.is_sampling:
                eval_ops_dict[C.Q_PI] = self.q_pi_probs
            eval_ops_dict[C.P_PI] = self.p_pi_probs
        if summary_ops_dict is not None:
            # Entropy of categorical distributions.
            summary_ops_dict["mean_entropy_" + C.Q_PI] = tf.reduce_mean(tf_loss.entropy(tf.reshape(self.q_pi_probs, [-1, self.latent_num_components])))
            summary_ops_dict["mean_entropy_" + C.P_PI] = tf.reduce_mean(tf_loss.entropy(tf.reshape(self.p_pi_probs, [-1, self.latent_num_components])))
            tf.summary.histogram("pi_sample", self.pi_sample, collections=[self.mode + '_summary_plot'])

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
                    self.ops_loss[loss_key] = self.loss_diversity_weight*reduce_loss_fn(sequence_mask*temporal_entropy_loss)
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


class StochasticCodebookCell(VariationalCodebookCell):
    """
    Latent space consists of embeddings either modeled by probabilistic Gaussian distributions with diagonal covariance
    or deterministic representation vectors.

    The encoder maps the inputs to the most relevant representation component. It is unsupervised. Hence, both the
    encoder and latent representations are learned.
    """
    def __init__(self, config, mode, reuse):
        super(StochasticCodebookCell, self).__init__(config, mode, reuse)

        # sigma parameters of codebook.
        with tf.variable_scope("latent_codebook", reuse=self.reuse):
            self.codebook_sigma = tf.get_variable(name="codebook_sigma", dtype=tf.float32, initializer=tf.constant_initializer(0.1), shape=[self.latent_num_components, self.latent_size_components])

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
        # dist_pi = tf.contrib.distributions.Categorical(logits=pi, name="dist_pi")
        # code_idx = tf.expand_dims(dist_pi.sample(), axis=-1)
        # code_idx = tf.expand_dims(tf.argmax(pi, axis=-1), axis=-1)
        code_idx = tf.multinomial(logits=pi, num_samples=1, name=None, output_dtype=tf.int32)
        mu = tf.gather_nd(codebook_mu, code_idx)
        sigma = tf.gather_nd(codebook_sigma, code_idx)

        eps = tf.random_normal(tf.stack([tf.shape(pi)[0], component_size]), 0.0, 1.0, dtype=tf.float32)
        z_sample = tf.add(mu, tf.multiply(sigma, eps))

        return z_sample, code_idx

    def __call__(self, inputs, state, scope=None):
        """
        Prior distribution is estimated by using information until the current time-step t. On the other hand,
        approximate-posterior distribution is estimated by using some future steps.
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            cell_state_prev, cell_output_prev, z_sample_prev = self.get_cell_state(state)

            if self.is_sampling:
                # The input is prediction from the previous step.
                # First update the internal cell by using cell state and latent sample from previous steps.
                input_cell = tf.concat((inputs, z_sample_prev), axis=1)
                self.internal_cell_output, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                # Draw a latent sample from prior distribution.
                self.p_pi_t = self.build_latent_dist(self.internal_cell_output, C.LATENT_P, self.reuse)
                if self.use_reinforce:
                    self.z_t, self.pi_sample_t = StochasticCodebookCell.draw_stochastic_latent_sample_reinforce(self.p_pi_t, self.codebook_mu, self.codebook_sigma)
                else:
                    self.z_t = StochasticCodebookCell.draw_stochastic_latent_sample(self.p_pi_t, self.codebook_mu, self.codebook_sigma)
                    self.pi_sample_t = tf.expand_dims(tf.argmax(self.p_pi_t, axis=-1), axis=-1)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)

                # Although we don't need q_mu_t, q_sigma_t in the sampling mode, __call__ method still requires them.
                # Make an arbitrary assignment.
                self.q_pi_t = self.p_pi_t

            elif self.is_training or self.is_validation:
                # First draw a latent sample from approx. posterior distribution.
                input_z = tf.concat((inputs, cell_output_prev), axis=1)
                self.q_pi_t = self.build_latent_dist(input_z, C.LATENT_Q, self.reuse)
                if self.use_reinforce:
                    self.z_t, self.pi_sample_t = StochasticCodebookCell.draw_stochastic_latent_sample_reinforce(self.q_pi_t, self.codebook_mu, self.codebook_sigma)
                else:
                    self.z_t = StochasticCodebookCell.draw_stochastic_latent_sample(self.q_pi_t, self.codebook_mu, self.codebook_sigma)
                    self.pi_sample_t = tf.expand_dims(tf.argmax(self.q_pi_t, axis=-1), axis=-1)
                self.z_t_phi = self.phi(self.z_t, scope="phi_z", reuse=self.reuse)

                # Estimate prior.
                self.p_pi_t = self.build_latent_dist(cell_output_prev, C.LATENT_P, self.reuse)

                # Update the internal cell by using cell state from the previous step and latent sample.
                input_cell = tf.concat((inputs, self.z_t_phi), axis=1)
                _, self.internal_cell_state = self.internal_cell(input_cell, cell_state_prev)
                self.internal_cell_output = cell_output_prev

            # Prepare cell output.
            cell_output = [self.q_pi_t, self.p_pi_t, tf.to_float(self.pi_sample_t), self.internal_cell_output, self.z_t_phi]
            # Prepare cell state.
            cell_state = [self.internal_cell_state, self.z_t_phi]
            return tuple(cell_output), tuple(cell_state)


class VRNNCell(tf.contrib.rnn.RNNCell):
    """
    Variational RNN cell.

    Training time behaviour: draws latent vectors from approximate posterior distribution and tries to decrease the
    discrepancy between prior and the approximate posterior distributions.

    Sampling time behaviour: draws latent vectors from the prior distribution to synthesize a sample. This synthetic
    sample is then used to calculate approximate posterior distribution which is fed to RNN to update the state.
    The inputs to the forward call are not used and can be dummy.
    """

    def __init__(self, reuse, mode, sample_fn, config):
        """

        Args:
            reuse: reuse model parameters.
            mode: 'training' or 'sampling'.
            sample_fn: function to generate sample given model outputs.

            config (dict): In addition to standard <key, value> pairs, stores the following dictionaries for rnn and
                output configurations.

                config['output_layer'] = {}
                config['output_layer']['out_keys']
                config['output_layer']['out_dims']
                config['output_layer']['out_activation_fn']

                config['*_rnn'] = {}
                config['*_rnn']['num_layers'] (default: 1)
                config['*_rnn']['cell_type'] (default: lstm)
                config['*_rnn']['size'] (default: 512)
        """
        self.reuse = reuse
        self.mode = mode
        self.sample_fn = sample_fn
        self.is_sampling = mode == 'sampling'
        self.is_evaluation = mode == "validation" or mode == "test"

        self.input_dims = config['input_dims']
        self.h_dim = config['hidden_size']
        self.latent_h_dim = config.get('latent_hidden_size', self.h_dim)
        self.z_dim = config['latent_size']
        self.additive_q_mu = config['additive_q_mu']

        self.dropout_keep_prob = config.get('input_keep_prop', 1)
        self.num_linear_layers = config.get('num_fc_layers', 1)
        self.use_latent_h_in_outputs = config.get('use_latent_h_in_outputs', True)
        self.use_batch_norm = config['use_batch_norm_fc']

        if not (mode == "training"):
            self.dropout_keep_prob = 1.0

        self.output_config = config['output_layer']

        self.output_size_ = [self.z_dim]*4
        self.output_size_.extend(self.output_config['out_dims']) # q_mu, q_sigma, p_mu, p_sigma + model outputs

        self.state_size_ = []
        # Optional. Linear layers will be used if not passed.
        self.input_rnn = False
        if 'input_rnn' in config and not(config['input_rnn'] is None) and len(config['input_rnn'].keys()) > 0:
            self.input_rnn = True
            self.input_rnn_config = config['input_rnn']

            self.input_rnn_cell = get_rnn_cell(scope='input_rnn', **config['input_rnn'])

            # Variational dropout
            if config['input_rnn'].get('use_variational_dropout', False):
                # TODO input dimensions are hard-coded.
                self.input_rnn_cell = tf.contrib.rnn.DropoutWrapper(self.input_rnn_cell, input_keep_prob=self.dropout_keep_prob, output_keep_prob=self.dropout_keep_prob, variational_recurrent=True, input_size=(216), dtype=tf.float32)
                self.dropout_keep_prob = 1.0

            self.state_size_.append(self.input_rnn_cell.state_size)

        self.latent_rnn_config = config['latent_rnn']
        self.latent_rnn_cell_type = config['latent_rnn']['cell_type']
        self.latent_rnn_cell = get_rnn_cell(scope='latent_rnn', **config['latent_rnn'])
        self.state_size_.append(self.latent_rnn_cell.state_size)

        # Optional. Linear layers will be used if not passed.
        self.output_rnn = False
        if 'output_rnn' in config and not(config['output_rnn'] is None) and len(config['output_rnn'].keys()) > 0:
            self.output_rnn = True
            self.output_rnn_config = config['output_rnn']

            self.output_rnn_cell = get_rnn_cell(scope='output_rnn', **config['output_rnn'])
            self.state_size_.append(self.output_rnn_cell.state_size)

        self.activation_func = get_activation_fn(config.get('fc_layer_activation_func', 'relu'))
        self.sigma_activaction_fn = tf.nn.softplus

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_size_)

    #
    # Auxiliary functions
    #
    def draw_sample(self):
        """
        Draws a sample by using cell outputs.

        Returns:

        """
        return self.sample_fn(self.output_components)

    def reparametrization(self, mu, sigma, scope):
        """
        Given an isotropic normal distribution (mu and sigma), draws a sample by using reparametrization trick:
        z = mu + sigma*epsilon

        Args:
            mu: mean of isotropic Gaussian distribution.
            sigma: standard deviation of isotropic Gaussian distribution.

        Returns:

        """
        with tf.variable_scope(scope):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            z = tf.add(mu, tf.multiply(sigma, eps))

            return z

    def phi(self, input_, scope, reuse=None):
        """
        A fully connected layer to increase model capacity and learn and intermediate representation. It is reported to
        be useful in https://arxiv.org/pdf/1506.02216.pdf

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope, reuse=reuse):
            phi_hidden = input_
            for i in range(self.num_linear_layers):
                phi_hidden = linear(phi_hidden, self.h_dim, self.activation_func, batch_norm=self.use_batch_norm)

            return phi_hidden

    def latent(self, input_, scope):
        """
        Creates mu and sigma components of a latent distribution. Given an input layer, first applies a fully connected
        layer and then calculates mu & sigma.

        Args:
            input_:
            scope:

        Returns:

        """
        with tf.variable_scope(scope):
            latent_hidden = linear(input_, self.latent_h_dim, self.activation_func, batch_norm=self.use_batch_norm)
            with tf.variable_scope("mu"):
                mu = linear(latent_hidden, self.z_dim)
            with tf.variable_scope("sigma"):
                sigma = linear(latent_hidden, self.z_dim, self.sigma_activaction_fn)

            return mu, sigma

    def parse_rnn_state(self, state):
        """
        Sets self.latent_h and rnn states.

        Args:
            state:

        Returns:

        """
        latent_rnn_state_idx = 0
        if self.input_rnn is True:
            self.input_rnn_state = state[0]
            latent_rnn_state_idx = 1
        if self.output_rnn is True:
            self.output_rnn_state = state[latent_rnn_state_idx+1]

        # Check if the cell consists of multiple cells.
        self.latent_rnn_state = state[latent_rnn_state_idx]

        if self.latent_rnn_cell_type == C.GRU:
            self.latent_h = self.latent_rnn_state[-1] if type(self.latent_rnn_state) == tuple else self.latent_rnn_state
        else:
            self.latent_h = self.latent_rnn_state[-1].h if type(self.latent_rnn_state) == tuple else self.latent_rnn_state.h

    #
    # Functions to build graph.
    #
    def build_training_graph(self, input_, state):
        """

        Args:
            input_:
            state:

        Returns:

        """
        self.parse_rnn_state(state)
        self.input_layer(input_, state)
        self.input_layer_hidden()

        self.latent_p_layer()
        self.latent_q_layer()
        #if self.is_evaluation:
        #    self.phi_z = self.phi_z_p
        #else:
        self.phi_z = self.phi_z_q

        self.output_layer_hidden()
        self.output_layer()
        self.update_latent_rnn_layer()

    def build_sampling_graph(self, input_, state):
        self.parse_rnn_state(state)
        self.latent_p_layer()
        self.phi_z = self.phi_z_p

        self.output_layer_hidden()
        self.output_layer()

        # Draw a sample by using predictive distribution.
        synthetic_sample = self.draw_sample()
        # TODO: Is dropout required in `sampling` mode?
        self.input_layer(synthetic_sample, state)
        self.input_layer_hidden()
        self.latent_q_layer()
        self.update_latent_rnn_layer()


    def input_layer(self, input_, state):
        """
        Set self.x by applying dropout.
        Args:
            input_:
            state:

        Returns:

        """
        with tf.variable_scope("input"):
            input_components = tf.split(input_, self.input_dims, axis=1)
            self.x = input_components[0]

    def input_layer_hidden(self):
        if self.input_rnn is True:
            self.phi_x_input, self.input_rnn_state = self.input_rnn_cell(self.x, self.input_rnn_state, scope='phi_x_input')
        else:
            self.phi_x_input = self.phi(self.x, scope='phi_x_input')

        if self.dropout_keep_prob < 1.0:
            self.phi_x_input = tf.nn.dropout(self.phi_x_input, keep_prob=self.dropout_keep_prob)

    def latent_q_layer(self):
        input_latent_q = tf.concat((self.phi_x_input, self.latent_h), axis=1)
        if self.additive_q_mu:
            q_mu_delta, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")
            self.q_mu = q_mu_delta + self.p_mu
        else:
            self.q_mu, self.q_sigma = self.latent(input_latent_q, scope="latent_z_q")

        q_z = self.reparametrization(self.q_mu, self.q_sigma, scope="z_q")
        self.phi_z_q = self.phi(q_z, scope="phi_z", reuse=True)

    def latent_p_layer(self):
        input_latent_p = tf.concat((self.latent_h), axis=1)
        self.p_mu, self.p_sigma = self.latent(input_latent_p, scope="latent_z_p")

        p_z = self.reparametrization(self.p_mu, self.p_sigma, scope="z_p")
        self.phi_z_p = self.phi(p_z, scope="phi_z")

    def output_layer_hidden(self):
        if self.use_latent_h_in_outputs is True:
            output_layer_hidden = tf.concat((self.phi_z, self.latent_h), axis=1)
        else:
            output_layer_hidden = tf.concat((self.phi_z), axis=1)

        if self.output_rnn is True:
            self.phi_x_output, self.output_rnn_state = self.output_rnn_cell(output_layer_hidden, self.output_rnn_state, scope='phi_x_output')
        else:
            self.phi_x_output = self.phi(output_layer_hidden, scope="phi_x_output")

    def output_layer(self):
        self.output_components = {}
        for key, size, activation_func in zip(self.output_config['out_keys'], self.output_config['out_dims'], self.output_config['out_activation_fn']):
            with tf.variable_scope(key):
                if not callable(activation_func):
                    activation_func = get_activation_fn(activation_func)
                output_component = linear(self.phi_x_output, size, activation_fn=activation_func)
                self.output_components[key] = output_component

    def update_latent_rnn_layer(self):
        input_latent_rnn = tf.concat((self.phi_x_input, self.phi_z), axis=1)
        self.latent_rnn_output, self.latent_rnn_state = self.latent_rnn_cell(input_latent_rnn, self.latent_rnn_state)

    def __call__(self, input_, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            if self.is_sampling:
                self.build_sampling_graph(input_, state)
            else:
                self.build_training_graph(input_, state)

            # Prepare cell output.
            vrnn_cell_output = [self.q_mu, self.q_sigma, self.p_mu, self.p_sigma]
            for key in self.output_config['out_keys']:
                vrnn_cell_output.append(self.output_components[key])

            # Prepare cell state.
            vrnn_cell_state = []
            if self.input_rnn:
                vrnn_cell_state.append(self.input_rnn_state)

            vrnn_cell_state.append(self.latent_rnn_state)

            if self.output_rnn:
                vrnn_cell_state.append(self.output_rnn_state)

            return tuple(vrnn_cell_output), tuple(vrnn_cell_state)


class ZForcingCell(tf.contrib.rnn.RNNCell):
    def __init__(self, forward_cell, reuse, mode, **kwargs):
        self.forward_cell = forward_cell
        self.reuse = reuse
        self.mode = mode
        self.is_sampling = mode == "sampling"
        self.is_validation = mode == "validation"
        self.is_training = mode == "training"

        config = kwargs
        self.backward_h_size = config["backward_h_size"] # List of input data and backward-cell output dimensionality.
        self.h_size = config["hidden_size"]
        self.latent_h_size = config.get("latent_hidden_size", self.h_size)
        self.z_size = config["latent_size"]
        self.num_linear_layers = config.get("num_fc_layers", 0)
        self.input_keep_prop = config.get("input_keep_prop", 0)
        self.output_config = config["output_layer"]

        # List containing output dimensions and cell dimension.
        self.output_dims_ = [self.z_size]*5
        self.output_dims_.extend(self.output_config["out_dims"])  # q_mu, q_sigma, p_mu, p_sigma, z + model outputs
        self.state_size_ = self.forward_cell.state_size

        self.activation_func = get_activation_fn(config.get("fc_layer_activation_func", "relu"))

        output_sigma_bias = config.get("output_sigma_bias", 0)
        self.sigma_activaction_fn = tf.nn.softplus

    @property
    def state_size(self):
        return tuple(self.state_size_)

    @property
    def output_size(self):
        return tuple(self.output_dims_)

    def __call__(self, input_, state, scope=None):
        """
        Args:
            input_: Input data and backward-cell output are concatenated.
            state: forward-cell state.
            scope:
        """
        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            if self.is_sampling:
                z, forward_cell_state = self.build_sampling_graph(input_, state)
            else:
                z, forward_cell_state = self.build_training_graph(input_, state)

            # Prepare cell output.
            cell_output = [self.q_mu, self.q_sigma, self.p_mu, self.p_sigma, z]
            for key in self.output_config['out_keys']:
                cell_output.append(self.output_components[key])

            return tuple(cell_output), forward_cell_state

    def get_inputs(self, input_, state):
        # In sampling mode backward cell state is not passed.
        # Assuming that we use LSTM + Check if the cell consists of multiple cells.
        forward_h = state[-1].h if type(state) == tuple else state.h
        if self.is_sampling:
            return input_, forward_h
        else:
            return input_[:, :-self.backward_h_size], input_[:, -self.backward_h_size:], forward_h

    def input_layer(self, input_):
        with tf.name_scope("input_layer"):
            return self.phi(input_, scope='phi_x_input')

    def latent_layer_q(self, forward_h, backward_h):
        with tf.variable_scope("approximate_posterior"):
            input_latent_q = tf.concat((forward_h, backward_h), axis=1)

            latent_hidden = linear(input_latent_q, self.latent_h_size, self.activation_func)
            with tf.variable_scope("q_mu"):
                mu = linear(latent_hidden, self.z_size)
            with tf.variable_scope("q_sigma"):
                sigma = linear(latent_hidden, self.z_size, self.sigma_activaction_fn)

            return mu, sigma

    def latent_layer_p(self, forward_h):
        with tf.variable_scope("prior"):
            latent_hidden = linear(forward_h, self.latent_h_size, self.activation_func)
            with tf.variable_scope("p_mu"):
                mu = linear(latent_hidden, self.z_size)
            with tf.variable_scope("p_sigma"):
                sigma = linear(latent_hidden, self.z_size, self.sigma_activaction_fn)

            return mu, sigma

    def output_layer_f(self, forward_h):
        with tf.name_scope("output_layer"):
            self.phi_x_forward_h = self.phi(forward_h, scope="phi_x_forward_h")

            self.output_components = {}
            for key, size, activation_func in zip(self.output_config['out_keys'], self.output_config['out_dims'], self.output_config['out_activation_fn']):
                with tf.variable_scope(key):
                    if not callable(activation_func):
                        activation_func = get_activation_fn(activation_func)
                    output_component = linear(self.phi_x_forward_h, size, activation_fn=activation_func)
                    self.output_components[key] = output_component

    def forward_rnn_layer(self, input_x, latent_z, forward_cell_state_prev):
        input_rnn = tf.concat((input_x, latent_z), axis=1)
        forward_h, forward_cell_state = self.forward_cell(input_rnn, forward_cell_state_prev)

        return forward_h, forward_cell_state

    def build_training_graph(self, input_, state):
        input_x, backward_h, forward_h = self.get_inputs(input_, state)

        self.q_mu, self.q_sigma = self.latent_layer_q(forward_h, backward_h)
        self.p_mu, self.p_sigma = self.latent_layer_p(forward_h)

        with tf.variable_scope('z', reuse=self.reuse):
            q_z = self.reparametrization(self.q_mu, self.q_sigma)
            phi_z_q = self.phi(q_z, scope="phi_z", reuse=self.reuse)

        self.output_layer_f(forward_h)
        forward_h_t, forward_cell_state = self.forward_rnn_layer(input_x, phi_z_q, state)

        return q_z, forward_cell_state

    def build_sampling_graph(self, input_, state):
        input_x, forward_h = self.get_inputs(input_, state)

        self.p_mu, self.p_sigma = self.latent_layer_p(forward_h)
        self.q_mu, self.q_sigma = self.p_mu, self.p_sigma # Need to return. They are not used.

        with tf.variable_scope('z', reuse=self.reuse):
            p_z = self.reparametrization(self.p_mu, self.p_sigma)
            phi_p_q = self.phi(p_z, scope="phi_z", reuse=self.reuse)

        self.output_layer_f(forward_h)
        forward_h_t, forward_cell_state = self.forward_rnn_layer(input_x, phi_p_q, state)

        return p_z, forward_cell_state

    # Auxiliary functions.
    def phi(self, input_, scope, reuse=None):
        """
        A fully connected layer to increase model capacity and learn and intermediate representation. It is reported to
        be useful in https://arxiv.org/pdf/1506.02216.pdf
        """
        with tf.variable_scope(scope, reuse=reuse):
            phi_hidden = input_
            for i in range(self.num_linear_layers):
                phi_hidden = linear(phi_hidden, self.h_size, self.activation_func)
            return phi_hidden

    def reparametrization(self, mu, sigma):
        """
        Given an isotropic normal distribution (mu and sigma), draws a sample by using reparametrization trick:
        z = mu + sigma*epsilon
        """
        with tf.name_scope("reparametrization"):
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            z = tf.add(mu, tf.multiply(sigma, eps))
            return z
