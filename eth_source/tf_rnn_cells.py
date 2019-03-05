import tensorflow as tf

from tf_model_utils import linear, get_activation_fn, get_rnn_cell, fully_connected_layer, get_decay_variable
import tf_loss
from constants import Constants as C


class LatentCell(tf.contrib.rnn.RNNCell):
    """
    Base class for latent cells.

    Instances of this cell is used by rnn wrappers (e.g., dynamic_rnn, static_rnn). Hence, for a given input step t, all
    calculations are made for this single step. The latent components at step t are returned as part of the output.

    The rnn wrapper accumulates outputs of all steps. In order to access latent components of the whole sequence
    register_sequence_components method should be called first. Then the cell can operate on the entire sequence (i.e.,
    implementing latent loss terms, etc.)
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(LatentCell, self).__init__()

        self.config = config
        self.reuse = reuse
        assert mode in [C.TRAIN, C.VALID, C.TEST, C.SAMPLE]
        self.mode = mode
        self.is_sampling = mode == C.SAMPLE
        self.is_validation = mode == C.VALID or mode == C.TEST
        self.is_training = mode == C.TRAIN
        self.global_step = kwargs.get("global_step", None)

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
            return GaussianLatentCell(config, mode, reuse, **kwargs)
        else:
            raise Exception("Unknown latent cell.")


class GaussianLatentCell(LatentCell):
    """
    VAE latent space for time-series data, modeled by a Gaussian distribution with diagonal covariance matrix.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(GaussianLatentCell, self).__init__(config, mode, reuse, **kwargs)

        self.use_temporal_kld = self.config.get('use_temporal_kld', False)
        self.tkld_weight = self.config.get('tkld_weight', 0.1)

        # Annealing KL-divergence weight or using fixed weight.
        kld_weight = self.config.get('kld_weight', 1)
        if isinstance(kld_weight, dict) and self.global_step is not None:
            self.kld_weight = get_decay_variable(global_step=self.global_step, config=kld_weight, name="kld_weight")
        else:
            self.kld_weight = kld_weight

        # It is always 1 when we report the loss.
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

    def build_loss(self, sequence_mask, reduce_loss_fn, loss_ops_dict=None, **kwargs):
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
            if loss_ops_dict is not None:
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
                if loss_ops_dict is not None:
                    loss_ops_dict[loss_key] = self.ops_loss[loss_key]
        return self.ops_loss
