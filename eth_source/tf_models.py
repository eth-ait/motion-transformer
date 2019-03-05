import tensorflow as tf
import tf_loss
from tf_model_utils import linear, fully_connected_layer, get_activation_fn, get_decay_variable
from constants import Constants

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
        if layer_type == C.LATENT_LADDER_GAUSSIAN:
            return LadderLatentLayer(config, mode, reuse, **kwargs)
        elif layer_type == C.LATENT_STRUCTURED_HUMAN:
            return StructuredLadderLatentLayer(config, mode, reuse, **kwargs)
        else:
            raise Exception("Unknown latent layer.")


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
                        if not self.is_training and loss_ops_dict is not None:
                            loss_ops_dict["KL"+str(sl)] = tf.stop_gradient(kld_term)

                        self.kld_loss_terms.append(kld_term)
                        kld_loss += kld_term
                        if eval_dict is not None:
                            eval_dict["summary_kld_" + str(sl)] = kld_term
                            eval_dict["sequence_kld_" + str(sl)] = seq_kld_loss

                # Optimization is done through the accumulated term (i.e., loss_ops_dict[loss_key]).
                self.ops_loss[loss_key] = kld_loss
                if loss_ops_dict is not None:
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


class StructuredLadderLatentLayer(LadderLatentLayer):
    """
    Ladder VAE latent space for time-series data with human body structure in the hierarchy of latent variables. Each
    random latent variable corresponds to a joint in the skeleton.
    """
    def __init__(self, config, mode, reuse, **kwargs):
        super(StructuredLadderLatentLayer, self).__init__(config, mode, reuse, **kwargs)
        """
        # [(Parent ID, Joint ID, Joint Name), (...)] where each entry in a list corresponds to the joints at the same
        # level in the joint tree.
        # This also sets the number of stochastic layers. Hence, the number of deterministic layers should be equal to
        # vertical_dilation x 7.
        self.structure = [[(-1, 0, "Hips")],
                          [(0, 1, "RightUpLeg"), (0, 5, "LeftUpLeg"), (0, 9, "Spine")],
                          [(1, 2, "RightLeg"), (5, 6, "LeftLeg"), (9, 10, "Spine1")],
                          [(2, 3, "RightFoot"), (6, 7, "LeftFoot"), (10, 17, "RightShoulder"), (10, 13, "LeftShoulder"), (10, 11, "Neck")],
                          [(3, 4, "RightToeBase"), (7, 8, "LeftToeBase"), (17, 18, "RightArm"), (13, 14, "LeftArm"), (11, 12, "Head")],
                          [(18, 19, "RightForeArm"), (14, 15, "LeftForeArm")],
                          [(19, 20, "RightHand"), (15, 16, "LeftHand")]]
        """
        self.structure = kwargs["structure"]
        self.config['latent_size'] = self.config['latent_size'] if isinstance(self.config['latent_size'], list) else [self.config['latent_size']]*len(self.structure)
        # Store latent distributions by using corresponding unique joint ID.
        self.q_dists = dict()
        self.p_dists = dict()
        self.posterior_dists = dict()
        self.latent_samples_indexed = dict()  # Stores latent samples along with join ID and joint name.

    def create_p_variable(self, deterministic_inp, stochastic_inp, idx, name):
        scope = C.LATENT_P + "_" + str(idx + 1) + "_" + name
        input_list = []
        if deterministic_inp is not None:
            input_list.append(deterministic_inp)
        if stochastic_inp is not None:
            input_list.append(stochastic_inp)
        inputs = tf.concat(input_list, axis=-1)

        if not self.dynamic_prior:
            # Use N(0,1) prior.
            with tf.name_scope(scope):
                latent_size = self.config['latent_size'][idx]
                prior_shape = (tf.shape(inputs)[0], tf.shape(inputs)[1], latent_size)
                return tf.zeros(prior_shape, dtype=tf.float32), tf.ones(prior_shape, dtype=tf.float32)
        else:
            return self.build_latent_dist(inputs, idx=idx, scope=scope, reuse=self.reuse)[0]

    def create_q_variable(self, deterministic_inp, stochastic_inp, prior, idx, name):
        scope = C.LATENT_Q + "_" + str(idx + 1) + "_" + name
        input_list = []
        if deterministic_inp is not None:
            input_list.append(deterministic_inp)
        if stochastic_inp is not None:
            input_list.append(stochastic_inp)
        inputs = tf.concat(input_list, axis=-1)
        q_dist_approx, q_dist_approx_flat = self.build_latent_dist(inputs, idx=idx, scope=scope, reuse=self.reuse)

        # Estimate the approximate posterior distribution as a precision-weighted combination.
        if self.precision_weighted_update:
            scope = C.LATENT_Q + "_pwu_" + str(idx + 1) + "_" + name
            return self.combine_normal_dist(q_dist_approx, prior, scope=scope)
        else:
            return q_dist_approx

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
        self.num_d_layers = len(q_input)
        assert self.num_d_layers == self.vertical_dilation*len(self.structure), "# of deterministic layers != vertical dilation * len(structure)."
        self.num_s_layers = len(self.structure)

        # Build hierarchy.
        if self.top_down_latents:
            loop_indices = range(self.num_s_layers-1, -1, -1)
        else:
            loop_indices = range(0, self.num_s_layers, 1)
        # The hierarchy in self.structure list is the same. The root is always the first entry.
        skeleton_indices = range(0, self.num_s_layers, 1)

        for sl_idx, skeleton_idx in zip(loop_indices, skeleton_indices):
            dl_idx = (sl_idx + 1)*self.vertical_dilation - 1

            for joint in self.structure[skeleton_idx]:
                parent_idx, joint_idx, joint_name = joint
                posterior_sample_scope = "posterior_sample_" + str(sl_idx + 1) + "_" + joint_name

                # Estimate the prior of the first stochastic layer.
                # Draw a latent sample from the parent posterior.
                parent_sample = None
                if parent_idx >= 0:
                    parent_sample = self.draw_latent_sample(posterior_mu=self.posterior_dists[parent_idx][0][0],
                                                            posterior_sigma=self.posterior_dists[parent_idx][0][1],
                                                            scope=posterior_sample_scope)

                p_dist = self.create_p_variable(deterministic_inp=p_input[dl_idx],
                                                stochastic_inp=parent_sample,
                                                idx=sl_idx, name=joint_name)
                self.p_dists[joint_idx] = p_dist

                # Estimate the approximate posterior.
                # If it is not training, then we draw latent samples from the prior distribution.
                if self.is_sampling:
                    self.posterior_dists[joint_idx] = (p_dist, joint_name)
                else:
                    parent_sample = None
                    if parent_idx >= 0:
                        # Draw a latent sample from the parent posterior.
                        parent_sample = self.draw_latent_sample(posterior_mu=self.posterior_dists[parent_idx][0][0],
                                                                posterior_sigma=self.posterior_dists[parent_idx][0][1],
                                                                scope=posterior_sample_scope)

                    q_dist = self.create_q_variable(deterministic_inp=q_input[dl_idx],
                                                    stochastic_inp=parent_sample,
                                                    prior=p_dist, idx=sl_idx, name=joint_name)
                    self.q_dists[joint_idx] = q_dist
                    self.posterior_dists[joint_idx] = (q_dist, joint_name)  # Set the posterior.

        # TODO Joint name in the scope.
        for idx in sorted(self.posterior_dists.keys()):
            posterior_dist, joint_name = self.posterior_dists[idx]
            posterior_sample_scope = "posterior_sample_" + str(idx + 1)
            latent_sample = self.draw_latent_sample(posterior_mu=posterior_dist[0],
                                                    posterior_sigma=posterior_dist[1],
                                                    scope=posterior_sample_scope)
            self.latent_samples_indexed[idx] = latent_sample
            self.latent_samples.append(latent_sample)
        return self.latent_samples
