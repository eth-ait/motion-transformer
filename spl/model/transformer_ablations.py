"""
Implementing ablations on sharing various components for all joints as opposed
to using separate components per joint.
"""
"""
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
"""

import time

import numpy as np
import tensorflow as tf
from spl.model.base_model import BaseModel
from common.constants import Constants as C


# The two dimensional Transformer
class Transformer2d(BaseModel):
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.num_heads_temporal = config.get('transformer_num_heads_temporal')
        self.num_heads_spacial = config.get('transformer_num_heads_spacial')
        self.d_model = config.get('transformer_d_model')
        self.num_layers = config.get('transformer_num_layers')
        self.dropout_rate = config.get('transformer_dropout_rate')
        self.dff = config.get('transformer_dff')
        self.lr_type = config.get('transformer_lr')
        self.warm_up_steps = config.get('transformer_warm_up_steps')  # 1000 for h3.6m, 10000 for amass
        self.shared_embedding_layer = config.get('shared_embedding_layer', False)
        self.shared_output_layer = config.get('shared_output_layer', False)
        self.shared_temporal_layer = config.get('shared_temporal_layer', False)
        self.shared_spatial_layer = config.get('shared_spatial_layer', False)
        self.shared_attention_block = config.get('shared_attention_block', False)
        self.shared_pw_ffn = config.get('shared_pw_ffn', False)
        self.residual_attention_block = config.get('residual_attention_block', False)

        super(Transformer2d, self).__init__(config, data_pl, mode, reuse, **kwargs)

        self.window_len = config.get('transformer_window_length')  # attention window length
        self.random_window_min = config.get('random_window_min', 0)
        self.temporal_mask_drop = config.get('temporal_mask_drop', 0.0)

        # data
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1
        else:
            self.sequence_length = self.source_seq_len
        
        self.loss_seq_len = config.get('loss_seq_len', 0)
        if self.loss_seq_len == 0:
            self.loss_seq_len = self.sequence_length
            
        self.target_input = self.data_inputs[:, :-1, :]
        self.target_real = self.data_inputs[:, 1:, :]
        
        self.pos_encoding = self.positional_encoding()
        self.look_ahead_mask = self.create_look_ahead_mask()
        self.attention_weights = None  # Set later

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
            config['input_dropout_rate'] = args.input_dropout_rate
    
            config['residual_velocity'] = args.residual_velocity
            config['loss_type'] = args.loss_type
    
            config['transformer_lr'] = args.transformer_lr
            config['transformer_d_model'] = args.transformer_d_model
            config['transformer_dropout_rate'] = args.transformer_dropout_rate
            config['transformer_dff'] = args.transformer_dff
            config['transformer_num_layers'] = args.transformer_num_layers
            config['transformer_num_heads_temporal'] = args.transformer_num_heads_temporal
            config['transformer_num_heads_spacial'] = args.transformer_num_heads_spacial
            config['transformer_warm_up_steps'] = args.warm_up_steps
            config['transformer_window_length'] = args.transformer_window_length
            config['shared_embedding_layer'] = args.shared_embedding_layer
            config['shared_output_layer'] = args.shared_output_layer
            config['shared_temporal_layer'] = args.shared_temporal_layer
            config['shared_spatial_layer'] = args.shared_spatial_layer
            config['shared_attention_block'] = args.shared_attention_block
            config['shared_pw_ffn'] = args.shared_pw_ffn
            config['residual_attention_block'] = args.residual_attention_block
            config['random_window_min'] = args.random_window_min
            config['temporal_mask_drop'] = args.temporal_mask_drop
            
            config['random_noise_ratio'] = args.random_noise_ratio
            config['loss_seq_len'] = args.loss_seq_len
        else:
            config = from_config

        if args.new_experiment_id is not None:
            config["experiment_id"] = args.new_experiment_id
        else:
            config["experiment_id"] = str(int(time.time()))
        experiment_name_format = "{}-{}-{}_{}-b{}-in{}_out{}-t{}-s{}-l{}-dm{}-df{}-w{}-{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        config["model_type"],
                                                        "h36m" if config["use_h36m"] else "amass",
                                                        config["data_type"],
                                                        config["batch_size"],
                                                        config["source_seq_len"],
                                                        config["target_seq_len"],
                                                        config["transformer_num_heads_temporal"],
                                                        config["transformer_num_heads_spacial"],
                                                        config["transformer_num_layers"],
                                                        config["transformer_d_model"],
                                                        config["transformer_dff"],
                                                        config["transformer_window_length"],
                                                        config["seed"])
        return config, experiment_name

    def _learning_rate_scheduler(self, global_step):
        d_model = tf.cast(self.d_model, tf.float32)
        step = tf.cast(global_step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warm_up_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def optimization_routines(self):
        """Creates and optimizer, applies gradient regularizations and sets the parameter_update operation."""
        global_step = tf.train.get_global_step(graph=None)
        if self.lr_type == 1:
            learning_rate = self._learning_rate_scheduler(global_step)
        else:
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

    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask):
        # attn_dim: num_joints for spatial and seq_len for temporal
        '''
        The scaled dot product attention mechanism introduced in the Transformer
        :param q: the query vectors matrix (..., attn_dim, d_model/num_heads)
        :param k: the key vector matrix (..., attn_dim, d_model/num_heads)
        :param v: the value vector matrix (..., attn_dim, d_model/num_heads)
        :param mask: a mask for attention
        :return: the updated encoding and the attention weights matrix
        '''

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., num_heads, attn_dim, attn_dim)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., num_heads, attn_dim, attn_dim)

        output = tf.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)

        return output, attention_weights

    def create_look_ahead_mask(self):
        '''
        create a look ahead mask given a certain window length
        :return: the mask (window_length, window_length)
        '''
        ahead_mask = 1 - tf.linalg.band_part(tf.ones((self.sequence_length, self.sequence_length)), -1, 0)

        if self.random_window_min > 0 and self.is_training:
            random_win_len = tf.random.uniform([self.sequence_length], self.random_window_min, self.window_len, dtype=tf.int32)
            window_mask_padding = tf.maximum(0, tf.range(1, self.sequence_length + 1) - random_win_len)
        else:
            window_mask_padding = tf.maximum(0, tf.range(1, self.sequence_length+1) - self.window_len)
        window_mask = tf.sequence_mask(window_mask_padding, dtype=tf.float32, maxlen=self.sequence_length)

        mask = tf.maximum(ahead_mask, window_mask)
        
        if self.temporal_mask_drop > 0 and self.is_training:
            drop_mask = tf.cast(tf.random.uniform((self.sequence_length, self.sequence_length), 0, 1) < self.temporal_mask_drop, tf.float32)
            mask_ = tf.maximum(mask, drop_mask)
            # Ensure that the first timestep is not masked.
            mask = tf.concat([mask[:, 0:1], mask_[:, 1:]], axis=1)
            
        return mask  # (seq_len, seq_len)

    def get_angles(self, pos, i):
        '''
        calculate the angles giving position and i for the positional encoding formula
        :param pos: pos in the formula
        :param i: i in the formula
        :return: angle rad
        '''
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.d_model))
        return pos * angle_rates

    def positional_encoding(self):
        '''
        calculate the positional encoding given the window length
        :return: positional encoding (1, window_length, 1, d_model)
        '''
        angle_rads = self.get_angles(np.arange(self.sequence_length)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :])

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :, np.newaxis, :]

        return tf.cast(pos_encoding, dtype=tf.float32)  # (1, seq_len, 1, d_model)

    def sep_split_heads(self, x, batch_size, seq_len, num_heads):
        '''
        split the embedding vector for different heads for the temporal attention
        :param x: the embedding vector (batch_size, seq_len, d_model)
        :param batch_size: batch size
        :param seq_len: sequence length
        :param num_heads: number of temporal heads
        :return: the split vector (batch_size, num_heads, seq_len, depth)
        '''
        depth = self.d_model // num_heads
        x = tf.reshape(x, (batch_size, seq_len, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def sep_temporal_attention(self, x, mask, scope):
        '''
        the temporal attention block
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param mask: temporal mask (usually the look ahead mask)
        :param scope: the name of the scope
        :return: the output (batch_size, seq_len, num_joints, d_model)
        '''
        outputs = []
        attn_weights = []
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)

        value_var_scope = scope + "_value"
        key_var_scope = scope + "_key"
        dense_var_scope = scope + '_output_dense'
        # different joints have different embedding matrices.
        for joint_idx in range(self.NUM_JOINTS):
    
            query_var_scope = scope + "_query_" + str(joint_idx)
            if not self.shared_temporal_layer:
                value_var_scope = scope + "_value_" + str(joint_idx)
                key_var_scope = scope + "_key_" + str(joint_idx)
                dense_var_scope = scope + "_output_dense_" + str(joint_idx)
            
            # get the representation vector of the joint
            joint_rep = x[joint_idx]  # (batch_size, seq_len, d_model)
            
            with tf.variable_scope(query_var_scope, reuse=tf.AUTO_REUSE):
                q = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
            with tf.variable_scope(key_var_scope, reuse=tf.AUTO_REUSE):
                k = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
            with tf.variable_scope(value_var_scope, reuse=tf.AUTO_REUSE):
                v = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)

            # split it to several attention heads
            q = self.sep_split_heads(q, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            k = self.sep_split_heads(k, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            v = self.sep_split_heads(v, batch_size, seq_len, self.num_heads_temporal)
            # (batch_size, num_heads, seq_len, depth)
            # calculate the updated encoding by scaled dot product attention
            scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
            # (batch_size, num_heads, seq_len, depth)
            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            # (batch_size, seq_len, num_heads, depth)

            # concatenate the outputs from different heads
            concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.d_model])
            # (batch_size, seq_len, d_model)

            # go through a fully connected layer
            with tf.variable_scope(dense_var_scope, reuse=tf.AUTO_REUSE):
                output = tf.layers.dense(concat_attention, self.d_model)  # (batch_size, seq_len, d_model)
            
            outputs += [tf.expand_dims(output, axis=2)]
            last_attention_weights = attention_weights[:, :, -1, :]  # (batch_size, num_heads, seq_len)
            attn_weights += [last_attention_weights]
            
        outputs = tf.concat(outputs, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        attn_weights = tf.stack(attn_weights, axis=1)  # (batch_size, num_joints, num_heads, seq_len)
        return outputs, attn_weights

    def split_heads(self, x, shape0, shape1, attn_dim, num_heads):
        '''
        split the embedding vector for different heads for the spatial attention
        :param x: the embedding vector (batch_size, seq_len, num_joints, d_model)
        :param shape0: batch size
        :param shape1: sequence length
        :param attn_dim: number of joints
        :param num_heads: number of heads
        :return: the split vector (batch_size, seq_len, num_heads, num_joints, depth)
        '''
        depth = self.d_model // num_heads
        x = tf.reshape(x, (shape0, shape1, attn_dim, num_heads, depth))
        return tf.transpose(x, perm=[0, 1, 3, 2, 4])

    def sep_spacial_attention(self, x, mask, scope):
        '''
        the spatial attention block
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param mask: spatial mask (usually None)
        :param scope: the name of the scope
        :return: the output (batch_size, seq_len, num_joints, d_model)
        '''
        # Embed each vector to key, value and query vectors
        # Different joints have different query embedding matrices
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)
        
        q_joints, v_joints, k_joints = [], [], []
        value_var_scope = scope + "_value"
        key_var_scope = scope + "_key"
        for joint_idx in range(self.NUM_JOINTS):
    
            query_var_scope = scope + "_query_" + str(joint_idx)
            # if not self.shared_spatial_layer:
            #     value_var_scope = scope + "_value_" + str(joint_idx)
            #     key_var_scope = scope + "_key_" + str(joint_idx)
            
            joint_rep = x[joint_idx]
            # embed each vector to key, value and query vectors
            with tf.variable_scope(query_var_scope, reuse=tf.AUTO_REUSE):
                q = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                q_joints += [tf.expand_dims(q, axis=2)]
            with tf.variable_scope(key_var_scope, reuse=tf.AUTO_REUSE):
                k = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                k_joints += [tf.expand_dims(k, axis=2)]
            with tf.variable_scope(value_var_scope, reuse=tf.AUTO_REUSE):
                v = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                v_joints += [tf.expand_dims(v, axis=2)]
        
        q_joints = tf.concat(q_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        k_joints = tf.concat(k_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        v_joints = tf.concat(v_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        batch_size = tf.shape(q_joints)[0]
        seq_len = tf.shape(q_joints)[1]

        # split it to several attention heads
        q = self.split_heads(q_joints, batch_size, seq_len,
                             self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        k = self.split_heads(k_joints, batch_size, seq_len,
                             self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        v = self.split_heads(v_joints, batch_size, seq_len,
                             self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)

        # calculate the updated encoding by scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        # concatenate the outputs from different heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])
        concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.NUM_JOINTS, self.d_model])
        # (batch_size, seq_len, num_joints, d_model)

        # go through a fully connected layer
        if self.shared_spatial_layer:
            with tf.variable_scope(scope + '_output_dense', reuse=tf.AUTO_REUSE):
                output = tf.layers.dense(concat_attention, self.d_model)
        else:
            out_list = []
            for joint_idx in range(self.NUM_JOINTS):
                joint_out = concat_attention[:, :, joint_idx]
                with tf.variable_scope(scope + '_output_dense_' + str(joint_idx), reuse=tf.AUTO_REUSE):
                    dense_out = tf.layers.dense(joint_out, self.d_model)
                out_list.append(tf.expand_dims(dense_out, axis=2))
            output = tf.concat(out_list, axis=2)
        
        attention_weights = attention_weights[:, -1, :, :, :]  # (batch_size, num_heads, num_joints, num_joints)
            
        return output, attention_weights

    def point_wise_feed_forward_network(self, inputs, scope, share_layer=False):
        '''
        The feed forward block
        :param inputs: inputs (batch_size, seq_len, num_joints, d_model)
        :param scope: the name of the scope
        :param share_layer: whether to use the same layer for all joints or not.
        :return: outputs (batch_size, seq_len, num_joints, d_model)
        '''
        outputs = []
        if share_layer:
            with tf.variable_scope(scope + '_ff1', reuse=tf.AUTO_REUSE):
                joint_outputs = tf.layers.dense(inputs, self.dff, activation=tf.nn.relu)
            with tf.variable_scope(scope + '_ff2', reuse=tf.AUTO_REUSE):
                outputs = tf.layers.dense(joint_outputs, self.d_model)
        else:
            # different joints have different embedding matrices
            inputs = tf.transpose(inputs, [2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)
            for idx in range(self.NUM_JOINTS):
                with tf.variable_scope(scope + '_ff1_' + str(idx), reuse=tf.AUTO_REUSE):
                    joint_outputs = tf.layers.dense(inputs[idx], self.dff, activation=tf.nn.relu)
                with tf.variable_scope(scope + '_ff2_' + str(idx), reuse=tf.AUTO_REUSE):
                    joint_outputs = tf.layers.dense(joint_outputs, self.d_model)
                outputs += [tf.expand_dims(joint_outputs, axis=2)]
            outputs = tf.concat(outputs, axis=2)  # (batch_size, seq_len, num_joints, d_model)
            # outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], self.NUM_JOINTS, self.d_model])
        return outputs

    def para_transformer_layer(self, x, look_ahead_mask, scope):
        '''
        The layer with spatial and temporal blocks in parallel
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param look_ahead_mask: the look ahead mask
        :param scope: the name of the scope
        :return: outputs (batch_size, seq_len, num_joints, d_model) and the attention blocks
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # temporal attention
            attn1, attn_weights_block1 = self.sep_temporal_attention(x, look_ahead_mask, scope="temporal_attn")
            with tf.variable_scope("dropout_temporal", reuse=tf.AUTO_REUSE):
                attn1 = tf.layers.dropout(attn1, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln_temporal", reuse=tf.AUTO_REUSE):
                temporal_out = tf.contrib.layers.layer_norm(attn1 + x)

            # spatial attention
            attn2, attn_weights_block2 = self.sep_spacial_attention(x, None, scope="spatial_attn")
            with tf.variable_scope("dropout_spatial", reuse=tf.AUTO_REUSE):
                attn2 = tf.layers.dropout(attn2, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln_spatial", reuse=tf.AUTO_REUSE):
                spatial_out = tf.contrib.layers.layer_norm(attn2 + x)
                
            # add the temporal output and the spatial output
            out = temporal_out + spatial_out

            # feed forward
            ffn_output = self.point_wise_feed_forward_network(out, scope='feed_forward', share_layer=self.shared_pw_ffn)
            with tf.variable_scope("dropout_ff", reuse=tf.AUTO_REUSE):
                ffn_output = tf.layers.dropout(ffn_output, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln_ff", reuse=tf.AUTO_REUSE):
                final = tf.contrib.layers.layer_norm(ffn_output + out)

            return final, attn_weights_block1, attn_weights_block2

    def transformer(self, inputs, look_ahead_mask):
        '''
        The attention blocks
        :param inputs: inputs (batch_size, seq_len, num_joints, joint_size)
        :param look_ahead_mask: the look ahead mask
        :return: outputs (batch_size, seq_len, num_joints, joint_size)
        '''
        # encode each rotation matrix to the feature space (d_model)
        # different joints have different encoding matrices
        inputs = tf.transpose(inputs, [2, 0, 1, 3])  # (num_joints, batch_size, seq_len, joint_size)
        embed = []

        inp_drop_rate = self.config.get("input_dropout_rate", 0)
        if inp_drop_rate > 0:
            # Apply dropout on the entire rotation matrix of a joint.
            # inp_shape = tf.shape(inputs)
            # noise_shape = (inp_shape[0], inp_shape[1], inp_shape[2], 1)
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                inputs = tf.layers.dropout(inputs,
                                           rate=inp_drop_rate,
                                           seed=self.config["seed"],
                                           training=self.is_training)
        
        for joint_idx in range(self.NUM_JOINTS):
            emb_var_scope = "embedding_" + str(joint_idx)
            if self.shared_embedding_layer:
                emb_var_scope = "embedding"
            with tf.variable_scope(emb_var_scope, reuse=tf.AUTO_REUSE):
                joint_rep = tf.layers.dense(inputs[joint_idx], self.d_model)  # (batch_size, seq_len, d_model)
                embed += [tf.expand_dims(joint_rep, axis=2)]
        x = tf.concat(embed, axis=2)
        # x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.NUM_JOINTS, self.d_model])
        # add the positional encoding
        x += self.pos_encoding

        with tf.variable_scope("embedding_dropout", reuse=self.reuse):
            x = tf.layers.dropout(x, training=self.is_training, rate=self.dropout_rate)

        # put into several attention layers
        # (batch_size, seq_len, num_joints, d_model)
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}
        for i in range(self.num_layers):
            if self.shared_attention_block:
                transformer_var_scope = "transformer_layer"
            else:
                transformer_var_scope = "transformer_layer_" + str(i)
                
            x_out, block1, block2 = self.para_transformer_layer(x, look_ahead_mask, scope=transformer_var_scope)
            attention_weights_temporal += [block1]  # (batch_size, num_joints, num_heads, seq_len)
            attention_weights_spatial += [block2]  # (batch_size, num_heads, num_joints, num_joints)
            if self.residual_attention_block:
                x += x_out
            else:
                x = x_out
        # (batch_size, seq_len, num_joints, d_model)
        
        attention_weights['temporal'] = tf.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights['spatial'] = tf.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # decode each feature to the rotation matrix space
        if self.shared_output_layer:
            out_var_scope = "final_output"
            with tf.variable_scope(out_var_scope, reuse=self.reuse):
                final_output = tf.layers.dense(x, self.JOINT_SIZE)
        else:
            # different joints have different decoding matrices
            output = []
            # (num_joints, batch_size, seq_len, joint_size)
            x = tf.transpose(x, [2, 0, 1, 3])
            for joint_idx in range(self.NUM_JOINTS):
                out_var_scope = "final_output_" + str(joint_idx)
                
                with tf.variable_scope(out_var_scope, reuse=tf.AUTO_REUSE):
                    joint_output = tf.layers.dense(x[joint_idx], self.JOINT_SIZE)
                    output += [tf.expand_dims(joint_output, axis=2)]
            final_output = tf.concat(output, axis=2)

        return final_output, attention_weights

    def build_network(self):
        shape = tf.shape(self.target_input)
        batch_siz = shape[0]
        seq_len = shape[1]
        target_input = self.target_input
        target_input = tf.reshape(target_input, [batch_siz, seq_len, self.NUM_JOINTS, self.JOINT_SIZE])

        rand_noise = self.config.get("random_noise_ratio", 0)
        if rand_noise > 0:
            target_input += tf.random.uniform(tf.shape(target_input), minval=-rand_noise, maxval=rand_noise)
        
        outputs, attn_weights = self.transformer(target_input, self.look_ahead_mask)
        outputs = tf.reshape(outputs, [batch_siz, seq_len, self.HUMAN_SIZE])
        if self.residual_velocity:
            outputs += self.target_input
        
        self.attention_weights = attn_weights
        return outputs

    def build_loss(self):
        predictions_pose = self.outputs[:, :self.loss_seq_len]
        targets_pose = self.target_real[:, :self.loss_seq_len]
        # seq_len = self.sequence_length
        seq_len = self.loss_seq_len

        with tf.name_scope("loss_angles"):
            diff = targets_pose - predictions_pose
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
                raise Exception("Unknown angle loss.")

        return loss_

    def step(self, session):
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
        assert self.is_eval, "Only works in sampling mode."
        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        # To get rid of 0 paddings.
        seq_len = batch[C.BATCH_SEQ_LEN]
        max_len = seq_len.max()
        if (seq_len != max_len).sum() != 0:
            for i in range(seq_len.shape[0]):
                len_ = seq_len[i]
                data_sample[i, len_:] = np.tile(data_sample[i, len_-1], (max_len-len_, 1))
            
        targets = data_sample[:, self.source_seq_len:, :]
        seed_sequence = data_sample[:, :self.source_seq_len, :]
        prediction, attentions = self.sample(session=session,
                                 seed_sequence=seed_sequence,
                                 prediction_steps=self.target_seq_len)
        return prediction, targets, seed_sequence, data_id, attentions

    def sample(self, session, seed_sequence, prediction_steps, **kwargs):
        assert self.is_eval, "Only works in sampling mode."

        # input_sequence = seed_sequence[:, -self.window_len:, :]
        input_sequence = seed_sequence
        num_steps = prediction_steps
        dummy_frame = np.zeros([seed_sequence.shape[0], 1, seed_sequence.shape[2]])
        predictions = []
        attentions = []

        for step in range(num_steps):
            # Insert a dummy frame since the model shifts the inputs by one step.
            model_inputs = np.concatenate([input_sequence, dummy_frame], axis=1)
            model_outputs, attention = session.run([self.outputs, self.attention_weights], feed_dict={self.data_inputs: model_inputs})
            prediction = model_outputs[:, -1:, :]
            predictions.append(prediction)
            attentions += [attention]
            input_sequence = np.concatenate([input_sequence, predictions[-1]], axis=1)
            input_sequence = input_sequence[:, -self.source_seq_len:, :]

        return np.concatenate(predictions, axis=1), attentions


def _generate_relative_positions_matrix(length, max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length]),
                           [length, length])
    distance_mat = range_mat - tf.transpose(range_mat)
    distance_mat_clipped = tf.clip_by_value(distance_mat,
                                            -max_relative_position,
                                            max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, depth,
                                            max_relative_position, name):
    """Generates tensor of size [1 if cache else length, length, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(
                length, max_relative_position)
        vocab_size = max_relative_position*2 + 1
        # Generates embedding for each relative position of dimension depth.
        embeddings_table = tf.get_variable("embeddings",
                                           [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings
