"""
2D attention baseline implementing attention on the entire joints in all past
steps in contrast to our decoupled counterpart.
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

import numpy as np
import tensorflow as tf
from spl.model.base_model import BaseModel
from common.constants import Constants as C
from common.conversions import compute_rotation_matrix_from_ortho6d


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

        super(Transformer2d, self).__init__(config, data_pl, mode, reuse, **kwargs)

        self.window_len = config.get('transformer_window_length')
        self.use_6d_outputs = config.get('use_6d_outputs', False)

        self.abs_pos_encoding = config.get('abs_pos_encoding', True)
        self.temp_abs_pos_encoding = config.get('temp_abs_pos_encoding', False)
        self.temp_rel_pos_encoding = config.get('temp_rel_pos_encoding', False)
        self.shared_templ_kv = config.get('shared_templ_kv', False)
    
        self.window_len = min(config.get('transformer_window_length', 120), self.source_seq_len + self.target_seq_len - 1)  # It may not fit into gpu memory.
        
        # self.data_input and self.data_targets are aligned, but there might be
        # differences between them in terms of preprocessing or representation.
        self.target_input = self.data_inputs[:, :self.window_len, :]  # Ignore the rest if the input/target sequences are longer.
        self.target_real = self.data_targets[:, 1:self.window_len+1, :]

        self.pos_encoding = self.positional_encoding()
        self.look_ahead_mask = self.create_look_ahead_mask()
        
        self.max_relative_position = config.get('max_relative_position', 50)
        if self.temp_rel_pos_encoding:
            rel_emb_size = self.d_model // self.num_heads_temporal
            with tf.variable_scope("relative_embeddings", reuse=self.reuse):
                vocab_size = self.max_relative_position*2 + 1
                self.key_embedding_table = tf.get_variable("key_embeddings", [vocab_size, rel_emb_size])
                self.value_embedding_table = tf.get_variable("value_embeddings", [vocab_size, rel_emb_size])

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
        config, experiment_name = super(Transformer2d, cls).get_model_config(args, from_config)

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
        """
        Creates and optimizer, applies gradient regularization and sets the
        parameter_update operation.
        """
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
    def generate_relative_positions_matrix(length_q, length_k, max_relative_position):
        """
        Generates matrix of relative positions between inputs.
        Return a relative index matrix of shape [length_q, length_k]
        """
        range_vec_k = tf.range(length_k)
        range_vec_q = range_vec_k[-length_q:]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = tf.clip_by_value(distance_mat,
                                                -max_relative_position,
                                                max_relative_position)
        # Shift values to be >= 0. Each integer still uniquely identifies a
        # relative position difference.
        final_mat = distance_mat_clipped + max_relative_position
        return final_mat
    
    def get_relative_embeddings(self, length_q, length_k):
        """
        Generates tensor of size [1 if cache else length_q, length_k, depth].
        """
        relative_positions_matrix = self.generate_relative_positions_matrix(length_q, length_k, self.max_relative_position)
        key_emb = tf.gather(self.key_embedding_table, relative_positions_matrix)
        val_emb = tf.gather(self.value_embedding_table, relative_positions_matrix)
        return key_emb, val_emb
    
    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask, is_training=True, rel_key_emb=None, rel_val_emb=None):
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

        # attention_weights = tf.layers.dropout(attention_weights, training=is_training, rate=0.2)
        output = tf.matmul(attention_weights, v)  # (..., num_heads, attn_dim, depth)

        return output, attention_weights

    def create_look_ahead_mask(self):
        '''
        create a look ahead mask given a certain window length
        :return: the mask (window_length, window_length)
        '''
        size = self.window_len
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        # return mask  # (seq_len, seq_len)
        tiled_mask = tf.reshape(tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, self.NUM_JOINTS]), [size, size*self.NUM_JOINTS])
        return tiled_mask  # (seq_len, n_joints*seq_len)

    def get_angles(self, pos, i):
        '''
        calculate the angles givin postion and i for the positional encoding formula
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
        angle_rads = self.get_angles(np.arange(self.window_len)[:, np.newaxis], np.arange(self.d_model)[np.newaxis, :])

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
        
        if self.temp_abs_pos_encoding:
            inp_seq_len = tf.shape(x)[1]
            x += self.pos_encoding[:, :inp_seq_len]
        
        outputs = []
        attn_weights = []
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)
        
        if self.shared_templ_kv:
            with tf.variable_scope(scope + '_key', reuse=self.reuse):
                k_all = tf.layers.dense(x, self.d_model)  # (batch_size, seq_len, d_model)
            with tf.variable_scope(scope + '_value', reuse=self.reuse):
                v_all = tf.layers.dense(x, self.d_model)  # (batch_size, seq_len, d_model)

        rel_key_emb, rel_val_emb = None, None
        if self.temp_rel_pos_encoding:
            rel_key_emb, rel_val_emb = self.get_relative_embeddings(seq_len, seq_len)

        # different joints have different embedding matrices
        for joint_idx in range(self.NUM_JOINTS):
            joint_var_scope = scope + "joint_" + str(joint_idx)
            # joint_var_scope = scope + "joint_x"
            with tf.variable_scope(joint_var_scope, reuse=tf.AUTO_REUSE):
                # with tf.variable_scope('joint_' + str(joint_idx), reuse=self.reuse):

                # get the representation vector of the joint
                joint_rep = x[joint_idx]  # (batch_size, seq_len, d_model)

                # embed it to query, key and value vectors
                with tf.variable_scope('_query', reuse=self.reuse):
                    q = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                if self.shared_templ_kv:
                    v = tf.reshape(tf.transpose(v_all, [1, 2, 0, 3]), [batch_size, self.NUM_JOINTS*seq_len, self.d_model])  # (batch_size, seq_len*num_joints, d_model)
                    k = tf.reshape(tf.transpose(k_all, [1, 2, 0, 3]), [batch_size, self.NUM_JOINTS*seq_len, self.d_model])  # (batch_size, seq_len*num_joints, d_model)
                    # v = v_all[joint_idx]
                    # k = k_all[joint_idx]
                else:
                    kv_data = tf.reshape(tf.transpose(x, [1, 2, 0, 3]), [batch_size, self.NUM_JOINTS*seq_len, self.d_model])  # (batch_size, seq_len*num_joints, d_model)
                    with tf.variable_scope('_key', reuse=self.reuse):
                        # k = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                        k = tf.layers.dense(kv_data, self.d_model)  # (batch_sizm_seq_len*num_joints, d_model)
                    with tf.variable_scope('_value', reuse=self.reuse):
                        # v = tf.layers.dense(joint_rep, self.d_model)  # (batch_size, seq_len, d_model)
                        v = tf.layers.dense(kv_data, self.d_model)  # (batch_size, seq_len*num_joints, d_model)

                # split it to several attention heads
                q = self.sep_split_heads(q, batch_size, seq_len, self.num_heads_temporal)
                # (batch_size, num_heads, seq_len, depth)
                # k = self.sep_split_heads(k, batch_size, seq_len, self.num_heads_temporal)
                k = self.sep_split_heads(k, batch_size, self.NUM_JOINTS*seq_len, self.num_heads_temporal)
                # (batch_size, num_heads, seq_len, depth)
                # v = self.sep_split_heads(v, batch_size, seq_len, self.num_heads_temporal)
                v = self.sep_split_heads(v, batch_size, self.NUM_JOINTS*seq_len, self.num_heads_temporal)
                # (batch_size, num_heads, seq_len, depth)
                # calculate the updated encoding by scaled dot product attention
                scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask, self.is_training, rel_key_emb, rel_val_emb)
                # (batch_size, num_heads, seq_len, depth)
                scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
                # (batch_size, seq_len, num_heads, depth)

                # concatenate the outputs from different heads
                concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.d_model])
                # (batch_size, seq_len, d_model)

                # go through a fully connected layer
                with tf.variable_scope(scope + '_output_dense', reuse=self.reuse):
                    output = tf.expand_dims(tf.layers.dense(concat_attention, self.d_model), axis=2)
                # (batch_size, seq_len, 1, d_model)
                outputs += [output]

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
        # embed each vector to key, value and query vectors
        with tf.variable_scope(scope + '_key', reuse=self.reuse):
            k = tf.layers.dense(x, self.d_model)  # (batch_size, seq_len, num_joints, d_model)
        with tf.variable_scope(scope + '_value', reuse=self.reuse):
            v = tf.layers.dense(x, self.d_model)  # (batch_size, seq_len, num_joints, d_model)
        # Different joints have different query embedding matrices
        x = tf.transpose(x, perm=[2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)
        q_joints = []
        for joint_idx in range(self.NUM_JOINTS):
            query_var_scope = "_query_" + str(joint_idx)
            with tf.variable_scope(query_var_scope, reuse=tf.AUTO_REUSE):
                q = tf.expand_dims(tf.layers.dense(x[joint_idx], self.d_model), axis=2)  # (batch_size, seq_len, d_model)
                q_joints += [q]
        q_joints = tf.concat(q_joints, axis=2)  # (batch_size, seq_len, num_joints, d_model)
        batch_size = tf.shape(q_joints)[0]
        seq_len = tf.shape(q_joints)[1]
        # q_joints = tf.reshape(q_joints, [batch_size, seq_len, self.NUM_JOINTS, self.d_model])

        # split it to several attention heads
        q_joints = self.split_heads(q_joints, batch_size, seq_len,
                                    self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        k = self.split_heads(k, batch_size, seq_len,
                             self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        v = self.split_heads(v, batch_size, seq_len,
                             self.NUM_JOINTS, self.num_heads_spacial)
        # (batch_size, seq_len, num_heads, num_joints, depth)

        # calculate the updated encoding by scaled dot product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q_joints, k, v, mask, self.is_training)
        # (batch_size, seq_len, num_heads, num_joints, depth)
        # concatenate the outputs from different heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 1, 3, 2, 4])
        concat_attention = tf.reshape(scaled_attention, [batch_size, seq_len, self.NUM_JOINTS, self.d_model])
        # (batch_size, seq_len, num_joints, d_model)

        # go through a fully connected layer
        with tf.variable_scope(scope + '_output_dense', reuse=self.reuse):
            output = tf.layers.dense(concat_attention, self.d_model)

        attention_weights = attention_weights[:, -1, :, :, :]  # (batch_size, num_heads, num_joints, num_joints)

        return output, attention_weights

    def point_wise_feed_forward_network(self, inputs, scope):
        '''
        The feed forward block
        :param inputs: inputs (batch_size, seq_len, num_joints, d_model)
        :param scope: the name of the scope
        :return: outputs (batch_size, seq_len, num_joints, d_model)
        '''
        inputs = tf.transpose(inputs, [2, 0, 1, 3])  # (num_joints, batch_size, seq_len, d_model)
        outputs = []
        # different joints have different embedding matrices
        for idx in range(self.NUM_JOINTS):
            with tf.variable_scope(scope + '_ff1_' + str(idx), reuse=self.reuse):
                joint_outputs = tf.layers.dense(inputs[idx], self.dff, activation='relu')
            with tf.variable_scope(scope + '_ff2_' + str(idx), reuse=self.reuse):
                joint_outputs = tf.layers.dense(joint_outputs, self.d_model)
            outputs += [joint_outputs]
        outputs = tf.concat(outputs, axis=-1)  # (batch_size, seq_len, num_joints * d_model)
        outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], self.NUM_JOINTS, self.d_model])
        return outputs

    def para_transformer_layer(self, x, look_ahead_mask, scope):
        '''
        The layer with spatial and temporal blocks in parallel
        :param x: the input (batch_size, seq_len, num_joints, d_model)
        :param look_ahead_mask: the look ahead mask
        :param scope: the name of the scope
        :return: outputs (batch_size, seq_len, num_joints, d_model) and the attention blocks
        '''
        with tf.variable_scope(scope, reuse=self.reuse):
            # temporal attention
            attn1, attn_weights_block1 = self.sep_temporal_attention(x, look_ahead_mask, scope="temporal_attn")
            with tf.variable_scope("dropout_temporal", reuse=self.reuse):
                attn1 = tf.layers.dropout(attn1, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln_temporal", reuse=self.reuse):
                temporal_out = tf.contrib.layers.layer_norm(attn1 + x)

            out = temporal_out

            # feed forward
            ffn_output = self.point_wise_feed_forward_network(out, scope='feed_forward')
            with tf.variable_scope("dropout_ff", reuse=self.reuse):
                ffn_output = tf.layers.dropout(ffn_output, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln_ff", reuse=self.reuse):
                final = tf.contrib.layers.layer_norm(ffn_output + out)

            return final, attn_weights_block1, tf.ones_like(attn_weights_block1)

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
        for joint_idx in range(self.NUM_JOINTS):
            with tf.variable_scope("embedding_" + str(joint_idx), reuse=self.reuse):
                joint_rep = tf.layers.dense(inputs[joint_idx], self.d_model)  # (batch_size, seq_len, d_model)
                embed += [joint_rep]
        x = tf.concat(embed, axis=-1)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], self.NUM_JOINTS, self.d_model])
        
        # add the positional encoding
        inp_seq_len = tf.shape(inputs)[2]
        if self.abs_pos_encoding:
            x += self.pos_encoding[:, :inp_seq_len]
        
        with tf.variable_scope("input_dropout", reuse=self.reuse):
            x = tf.layers.dropout(x, training=self.is_training, rate=self.dropout_rate)

        # put into several attention layers
        # (batch_size, seq_len, num_joints, d_model)
        attention_weights_temporal = []
        attention_weights_spatial = []
        attention_weights = {}
        
        # look_ahead_mask = look_ahead_mask[:inp_seq_len, :inp_seq_len]
        for i in range(self.num_layers):
            x, block1, block2 = self.para_transformer_layer(x, look_ahead_mask, scope="transformer_layer_" + str(i))
            attention_weights_temporal += [block1]  # (batch_size, num_joints, num_heads, seq_len)
            attention_weights_spatial += [block2]  # (batch_size, num_heads, num_joints, num_joints)
        # (batch_size, seq_len, num_joints, d_model)

        attention_weights['temporal'] = tf.stack(attention_weights_temporal, axis=1)  # (batch_size, num_layers, num_joints, num_heads, seq_len)
        attention_weights['spatial'] = tf.stack(attention_weights_spatial, axis=1)  # (batch_size, num_layers, num_heads, num_joints, num_joints)

        # decode each feature to the rotation matrix space
        # different joints have different decoding matrices
        if not self.use_6d_outputs:
            # (num_joints, batch_size, seq_len, joint_size)
            x = tf.transpose(x, [2, 0, 1, 3])
            output = []
            for joint_idx in range(self.NUM_JOINTS):
                with tf.variable_scope("final_output_" + str(joint_idx), reuse=self.reuse):
                    joint_output = tf.layers.dense(x[joint_idx], self.JOINT_SIZE)
                    output += [joint_output]
    
            final_output = tf.concat(output, axis=-1)
            final_output = tf.reshape(final_output, [tf.shape(final_output)[0],
                                                     tf.shape(final_output)[1],
                                                     self.NUM_JOINTS,
                                                     self.JOINT_SIZE])
        else:
            # (num_joints, batch_size, seq_len, joint_size)
            x = tf.transpose(x, [2, 0, 1, 3])
            output = []
            for joint_idx in range(self.NUM_JOINTS):
                with tf.variable_scope("final_output_" + str(joint_idx), reuse=self.reuse):
                    joint_output = tf.layers.dense(x[joint_idx], 6)
                    output += [joint_output]
            
            n_joints = tf.shape(x)[0]
            batch_size = tf.shape(x)[1]
            seq_len = tf.shape(x)[2]

            orto6d = tf.concat(output, axis=-1)
            orto6d = tf.reshape(orto6d, [-1, 6])
            rot_mat = compute_rotation_matrix_from_ortho6d(orto6d)
            final_output = tf.reshape(rot_mat, [batch_size, seq_len, n_joints, 9])

        return final_output, attention_weights

    def build_network(self):
        shape = tf.shape(self.target_input)
        batch_siz = shape[0]
        seq_len = shape[1]
        target_input = self.target_input
        target_input = tf.reshape(target_input, [batch_siz, seq_len, self.NUM_JOINTS, self.JOINT_SIZE])
        outputs, self.attn_weights = self.transformer(target_input, self.look_ahead_mask)
        outputs = tf.reshape(outputs, [batch_siz, seq_len, self.HUMAN_SIZE])
        if self.residual_velocity:
            outputs += self.target_input

        return outputs

    def build_loss(self):
        predictions_pose = self.outputs
        targets_pose = self.target_real
        seq_len = self.window_len

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
                
            elif self.loss_type == C.LOSS_GEODESIC:
                target_angles = tf.reshape(targets_pose, [-1, seq_len, self.NUM_JOINTS, 3, 3])
                predicted_angles = tf.reshape(predictions_pose, [-1, seq_len, self.NUM_JOINTS, 3, 3])
                m = tf.matmul(target_angles, predicted_angles, transpose_b=True)
                cos = (m[:,:,:, 0,0] + m[:,:,:, 1,1] + m[:,:,:, 2,2] - 1) / 2
                cos = tf.minimum(cos, tf.ones_like(cos))
                cos = tf.maximum(cos, -1*tf.ones_like(cos))
                theta = tf.acos(cos)

                per_joint_loss = tf.reduce_sum(theta, axis=-1)
                per_joint_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                loss_ = tf.reduce_mean(per_joint_loss)
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

    def sampled_step(self, session, prediction_steps=None):
        prediction_steps = prediction_steps or self.target_seq_len
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
                data_sample[i, len_:] = np.tile(data_sample[i, len_ - 1],
                                                (max_len - len_, 1))
                
        targets = data_sample[:, self.source_seq_len:, :]
        seed_sequence = data_sample[:, :self.source_seq_len, :]
        prediction, attentions = self.sample(session=session,
                                             seed_sequence=seed_sequence,
                                             prediction_steps=prediction_steps)
        return prediction, targets, seed_sequence, data_id, attentions

    def sample(self, session, seed_sequence, prediction_steps, **kwargs):
        assert self.is_eval, "Only works in sampling mode."

        input_sequence = seed_sequence[:, -self.window_len:, :]
        num_steps = prediction_steps
        dummy_frame = np.zeros([seed_sequence.shape[0], 1, seed_sequence.shape[2]])
        predictions = []
        attentions = []

        for step in range(num_steps):
            # Insert a dummy frame since the model shifts the inputs by one step.
            model_inputs = np.concatenate([input_sequence, dummy_frame], axis=1)
            model_outputs, attention = session.run([self.outputs, self.attn_weights], feed_dict={self.data_inputs: model_inputs})
            prediction = model_outputs[:, -1:, :]
            predictions.append(prediction)
            attentions += [attention]
            input_sequence = np.concatenate([input_sequence, predictions[-1]], axis=1)
            # input_sequence = input_sequence[:, 1:, :]
            input_sequence = input_sequence[:, -self.window_len:]

        return np.concatenate(predictions, axis=1), attentions