"""
Implementation of the vanilla Transformer with 1D attention.
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


# the vanilla transformer
class Transformer1d(BaseModel):
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.num_heads = config.get('transformer_num_heads_temporal')
        self.d_model = config.get('transformer_d_model')
        self.num_layers = config.get('transformer_num_layers')
        self.dropout_rate = config.get('transformer_dropout_rate')
        self.dff = config.get('transformer_dff')
        self.lr_type = config.get('transformer_lr')
        self.warm_up_steps = config.get('transformer_warm_up_steps')  # 1000 for h3.6m, 10000 for amass
        self.depth = self.d_model // self.num_heads

        super(Transformer1d, self).__init__(config, data_pl, mode, reuse, **kwargs)

        self.window_len = config.get('transformer_window_length')  # self.source_seq_len  # attention window length

        # data
        self.prediction_targets = self.data_inputs[:, :self.window_len + 1, :]
        self.pos_encoding = self.positional_encoding()
        self.look_ahead_mask = self.create_look_ahead_mask()
        self.target_input = self.prediction_targets[:, :-1, :]
        self.target_real = self.prediction_targets[:, 1:, :]
        self.adding_zero = False

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
        config, experiment_name = super(Transformer1d, cls).get_model_config(args, from_config)

        experiment_name_format = "{}-{}-{}-{}_{}-b{}-in{}_out{}-t{}-l{}-dm{}-df{}-w{}-{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        args.model_type,
                                                        config["joint_prediction_layer"],
                                                        "h36m" if args.use_h36m else "amass",
                                                        args.data_type,
                                                        args.batch_size,
                                                        args.source_seq_len,
                                                        args.target_seq_len,
                                                        config["transformer_num_heads_temporal"],
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
        size = self.window_len
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

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

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def multihead_attention(self, v, k, q, mask, scope="multihead_attention"):
        with tf.variable_scope(scope + '_query', reuse=self.reuse):
            q = tf.layers.dense(q, self.d_model, use_bias=False)  # (batch_size, seq_len, d_model)
        with tf.variable_scope(scope + '_key', reuse=self.reuse):
            k = tf.layers.dense(k, self.d_model, use_bias=False)  # (batch_size, seq_len, d_model)
        with tf.variable_scope(scope + '_value', reuse=self.reuse):
            v = tf.layers.dense(v, self.d_model, use_bias=False)  # (batch_size, seq_len, d_model)

        batch_size = tf.shape(q)[0]

        if self.adding_zero:
            with tf.variable_scope(scope + '_adding_zero', reuse=self.reuse):
                temp = tf.ones([batch_size, 1, 1])
                temp = tf.layers.dense(temp, self.d_model, use_bias=False)  # (batch_size, 1, d_model)
            k = tf.concat([k, temp], axis=1)  # (batch_size, seq_len + 1, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        with tf.variable_scope(scope + '_output_dense', reuse=self.reuse):
            output = tf.layers.dense(concat_attention, self.d_model)

        return output, attention_weights

    def point_wise_feed_forward_network(self, inputs, scope="feedforward"):
        with tf.variable_scope(scope + '_ff1', reuse=self.reuse):
            outputs = tf.layers.dense(inputs, self.dff, activation=tf.nn.relu)
        with tf.variable_scope(scope + '_ff2', reuse=self.reuse):
            outputs = tf.layers.dense(outputs, self.d_model)
        return outputs

    def decoder_layer(self, x, look_ahead_mask, scope="decoder_layer"):
        with tf.variable_scope(scope, reuse=self.reuse):
            attn1, attn_weights_block = self.multihead_attention(x, x, x, look_ahead_mask,
                                                                 scope="attn1")  # (batch_size, target_seq_len, d_model)
            with tf.variable_scope("dropout1", reuse=self.reuse):
                attn1 = tf.layers.dropout(attn1, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln1", reuse=self.reuse):
                out1 = tf.contrib.layers.layer_norm(attn1 + x)

            ffn_output = self.point_wise_feed_forward_network(out1)  # (batch_size, target_seq_len, d_model)
            with tf.variable_scope("dropout3", reuse=self.reuse):
                ffn_output = tf.layers.dropout(ffn_output, training=self.is_training, rate=self.dropout_rate)
            with tf.variable_scope("ln3", reuse=self.reuse):
                out2 = tf.contrib.layers.layer_norm(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

            return out2, attn_weights_block

    def decoder(self, x, look_ahead_mask, scope='decoder'):
        with tf.variable_scope(scope, reuse=self.reuse):
            with tf.variable_scope("decoder_embedding", reuse=self.reuse):
                x = tf.layers.dense(x, self.d_model)
            x += self.pos_encoding
            with tf.variable_scope("decoder_dropout", reuse=self.reuse):
                x = tf.layers.dropout(x, training=self.is_training, rate=self.dropout_rate)
            attention_weights = {}
            for i in range(self.num_layers):
                x, block = self.decoder_layer(x, look_ahead_mask, scope="decoder_layer_" + str(i))
                attention_weights['decoder_layer{}_block'.format(i + 1)] = block
        return x, attention_weights

    def transformer(self, inputs, look_ahead_mask, outputs_size):
        dec_output, attention_weights = self.decoder(inputs, look_ahead_mask)
        with tf.variable_scope("final_output_dense", reuse=self.reuse):
            final_output = tf.layers.dense(dec_output, outputs_size)  # (batch_size, tar_seq_len, human_size)
        return final_output, attention_weights

    def build_network(self):
        outputs, attention_weights = self.transformer(self.target_input, self.look_ahead_mask, self.HUMAN_SIZE)
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
        assert self.is_eval, "Only works in sampling mode."
        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        targets = data_sample[:, self.source_seq_len:, :]

        # To get rid of 0 paddings.
        seq_len = batch[C.BATCH_SEQ_LEN]
        max_len = seq_len.max()
        if (seq_len != max_len).sum() != 0:
            for i in range(seq_len.shape[0]):
                len_ = seq_len[i]
                data_sample[i, len_:] = np.tile(data_sample[i, len_ - 1],
                                                (max_len - len_, 1))
        
        seed_sequence = data_sample[:, :self.source_seq_len, :]
        prediction = self.sample(session=session,
                                 seed_sequence=seed_sequence,
                                 prediction_steps=self.target_seq_len)
        return prediction, targets, seed_sequence, data_id

    def sample(self, session, seed_sequence, prediction_steps, **kwargs):
        assert self.is_eval, "Only works in sampling mode."

        input_sequence = seed_sequence[:, -self.window_len:, :]
        num_steps = prediction_steps
        dummy_frame = np.zeros([seed_sequence.shape[0], 1, seed_sequence.shape[2]])
        predictions = []

        for step in range(num_steps):
            # Insert a dummy frame since the model shifts the inputs by one step.
            model_inputs = np.concatenate([input_sequence, dummy_frame], axis=1)
            model_outputs = session.run(self.outputs, feed_dict={self.data_inputs: model_inputs})
            prediction = model_outputs[:, -1:, :]
            predictions.append(prediction)
            input_sequence = np.concatenate([input_sequence, predictions[-1]], axis=1)
            input_sequence = input_sequence[:, 1:, :]

        return np.concatenate(predictions, axis=1)



