"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import rnn_cell_extensions  # my extensions of the tf repos
import data_utils

# ETH imports
from constants import Constants as C


class BaseModel(object):
    def __init__(self, **kwargs):
        self.session = kwargs["session"]
        self.mode = kwargs["mode"]
        self.reuse = kwargs["reuse"]
        self.source_seq_len = kwargs["source_seq_len"]
        self.target_seq_len = kwargs["target_seq_len"]
        self.batch_size = kwargs["batch_size"]
        self.max_gradient_norm = kwargs["max_gradient_norm"]
        self.number_of_actions = kwargs["number_of_actions"]
        self.one_hot = kwargs["one_hot"]
        self.loss_to_use = kwargs["loss_to_use"]
        self.dtype = kwargs["dtype"]
        self.global_step = tf.train.get_global_step(graph=None)

        self.is_eval = self.mode == C.SAMPLE
        self.is_training = self.mode == C.TRAIN

        # Set by the child model class.
        self.outputs = None
        self.loss = None
        self.loss_summary = None
        self.learning_rate = None
        self.gradient_norms = None
        self.updates = None

        # Hard-coded parameters.
        self.HUMAN_SIZE = 54
        self.input_size = self.HUMAN_SIZE + self.number_of_actions if self.one_hot else self.HUMAN_SIZE

    def build_graph(self):
        self.build_network()

    def build_network(self):
        pass

    def optimization_routines(self):
        pass

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        pass

    def summary_routines(self):
        self.loss_summary = tf.summary.scalar(self.mode + '/loss', self.loss)

        # Keep track of the learning rate
        if self.is_training:
            self.learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate)
            self.gradient_norm_summary = tf.summary.scalar('gradient_norms', self.gradient_norms)

        # === variables for loss in Euler Angles -- for each action
        if self.is_eval:
            with tf.name_scope("euler_error_walking"):
                self.walking_err80 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0080")
                self.walking_err160 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0160")
                self.walking_err320 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0320")
                self.walking_err400 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0400")
                self.walking_err560 = tf.placeholder(tf.float32, name="walking_srnn_seeds_0560")
                self.walking_err1000 = tf.placeholder(tf.float32, name="walking_srnn_seeds_1000")

                self.walking_err80_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0080', self.walking_err80)
                self.walking_err160_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0160', self.walking_err160)
                self.walking_err320_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0320', self.walking_err320)
                self.walking_err400_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0400', self.walking_err400)
                self.walking_err560_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_0560', self.walking_err560)
                self.walking_err1000_summary = tf.summary.scalar('euler_error_walking/srnn_seeds_1000', self.walking_err1000)
            with tf.name_scope("euler_error_eating"):
                self.eating_err80 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0080")
                self.eating_err160 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0160")
                self.eating_err320 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0320")
                self.eating_err400 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0400")
                self.eating_err560 = tf.placeholder(tf.float32, name="eating_srnn_seeds_0560")
                self.eating_err1000 = tf.placeholder(tf.float32, name="eating_srnn_seeds_1000")

                self.eating_err80_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0080', self.eating_err80)
                self.eating_err160_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0160', self.eating_err160)
                self.eating_err320_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0320', self.eating_err320)
                self.eating_err400_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0400', self.eating_err400)
                self.eating_err560_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_0560', self.eating_err560)
                self.eating_err1000_summary = tf.summary.scalar('euler_error_eating/srnn_seeds_1000', self.eating_err1000)
            with tf.name_scope("euler_error_smoking"):
                self.smoking_err80 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0080")
                self.smoking_err160 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0160")
                self.smoking_err320 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0320")
                self.smoking_err400 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0400")
                self.smoking_err560 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_0560")
                self.smoking_err1000 = tf.placeholder(tf.float32, name="smoking_srnn_seeds_1000")

                self.smoking_err80_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0080', self.smoking_err80)
                self.smoking_err160_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0160', self.smoking_err160)
                self.smoking_err320_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0320', self.smoking_err320)
                self.smoking_err400_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0400', self.smoking_err400)
                self.smoking_err560_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_0560', self.smoking_err560)
                self.smoking_err1000_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_1000',
                                                                 self.smoking_err1000)
            with tf.name_scope("euler_error_discussion"):
                self.discussion_err80 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0080")
                self.discussion_err160 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0160")
                self.discussion_err320 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0320")
                self.discussion_err400 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0400")
                self.discussion_err560 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0560")
                self.discussion_err1000 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_1000")

                self.discussion_err80_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0080',
                                                                  self.discussion_err80)
                self.discussion_err160_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0160',
                                                                   self.discussion_err160)
                self.discussion_err320_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0320',
                                                                   self.discussion_err320)
                self.discussion_err400_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0400',
                                                                   self.discussion_err400)
                self.discussion_err560_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0560',
                                                                   self.discussion_err560)
                self.discussion_err1000_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_1000',
                                                                    self.discussion_err1000)
            with tf.name_scope("euler_error_directions"):
                self.directions_err80 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0080")
                self.directions_err160 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0160")
                self.directions_err320 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0320")
                self.directions_err400 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0400")
                self.directions_err560 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0560")
                self.directions_err1000 = tf.placeholder(tf.float32, name="directions_srnn_seeds_1000")

                self.directions_err80_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0080',
                                                                  self.directions_err80)
                self.directions_err160_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0160',
                                                                   self.directions_err160)
                self.directions_err320_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0320',
                                                                   self.directions_err320)
                self.directions_err400_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0400',
                                                                   self.directions_err400)
                self.directions_err560_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0560',
                                                                   self.directions_err560)
                self.directions_err1000_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_1000',
                                                                    self.directions_err1000)
            with tf.name_scope("euler_error_greeting"):
                self.greeting_err80 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0080")
                self.greeting_err160 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0160")
                self.greeting_err320 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0320")
                self.greeting_err400 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0400")
                self.greeting_err560 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0560")
                self.greeting_err1000 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_1000")

                self.greeting_err80_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0080', self.greeting_err80)
                self.greeting_err160_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0160',
                                                                 self.greeting_err160)
                self.greeting_err320_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0320',
                                                                 self.greeting_err320)
                self.greeting_err400_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0400',
                                                                 self.greeting_err400)
                self.greeting_err560_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0560',
                                                                 self.greeting_err560)
                self.greeting_err1000_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_1000',
                                                                  self.greeting_err1000)
            with tf.name_scope("euler_error_phoning"):
                self.phoning_err80 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0080")
                self.phoning_err160 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0160")
                self.phoning_err320 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0320")
                self.phoning_err400 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0400")
                self.phoning_err560 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_0560")
                self.phoning_err1000 = tf.placeholder(tf.float32, name="phoning_srnn_seeds_1000")

                self.phoning_err80_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0080', self.phoning_err80)
                self.phoning_err160_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0160', self.phoning_err160)
                self.phoning_err320_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0320', self.phoning_err320)
                self.phoning_err400_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0400', self.phoning_err400)
                self.phoning_err560_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_0560', self.phoning_err560)
                self.phoning_err1000_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_1000',
                                                                 self.phoning_err1000)
            with tf.name_scope("euler_error_posing"):
                self.posing_err80 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0080")
                self.posing_err160 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0160")
                self.posing_err320 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0320")
                self.posing_err400 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0400")
                self.posing_err560 = tf.placeholder(tf.float32, name="posing_srnn_seeds_0560")
                self.posing_err1000 = tf.placeholder(tf.float32, name="posing_srnn_seeds_1000")

                self.posing_err80_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0080', self.posing_err80)
                self.posing_err160_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0160', self.posing_err160)
                self.posing_err320_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0320', self.posing_err320)
                self.posing_err400_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0400', self.posing_err400)
                self.posing_err560_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_0560', self.posing_err560)
                self.posing_err1000_summary = tf.summary.scalar('euler_error_posing/srnn_seeds_1000', self.posing_err1000)
            with tf.name_scope("euler_error_purchases"):
                self.purchases_err80 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0080")
                self.purchases_err160 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0160")
                self.purchases_err320 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0320")
                self.purchases_err400 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0400")
                self.purchases_err560 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_0560")
                self.purchases_err1000 = tf.placeholder(tf.float32, name="purchases_srnn_seeds_1000")

                self.purchases_err80_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0080',
                                                                 self.purchases_err80)
                self.purchases_err160_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0160',
                                                                  self.purchases_err160)
                self.purchases_err320_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0320',
                                                                  self.purchases_err320)
                self.purchases_err400_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0400',
                                                                  self.purchases_err400)
                self.purchases_err560_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0560',
                                                                  self.purchases_err560)
                self.purchases_err1000_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_1000',
                                                                   self.purchases_err1000)
            with tf.name_scope("euler_error_sitting"):
                self.sitting_err80 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0080")
                self.sitting_err160 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0160")
                self.sitting_err320 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0320")
                self.sitting_err400 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0400")
                self.sitting_err560 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_0560")
                self.sitting_err1000 = tf.placeholder(tf.float32, name="sitting_srnn_seeds_1000")

                self.sitting_err80_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0080', self.sitting_err80)
                self.sitting_err160_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0160', self.sitting_err160)
                self.sitting_err320_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0320', self.sitting_err320)
                self.sitting_err400_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0400', self.sitting_err400)
                self.sitting_err560_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_0560', self.sitting_err560)
                self.sitting_err1000_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_1000',
                                                                 self.sitting_err1000)
            with tf.name_scope("euler_error_sittingdown"):
                self.sittingdown_err80 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0080")
                self.sittingdown_err160 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0160")
                self.sittingdown_err320 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0320")
                self.sittingdown_err400 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0400")
                self.sittingdown_err560 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0560")
                self.sittingdown_err1000 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_1000")

                self.sittingdown_err80_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0080',
                                                                   self.sittingdown_err80)
                self.sittingdown_err160_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0160',
                                                                    self.sittingdown_err160)
                self.sittingdown_err320_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0320',
                                                                    self.sittingdown_err320)
                self.sittingdown_err400_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0400',
                                                                    self.sittingdown_err400)
                self.sittingdown_err560_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0560',
                                                                    self.sittingdown_err560)
                self.sittingdown_err1000_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_1000',
                                                                     self.sittingdown_err1000)
            with tf.name_scope("euler_error_takingphoto"):
                self.takingphoto_err80 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0080")
                self.takingphoto_err160 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0160")
                self.takingphoto_err320 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0320")
                self.takingphoto_err400 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0400")
                self.takingphoto_err560 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0560")
                self.takingphoto_err1000 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_1000")

                self.takingphoto_err80_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0080',
                                                                   self.takingphoto_err80)
                self.takingphoto_err160_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0160',
                                                                    self.takingphoto_err160)
                self.takingphoto_err320_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0320',
                                                                    self.takingphoto_err320)
                self.takingphoto_err400_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0400',
                                                                    self.takingphoto_err400)
                self.takingphoto_err560_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0560',
                                                                    self.takingphoto_err560)
                self.takingphoto_err1000_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_1000',
                                                                     self.takingphoto_err1000)
            with tf.name_scope("euler_error_waiting"):
                self.waiting_err80 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0080")
                self.waiting_err160 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0160")
                self.waiting_err320 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0320")
                self.waiting_err400 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0400")
                self.waiting_err560 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_0560")
                self.waiting_err1000 = tf.placeholder(tf.float32, name="waiting_srnn_seeds_1000")

                self.waiting_err80_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0080', self.waiting_err80)
                self.waiting_err160_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0160', self.waiting_err160)
                self.waiting_err320_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0320', self.waiting_err320)
                self.waiting_err400_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0400', self.waiting_err400)
                self.waiting_err560_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_0560', self.waiting_err560)
                self.waiting_err1000_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_1000',
                                                                 self.waiting_err1000)
            with tf.name_scope("euler_error_walkingdog"):
                self.walkingdog_err80 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0080")
                self.walkingdog_err160 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0160")
                self.walkingdog_err320 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0320")
                self.walkingdog_err400 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0400")
                self.walkingdog_err560 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0560")
                self.walkingdog_err1000 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_1000")

                self.walkingdog_err80_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0080',
                                                                  self.walkingdog_err80)
                self.walkingdog_err160_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0160',
                                                                   self.walkingdog_err160)
                self.walkingdog_err320_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0320',
                                                                   self.walkingdog_err320)
                self.walkingdog_err400_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0400',
                                                                   self.walkingdog_err400)
                self.walkingdog_err560_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0560',
                                                                   self.walkingdog_err560)
                self.walkingdog_err1000_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_1000',
                                                                    self.walkingdog_err1000)
            with tf.name_scope("euler_error_walkingtogether"):
                self.walkingtogether_err80 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0080")
                self.walkingtogether_err160 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0160")
                self.walkingtogether_err320 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0320")
                self.walkingtogether_err400 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0400")
                self.walkingtogether_err560 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_0560")
                self.walkingtogether_err1000 = tf.placeholder(tf.float32, name="walkingtogether_srnn_seeds_1000")
                self.walkingtogether_err80_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0080', self.walkingtogether_err80)
                self.walkingtogether_err160_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0160', self.walkingtogether_err160)
                self.walkingtogether_err320_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0320', self.walkingtogether_err320)
                self.walkingtogether_err400_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0400', self.walkingtogether_err400)
                self.walkingtogether_err560_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_0560', self.walkingtogether_err560)
                self.walkingtogether_err1000_summary = tf.summary.scalar('euler_error_walkingtogether/srnn_seeds_1000', self.walkingtogether_err1000)

    def get_batch(self, data, actions):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: a list of sequences of size n-by-d to fit the model to.
          actions: a list of the actions we are using
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys = list(data.keys())
        chosen_keys = np.random.choice(len(all_keys), self.batch_size)

        # How many frames in total do we need?
        total_frames = self.source_seq_len + self.target_seq_len

        encoder_inputs = np.zeros((self.batch_size, self.source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

        for i in range(self.batch_size):
            the_key = all_keys[chosen_keys[i]]
            # Get the number of frames
            n, _ = data[the_key].shape
            # Sample somewhere in the middle
            idx = np.random.randint(16, n - total_frames)
            # Select the data around the sampled points
            data_sel = data[the_key][idx:idx + total_frames, :]
            # Add the data
            encoder_inputs[i, :, 0:self.input_size] = data_sel[0:self.source_seq_len - 1, :]
            decoder_inputs[i, :, 0:self.input_size] = data_sel[
                                                      self.source_seq_len - 1:self.source_seq_len +
                                                                              self.target_seq_len - 1,
                                                      :]
            decoder_outputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]
        return encoder_inputs, decoder_inputs, decoder_outputs

    def find_indices_srnn(self, data, action):
        """
        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """
        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        seed = 1234567890
        rng = np.random.RandomState(seed)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        t1 = data[(subject, action, subaction1, 'even')].shape[0]
        t2 = data[(subject, action, subaction2, 'even')].shape[0]
        prefix, suffix = 50, 100

        idx = list()
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        idx.append(rng.randint(16, t1 - prefix - suffix))
        idx.append(rng.randint(16, t2 - prefix - suffix))
        return idx

    def get_batch_srnn(self, data, action):
        """
        Get a random batch of data from the specified bucket, prepare for step.

        Args
          data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
            v=nxd matrix with a sequence of poses
          action: the action to load data from
        Returns
          The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
          the constructed batches have the proper format to call step(...) later.
        """

        actions = ["directions", "discussion", "eating", "greeting", "phoning",
                   "posing", "purchases", "sitting", "sittingdown", "smoking",
                   "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

        if action not in actions:
            raise ValueError("Unrecognized action {0}".format(action))

        frames = dict()
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8  # we always evaluate 8 seeds
        subject = 5  # we always evaluate on subject 5
        source_seq_len = self.source_seq_len
        target_seq_len = self.target_seq_len

        seeds = [(action, (i%2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, self.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, self.input_size), dtype=float)

        # Compute the number of frames needed
        total_frames = source_seq_len + target_seq_len

        # Reproducing SRNN's sequence subsequence selection as done in
        # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence, 'even')]
            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1), :]
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]
        return encoder_inputs, decoder_inputs, decoder_outputs


class Seq2SeqModel(BaseModel):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 session,
                 mode,
                 reuse,
                 architecture,
                 source_seq_len,
                 target_seq_len,
                 rnn_size,  # hidden recurrent layer size
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 summaries_dir,
                 loss_to_use,
                 number_of_actions,
                 one_hot=True,
                 residual_velocities=False,
                 dtype=tf.float32,
                 **kwargs):
        """Create the model.

        Args:
          architecture: [basic, tied] whether to tie the decoder and decoder.
          source_seq_len: lenght of the input sequence.
          target_seq_len: lenght of the target sequence.
          rnn_size: number of units in the rnn.
          num_layers: number of rnns to stack.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          summaries_dir: where to log progress for tensorboard.
          loss_to_use: [supervised, sampling_based]. Whether to use ground truth in
            each timestep to compute the loss after decoding, or to feed back the
            prediction from the previous time-step.
          number_of_actions: number of classes we have.
          one_hot: whether to use one_hot encoding during train/test (sup models).
          residual_velocities: whether to use a residual connection that models velocities.
          dtype: the data type to use to store internal variables.
        """
        super(Seq2SeqModel, self).__init__(session=session, mode=mode, reuse=reuse, source_seq_len=source_seq_len,
                                           target_seq_len=target_seq_len, batch_size=batch_size,
                                           max_gradient_norm=max_gradient_norm, number_of_actions=number_of_actions,
                                           one_hot=one_hot, loss_to_use=loss_to_use, dtype=dtype, **kwargs)
        self.residual_velocities = residual_velocities
        self.num_layers = num_layers
        self.architecture = architecture
        self.rnn_size = rnn_size
        if self.is_training:
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
            self.learning_rate_scheduler = self.learning_rate.assign(self.learning_rate*learning_rate_decay_factor)

        if self.reuse is False:
            print("One hot is ", one_hot)
            print("Input size is %d" % self.input_size)
            print('rnn_size = {0}'.format(self.rnn_size))

        # === Create the RNN that will keep the state ===
        cell = tf.contrib.rnn.GRUCell(self.rnn_size)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.rnn_size) for _ in range(self.num_layers)])

        # === Transform the inputs ===
        with tf.name_scope("inputs"):
            self.encoder_inputs = tf.placeholder(self.dtype, shape=[None, self.source_seq_len - 1, self.input_size], name="enc_in")
            self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
            self.decoder_outputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

            enc_in = tf.transpose(self.encoder_inputs, [1, 0, 2])
            dec_in = tf.transpose(self.decoder_inputs, [1, 0, 2])
            dec_out = tf.transpose(self.decoder_outputs, [1, 0, 2])

            enc_in = tf.reshape(enc_in, [-1, self.input_size])
            dec_in = tf.reshape(dec_in, [-1, self.input_size])
            dec_out = tf.reshape(dec_out, [-1, self.input_size])

            enc_in = tf.split(enc_in, self.source_seq_len - 1, axis=0)
            dec_in = tf.split(dec_in, self.target_seq_len, axis=0)
            dec_out = tf.split(dec_out, self.target_seq_len, axis=0)

        with tf.name_scope(self.mode):
            with tf.variable_scope("seq2seq", reuse=self.reuse):
                # === Add space decoder ===
                cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.input_size)

                # Finally, wrap everything in a residual layer if we want to model velocities
                if self.residual_velocities:
                    cell = rnn_cell_extensions.ResidualWrapper(cell)

                # Define the loss function
                lf = None
                if self.loss_to_use == "sampling_based":
                    def lf(prev, i):  # function for sampling_based loss
                        return prev
                elif self.loss_to_use == "supervised":
                    pass
                else:
                    raise (ValueError, "unknown loss: %s" % self.loss_to_use)

                # Build the RNN
                if self.architecture == "basic":
                    # Basic RNN does not have a loop function in its API, so copying here.
                    with vs.variable_scope("basic_rnn_seq2seq"):
                        _, enc_state = tf.contrib.rnn.static_rnn(cell, enc_in, dtype=tf.float32)  # Encoder
                        self.outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder(dec_in, enc_state, cell, loop_function=lf)  # Decoder
                elif self.architecture == "tied":
                    self.outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(enc_in, dec_in, cell, loop_function=lf)
                else:
                    raise (ValueError, "Unknown architecture: %s"%self.architecture)

            with tf.name_scope("loss_angles"):
                self.loss = tf.reduce_mean(tf.square(tf.subtract(dec_out, self.outputs)))

    def optimization_routines(self):
        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        # Update all the trainable parameters
        gradients = tf.gradients(self.loss, params)

        # Apply gradient clipping.
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.gradient_norms = norm
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def step(self, encoder_inputs, decoder_inputs, decoder_outputs):
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
        input_feed = {self.encoder_inputs : encoder_inputs,
                      self.decoder_inputs : decoder_inputs,
                      self.decoder_outputs: decoder_outputs}

        # Output feed: depends on whether we do a backward step or not.
        if self.is_training:
            # Training step
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.loss,
                           self.loss_summary,
                           self.learning_rate_summary,
                           self.gradient_norm_summary]

            outputs = self.session.run(output_feed, input_feed)
            return outputs[1], outputs[2], outputs[3], outputs[4]

        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.loss_summary,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
