"""Sequence-to-sequence model for human motion prediction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.examples import predict
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
from tf_model_utils import get_activation_fn
from tf_models import LatentLayer


class BaseModel(object):
    def __init__(self, **kwargs):
        self.session = kwargs["session"]
        self.mode = kwargs["mode"]
        self.reuse = kwargs["reuse"]
        self.source_seq_len = kwargs["source_seq_len"]
        self.target_seq_len = kwargs["target_seq_len"]
        self.batch_size = kwargs["batch_size"]
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
        self.learning_rate = None
        self.learning_rate_scheduler = None
        self.gradient_norms = None
        self.parameter_update = None
        self.summary_update = None

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
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to summary name if needed.
        self.loss_summary = tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        # Keep track of the learning rate
        if self.is_training:
            self.learning_rate_summary = tf.summary.scalar(self.mode+"/learning_rate", self.learning_rate, collections=[self.mode+"/model_summary"])
            self.gradient_norm_summary = tf.summary.scalar(self.mode+"/gradient_norms", self.gradient_norms, collections=[self.mode+"/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")

        # === variables for loss in Euler Angles -- for each action
        if self.is_eval:
            with tf.name_scope("euler_error_all_mean"):
                self.all_mean_err = tf.placeholder(tf.float32, name="all_mean_err")
                self.all_mean_err_summary = tf.summary.scalar('euler_error_all_mean_err', self.all_mean_err)

                self.all_mean_err80 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0080")
                self.all_mean_err160 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0160")
                self.all_mean_err320 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0320")
                self.all_mean_err400 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0400")
                self.all_mean_err560 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_0560")
                self.all_mean_err1000 = tf.placeholder(tf.float32, name="all_mean_srnn_seeds_1000")
                self.all_mean_err80_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0080', self.all_mean_err80)
                self.all_mean_err160_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0160', self.all_mean_err160)
                self.all_mean_err320_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0320', self.all_mean_err320)
                self.all_mean_err400_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0400', self.all_mean_err400)
                self.all_mean_err560_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_0560', self.all_mean_err560)
                self.all_mean_err1000_summary = tf.summary.scalar('euler_error_all_mean/srnn_seeds_1000', self.all_mean_err1000)

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
                self.smoking_err1000_summary = tf.summary.scalar('euler_error_smoking/srnn_seeds_1000', self.smoking_err1000)

            with tf.name_scope("euler_error_discussion"):
                self.discussion_err80 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0080")
                self.discussion_err160 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0160")
                self.discussion_err320 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0320")
                self.discussion_err400 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0400")
                self.discussion_err560 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_0560")
                self.discussion_err1000 = tf.placeholder(tf.float32, name="discussion_srnn_seeds_1000")
                self.discussion_err80_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0080', self.discussion_err80)
                self.discussion_err160_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0160', self.discussion_err160)
                self.discussion_err320_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0320', self.discussion_err320)
                self.discussion_err400_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0400', self.discussion_err400)
                self.discussion_err560_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_0560', self.discussion_err560)
                self.discussion_err1000_summary = tf.summary.scalar('euler_error_discussion/srnn_seeds_1000', self.discussion_err1000)

            with tf.name_scope("euler_error_directions"):
                self.directions_err80 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0080")
                self.directions_err160 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0160")
                self.directions_err320 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0320")
                self.directions_err400 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0400")
                self.directions_err560 = tf.placeholder(tf.float32, name="directions_srnn_seeds_0560")
                self.directions_err1000 = tf.placeholder(tf.float32, name="directions_srnn_seeds_1000")
                self.directions_err80_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0080', self.directions_err80)
                self.directions_err160_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0160', self.directions_err160)
                self.directions_err320_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0320', self.directions_err320)
                self.directions_err400_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0400', self.directions_err400)
                self.directions_err560_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_0560', self.directions_err560)
                self.directions_err1000_summary = tf.summary.scalar('euler_error_directions/srnn_seeds_1000', self.directions_err1000)

            with tf.name_scope("euler_error_greeting"):
                self.greeting_err80 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0080")
                self.greeting_err160 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0160")
                self.greeting_err320 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0320")
                self.greeting_err400 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0400")
                self.greeting_err560 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_0560")
                self.greeting_err1000 = tf.placeholder(tf.float32, name="greeting_srnn_seeds_1000")
                self.greeting_err80_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0080', self.greeting_err80)
                self.greeting_err160_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0160', self.greeting_err160)
                self.greeting_err320_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0320', self.greeting_err320)
                self.greeting_err400_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0400', self.greeting_err400)
                self.greeting_err560_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_0560', self.greeting_err560)
                self.greeting_err1000_summary = tf.summary.scalar('euler_error_greeting/srnn_seeds_1000', self.greeting_err1000)

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
                self.phoning_err1000_summary = tf.summary.scalar('euler_error_phoning/srnn_seeds_1000', self.phoning_err1000)

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
                self.purchases_err80_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0080', self.purchases_err80)
                self.purchases_err160_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0160', self.purchases_err160)
                self.purchases_err320_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0320', self.purchases_err320)
                self.purchases_err400_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0400', self.purchases_err400)
                self.purchases_err560_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_0560', self.purchases_err560)
                self.purchases_err1000_summary = tf.summary.scalar('euler_error_purchases/srnn_seeds_1000', self.purchases_err1000)

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
                self.sitting_err1000_summary = tf.summary.scalar('euler_error_sitting/srnn_seeds_1000', self.sitting_err1000)

            with tf.name_scope("euler_error_sittingdown"):
                self.sittingdown_err80 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0080")
                self.sittingdown_err160 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0160")
                self.sittingdown_err320 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0320")
                self.sittingdown_err400 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0400")
                self.sittingdown_err560 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_0560")
                self.sittingdown_err1000 = tf.placeholder(tf.float32, name="sittingdown_srnn_seeds_1000")
                self.sittingdown_err80_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0080', self.sittingdown_err80)
                self.sittingdown_err160_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0160', self.sittingdown_err160)
                self.sittingdown_err320_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0320', self.sittingdown_err320)
                self.sittingdown_err400_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0400', self.sittingdown_err400)
                self.sittingdown_err560_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_0560', self.sittingdown_err560)
                self.sittingdown_err1000_summary = tf.summary.scalar('euler_error_sittingdown/srnn_seeds_1000', self.sittingdown_err1000)

            with tf.name_scope("euler_error_takingphoto"):
                self.takingphoto_err80 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0080")
                self.takingphoto_err160 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0160")
                self.takingphoto_err320 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0320")
                self.takingphoto_err400 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0400")
                self.takingphoto_err560 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_0560")
                self.takingphoto_err1000 = tf.placeholder(tf.float32, name="takingphoto_srnn_seeds_1000")
                self.takingphoto_err80_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0080', self.takingphoto_err80)
                self.takingphoto_err160_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0160', self.takingphoto_err160)
                self.takingphoto_err320_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0320', self.takingphoto_err320)
                self.takingphoto_err400_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0400', self.takingphoto_err400)
                self.takingphoto_err560_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_0560', self.takingphoto_err560)
                self.takingphoto_err1000_summary = tf.summary.scalar('euler_error_takingphoto/srnn_seeds_1000', self.takingphoto_err1000)

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
                self.waiting_err1000_summary = tf.summary.scalar('euler_error_waiting/srnn_seeds_1000', self.waiting_err1000)

            with tf.name_scope("euler_error_walkingdog"):
                self.walkingdog_err80 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0080")
                self.walkingdog_err160 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0160")
                self.walkingdog_err320 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0320")
                self.walkingdog_err400 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0400")
                self.walkingdog_err560 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_0560")
                self.walkingdog_err1000 = tf.placeholder(tf.float32, name="walkingdog_srnn_seeds_1000")
                self.walkingdog_err80_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0080', self.walkingdog_err80)
                self.walkingdog_err160_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0160', self.walkingdog_err160)
                self.walkingdog_err320_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0320', self.walkingdog_err320)
                self.walkingdog_err400_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0400', self.walkingdog_err400)
                self.walkingdog_err560_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_0560', self.walkingdog_err560)
                self.walkingdog_err1000_summary = tf.summary.scalar('euler_error_walkingdog/srnn_seeds_1000', self.walkingdog_err1000)

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
            decoder_inputs[i, :, 0:self.input_size] = data_sel[self.source_seq_len - 1:self.source_seq_len + self.target_seq_len - 1, :]
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
                                           number_of_actions=number_of_actions,
                                           one_hot=one_hot, loss_to_use=loss_to_use, dtype=dtype, **kwargs)
        self.residual_velocities = residual_velocities
        self.num_layers = num_layers
        self.architecture = architecture
        self.rnn_size = rnn_size
        self.max_gradient_norm = max_gradient_norm
        if self.is_training:
            self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype, name="learning_rate_op")
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
        # with tf.name_scope(self.mode):
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
        if self.max_gradient_norm > 0:
            clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        else:
            self.gradient_norms = tf.linalg.global_norm(gradients)
            clipped_gradients = gradients
        self.parameter_update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

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
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."
        return self.step(encoder_inputs, decoder_inputs, decoder_outputs)[2]


class Wavenet(BaseModel):
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 source_seq_len,
                 target_seq_len,
                 batch_size,
                 loss_to_use,
                 number_of_actions,
                 one_hot=True,
                 dtype=tf.float32,
                 **kwargs):
        super(Wavenet, self).__init__(session=session, mode=mode, reuse=reuse, source_seq_len=source_seq_len,
                                      target_seq_len=target_seq_len, batch_size=batch_size, one_hot=one_hot,
                                      number_of_actions=number_of_actions, loss_to_use=loss_to_use, dtype=dtype,
                                      **kwargs)
        self.config = config

        self.input_layer_config = config.get('input_layer', None)
        self.cnn_layer_config = config.get('cnn_layer')
        self.output_layer_config = config.get('output_layer')
        self.use_gate = self.cnn_layer_config.get('use_gating', False)
        self.use_residual = self.cnn_layer_config.get('use_residual', False)
        self.use_skip = self.cnn_layer_config.get('use_skip', False)

        # If True, at every layer the input sequence is padded with zeros at the beginning such that the output length
        # becomes equal to the input length.
        self.zero_padding = self.cnn_layer_config.get('zero_padding', False)
        self.activation_fn = get_activation_fn(self.cnn_layer_config['activation_fn'])

        # Inputs to the decoder or output layer.
        self.decoder_use_enc_skip = self.config.get('decoder_use_enc_skip', False)
        self.decoder_use_enc_last = self.config.get('decoder_use_enc_last', False)
        self.decoder_use_raw_inputs = self.config.get('decoder_use_raw_inputs', False)

        self.num_encoder_blocks = self.cnn_layer_config.get('num_encoder_layers')
        self.num_decoder_blocks = self.cnn_layer_config.get('num_decoder_layers')

        # List of temporal convolution layers that are used in encoder.
        self.encoder_blocks = []
        self.encoder_blocks_no_res = []
        # List of temporal convolution layers that are used in decoder.
        self.decoder_blocks = []
        self.decoder_blocks_no_res = []

        # Specific to this code:
        self.angle_loss_type = self.config.get("angle_loss_type", C.LOSS_POSE_ALL_MEAN)
        self.residual_velocities = self.config.get("residual_velocities", False)
        self.loss_encoder_inputs = self.config.get("loss_encoder_inputs", False)
        self.summary_ops = dict()  # A container for summary ops of this model. We use "model_summary" collection name.
        self.inputs_hidden = None
        self.receptive_field_width = None
        self.temporal_block_outputs = None
        self.output_width = None
        self.outputs_tensor = None  # self.outputs is a list of frames.

        if self.is_training:
            if config.get('learning_rate_type') == 'exponential':
                self.learning_rate = tf.train.exponential_decay(config.get('learning_rate'),
                                                                global_step=self.global_step,
                                                                decay_steps=config.get('learning_rate_decay_steps'),
                                                                decay_rate=config.get('learning_rate_decay_rate'),
                                                                staircase=False)
            elif config.get('learning_rate_type') == 'piecewise':
                self.learning_rate = tf.Variable(float(config.get('learning_rate')), trainable=False, dtype=dtype, name="learning_rate_op")
                self.learning_rate_scheduler = self.learning_rate.assign(self.learning_rate*config.get('learning_rate_decay_rate'))
            elif config.get('learning_rate_type') == 'fixed':
                self.learning_rate = config.get('learning_rate')
            else:
                raise Exception("Invalid learning rate type")

        with tf.name_scope("inputs"):
            # If we have a sequence of [0,1...13,14], source_seq_len and target_seq_len with values 10 and 5:
            # [0,1,2,3,4,5,6,7,8]
            self.encoder_inputs = tf.placeholder(self.dtype, shape=[None, self.source_seq_len - 1, self.input_size], name="enc_in")
            # [9,10,11,12,13]
            if self.is_training:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_in")
                self.sequence_length = self.source_seq_len + self.target_seq_len - 1
            else:
                self.decoder_inputs = tf.placeholder(self.dtype, shape=[None, None, self.input_size], name="dec_in")
                self.sequence_length = tf.shape(self.decoder_inputs)[1]
            # [10,11,12,13,14]
            self.decoder_outputs = tf.placeholder(self.dtype, shape=[None, self.target_seq_len, self.input_size], name="dec_out")

        # Get the last frame of decoder_outputs in order to use in approximate inference.
        last_frame = self.decoder_outputs[:, -1:, :]
        self.pl_inputs = tf.concat([self.encoder_inputs, self.decoder_inputs, last_frame], axis=1)
        self.pl_targets = self.pl_inputs[:, 1:, :]
        self.pl_sequence_length = tf.ones((tf.shape(self.pl_targets)[0]), dtype=tf.int32) * self.sequence_length

        # Ignoring the action labels.
        # self.pl_targets = self.pl_inputs[:, 1:, :-self.number_of_actions]
        # self.action_label = self.pl_inputs[0:1, 0:1, -self.number_of_actions:]

    def build_network(self):
        # We always pad the input sequences such that the output sequence has the same length with input sequence.
        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.pl_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden, rate=self.input_layer_config.get("dropout_rate"), seed=self.config.seed, training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

        decoder_inputs = []
        if self.decoder_use_enc_skip:
            skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks_no_res]
            decoder_inputs.append(self.activation_fn(sum(skip_connections)))
        if self.decoder_use_enc_last:
            decoder_inputs.append(self.encoder_blocks[-1][:, 0:-1])  # Top-most convolutional layer.
        if self.decoder_use_raw_inputs:
            decoder_inputs.append(self.pl_inputs[:, 0:-1])

        # Build causal decoder blocks if we have any. Otherwise, we just use a number of 1x1 convolutions in
        # build_output_layer. Note that there are several input options.
        if self.num_decoder_blocks > 0:
            with tf.variable_scope("decoder", reuse=self.reuse):
                decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer, self.num_decoder_blocks, self.reuse, kernel_size=decoder_filter_size)
                self.temporal_block_outputs = self.decoder_blocks[-1]
        else:
            self.temporal_block_outputs = tf.concat(decoder_inputs, axis=-1)

        self.output_width = tf.shape(self.temporal_block_outputs)[1]
        self.build_output_layer()
        self.build_loss()

    def build_temporal_block(self, input_layer, num_layers, reuse, kernel_size=2):
        current_layer = input_layer
        temporal_blocks = []
        temporal_blocks_no_res = []
        for idx in range(num_layers):
            with tf.variable_scope('temporal_block_' + str(idx + 1), reuse=reuse):
                temp_block, temp_wo_res = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                                  num_filters=self.cnn_layer_config['num_filters'],
                                                                  kernel_size=kernel_size,
                                                                  dilation=self.cnn_layer_config['dilation_size'][idx],
                                                                  activation_fn=self.activation_fn,
                                                                  num_extra_conv=0,
                                                                  use_gate=self.use_gate,
                                                                  use_residual=self.use_residual,
                                                                  zero_padding=self.zero_padding)
                temporal_blocks_no_res.append(temp_wo_res)
                temporal_blocks.append(temp_block)
                current_layer = temp_block

        return temporal_blocks, temporal_blocks_no_res

    def build_output_layer(self):
        """
        Builds layers to make predictions.
        """
        out_layer_type = self.output_layer_config.get('type', None)
        with tf.variable_scope('output_layer', reuse=self.reuse):
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
                        current_layer, _ = Wavenet.temporal_block_ccn(input_layer=current_layer,
                                                                      num_filters=num_filters,
                                                                      kernel_size=kernel_size,
                                                                      dilation=1,
                                                                      activation_fn=self.activation_fn,
                                                                      num_extra_conv=0,
                                                                      use_gate=self.use_gate,
                                                                      use_residual=self.use_residual,
                                                                      zero_padding=True)
            with tf.variable_scope('out_mu', reuse=self.reuse):
                self.outputs_mu = tf.layers.conv1d(inputs=current_layer,
                                                   filters=self.input_size,
                                                   # filters=self.HUMAN_SIZE,  # Ignoring the action labels.
                                                   kernel_size=1,
                                                   padding='valid',
                                                   activation=None)
            if self.residual_velocities:
                self.outputs_mu += self.pl_inputs[:, 0:-1]
                # self.outputs_mu += self.pl_inputs[:, 0:-1, :-self.number_of_actions]  # Ignoring the action labels.
            self.outputs_tensor = self.outputs_mu
            # self.outputs_tensor = tf.concat([self.outputs_mu, tf.tile(self.action_label, (tf.shape(self.outputs_mu)[0], tf.shape(self.outputs_mu)[1], 1))], axis=-1)  # Ignoring the action labels.

            if self.angle_loss_type == C.NLL_NORMAL:
                with tf.variable_scope('out_sigma', reuse=self.reuse):
                    self.outputs_sigma = tf.layers.conv1d(inputs=current_layer,
                                                          filters=self.input_size,
                                                          kernel_size=1,
                                                          padding='valid',
                                                          activation=tf.nn.softplus)
            if self.angle_loss_type == C.NLL_NORMAL:
                with tf.variable_scope('out_sigma', reuse=self.reuse):
                    self.outputs_coefficients = tf.layers.conv1d(inputs=current_layer,
                                                                 filters=20,        # Assuming 20 components.
                                                                 kernel_size=1,
                                                                 padding='valid',
                                                                 activation=tf.nn.softmax)
            # This code repository expects the outputs to be a list of time-steps.
            # outputs_list = tf.split(self.outputs_tensor, self.sequence_length, axis=1)
            # Select only the "decoder" predictions.
            self.outputs = [tf.squeeze(out_frame, axis=1) for out_frame in tf.split(self.outputs_tensor[:, -self.target_seq_len:], self.target_seq_len, axis=1)]

    def build_loss(self):
        if self.is_eval or not self.loss_encoder_inputs:
            predictions = self.outputs_mu[:, -self.target_seq_len:, :]
            targets = self.pl_targets[:, -self.target_seq_len:, :]
            seq_len = self.target_seq_len
        else:
            predictions = self.outputs_mu
            targets = self.pl_targets
            seq_len = self.sequence_length

        with tf.name_scope("loss_angles"):
            if self.angle_loss_type == C.LOSS_POSE_ALL_MEAN:
                self.loss = tf.reduce_mean(tf.square(targets - predictions))
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_MEAN:
                per_joint_loss = tf.sqrt(tf.reduce_sum(tf.reshape(tf.square(targets - predictions), (-1, seq_len, 23, 3)), axis=-1))
                # per_joint_loss = tf.sqrt(tf.reduce_sum(tf.reshape(tf.square(targets - predictions), (-1, seq_len, 18, 3)), axis=-1)) # Ignoring the action labels.
                self.loss = tf.reduce_mean(per_joint_loss)
            elif self.angle_loss_type == C.LOSS_POSE_JOINT_SUM:
                per_joint_loss = tf.sqrt(tf.reduce_sum(tf.reshape(tf.square(targets - predictions), (-1, seq_len, 23, 3)), axis=-1))
                # per_joint_loss = tf.sqrt(tf.reduce_sum(tf.reshape(tf.square(targets - predictions), (-1, seq_len, 18, 3)), axis=-1))  # Ignoring the action labels.
                per_pose_loss = tf.reduce_sum(per_joint_loss, axis=-1)
                self.loss = tf.reduce_mean(per_pose_loss)
            elif self.angle_loss_type == C.NLL_NORMAL:
                pass
            elif self.angle_loss_type == C.NLL_GMM:
                pass
            else:
                raise Exception("Unknown angle loss.")

    def optimization_routines(self):
        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # Gradient clipping.
            gradients = tf.gradients(self.loss, params)
            if self.config.get('grad_clip_by_norm', 0) > 0:
                gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, self.config.get('grad_clip_by_norm'))
            else:
                self.gradient_norms = tf.global_norm(gradients)

            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params), global_step=self.global_step)

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
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]  # Update Op that does SGD.
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary_update,
                           self.outputs]
            outputs = self.session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Generates a synthetic sequence by feeding the prediction at t+1.
        """
        assert self.is_eval, "Only works in sampling mode."

        input_sequence = np.concatenate([encoder_inputs, decoder_inputs[:, 0:1, :]], axis=1)
        dummy_frame = np.zeros([input_sequence.shape[0], 1, input_sequence.shape[2]])
        predictions = []
        for step in range(self.target_seq_len):
            end_idx = min(self.receptive_field_width, input_sequence.shape[1])
            model_inputs = input_sequence[:, -end_idx:]
            # Insert a dummy frame since the sampling model ignores the last step.
            model_inputs = np.concatenate([model_inputs, dummy_frame], axis=1)
            model_outputs = self.session.run(self.outputs_tensor, feed_dict={self.pl_inputs: model_inputs})
            predictions.append(model_outputs[:, -1, :])
            input_sequence = np.concatenate([input_sequence, np.expand_dims(predictions[-1], axis=1)], axis=1)

        return predictions

    @staticmethod
    def receptive_field_size(filter_size, dilation_size_list):
        return (filter_size - 1)*sum(dilation_size_list) + 1

    @staticmethod
    def causal_conv_layer(input_layer, num_filters, kernel_size, dilation, zero_padding, activation_fn):
        padded_input_layer = input_layer
        # Applies padding at the start of the sequence with (kernel_size-1)*dilation zeros.
        padding_steps = (kernel_size - 1)*dilation
        if zero_padding and padding_steps > 0:
            padded_input_layer = tf.pad(input_layer, tf.constant([(0, 0,), (1, 0), (0, 0)])*padding_steps,
                                        mode='CONSTANT')
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
            filter_op = Wavenet.causal_conv_layer(input_layer=input_layer,
                                                  num_filters=num_filters,
                                                  kernel_size=kernel_size,
                                                  dilation=dilation,
                                                  zero_padding=zero_padding,
                                                  activation_fn=tf.nn.tanh)
        with tf.name_scope('gate_conv'):
            gate_op = Wavenet.causal_conv_layer(input_layer=input_layer,
                                                num_filters=num_filters,
                                                kernel_size=kernel_size,
                                                dilation=dilation,
                                                zero_padding=zero_padding,
                                                activation_fn=tf.nn.sigmoid)
        with tf.name_scope('gating'):
            gated_dilation = gate_op*filter_op

        return gated_dilation

    @staticmethod
    def temporal_block_ccn(input_layer, num_filters, kernel_size, dilation, activation_fn, num_extra_conv=0,
                           use_gate=True, use_residual=True, zero_padding=False):
        if use_gate:
            with tf.name_scope('gated_causal_layer'):
                temp_out = Wavenet.causal_gated_layer(input_layer=input_layer,
                                                      kernel_size=kernel_size,
                                                      num_filters=num_filters,
                                                      dilation=dilation,
                                                      zero_padding=zero_padding)
        else:
            with tf.name_scope('causal_layer'):
                temp_out = Wavenet.causal_conv_layer(input_layer=input_layer,
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


class STCN(Wavenet):
    def __init__(self,
                 config,
                 session,
                 mode,
                 reuse,
                 source_seq_len,
                 target_seq_len,
                 batch_size,
                 loss_to_use,
                 number_of_actions,
                 one_hot=True,
                 dtype=tf.float32,
                 **kwargs):
        super(STCN, self).__init__(config=config, session=session, mode=mode, reuse=reuse, batch_size=batch_size,
                                   target_seq_len=target_seq_len, source_seq_len=source_seq_len, one_hot=one_hot,
                                   number_of_actions=number_of_actions, loss_to_use=loss_to_use, dtype=dtype, **kwargs)
        # Add latent layer related fields.
        self.latent_layer_config = self.config.get("latent_layer")
        self.latent_layer = LatentLayer.get(config=self.latent_layer_config,
                                            layer_type=self.latent_layer_config["type"],
                                            mode=mode,
                                            reuse=reuse,
                                            global_step=self.global_step)

        self.use_future_steps_in_q = self.config.get('use_future_steps_in_q', False)
        self.bw_encoder_blocks = []
        self.bw_encoder_blocks_no_res = []

    def build_network(self):
        self.receptive_field_width = Wavenet.receptive_field_size(self.cnn_layer_config['filter_size'], self.cnn_layer_config['dilation_size'])
        self.inputs_hidden = self.pl_inputs
        if self.input_layer_config is not None and self.input_layer_config.get("dropout_rate", 0) > 0:
            with tf.variable_scope('input_dropout', reuse=self.reuse):
                self.inputs_hidden = tf.layers.dropout(self.inputs_hidden, rate=self.input_layer_config.get("dropout_rate"), seed=12345, training=self.is_training)

        with tf.variable_scope("encoder", reuse=self.reuse):
            self.encoder_blocks, self.encoder_blocks_no_res = self.build_temporal_block(self.inputs_hidden, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

        if self.use_future_steps_in_q:
            reuse_params_in_bw = True
            reversed_inputs = tf.manip.reverse(self.pl_inputs, axis=[1])
            if reuse_params_in_bw:
                with tf.variable_scope("encoder", reuse=True):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, True, self.cnn_layer_config['filter_size'])
            else:
                with tf.variable_scope("bw_encoder", reuse=self.reuse):
                    self.bw_encoder_blocks, self.bw_encoder_blocks_no_res = self.build_temporal_block(reversed_inputs, self.num_encoder_blocks, self.reuse, self.cnn_layer_config['filter_size'])

            self.bw_encoder_blocks = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks]
            self.bw_encoder_blocks_no_res = [tf.manip.reverse(bw, axis=[1])[:, 1:] for bw in self.bw_encoder_blocks_no_res]

        with tf.variable_scope("latent", reuse=self.reuse):
            p_input = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks]
            if self.latent_layer_config.get('dynamic_prior', False):
                if self.use_future_steps_in_q:
                    q_input = [tf.concat([fw_enc[:, 1:], bw_enc], axis=-1) for fw_enc, bw_enc in zip(self.encoder_blocks, self.bw_encoder_blocks)]
                else:
                    q_input = [enc_layer[:, 1:] for enc_layer in self.encoder_blocks]
            else:
                q_input = p_input
            latent_sample = self.latent_layer.build_latent_layer(q_input=q_input, p_input=p_input)

        decoder_inputs = [latent_sample]
        if self.decoder_use_enc_skip:
            skip_connections = [enc_layer[:, 0:-1] for enc_layer in self.encoder_blocks_no_res]
            decoder_inputs.append(self.activation_fn(sum(skip_connections)))
        if self.decoder_use_enc_last:
            decoder_inputs.append(self.encoder_blocks[-1][:, 0:-1])  # Top-most convolutional layer.
        if self.decoder_use_raw_inputs:
            decoder_inputs.append(self.pl_inputs[:, 0:-1])

        # Build causal decoder blocks if we have any. Otherwise, we just use a number of 1x1 convolutions in
        # build_output_layer. Note that there are several input options.
        if self.num_decoder_blocks > 0:
            with tf.variable_scope("decoder", reuse=self.reuse):
                decoder_input_layer = tf.concat(decoder_inputs, axis=-1)
                decoder_filter_size = self.cnn_layer_config.get("decoder_filter_size", self.cnn_layer_config['filter_size'])
                self.decoder_blocks, self.decoder_blocks_no_res = self.build_temporal_block(decoder_input_layer,
                                                                                            self.num_decoder_blocks,
                                                                                            self.reuse,
                                                                                            kernel_size=decoder_filter_size)
                self.temporal_block_outputs = self.decoder_blocks[-1]
        else:
            self.temporal_block_outputs = tf.concat(decoder_inputs, axis=-1)

        self.output_width = tf.shape(self.temporal_block_outputs)[1]
        self.build_output_layer()
        self.build_loss()

    def build_loss(self):
        super(STCN, self).build_loss()

        # KLD Loss.
        if self.is_training:
            loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.pl_sequence_length, dtype=tf.float32), -1)
            latent_loss_dict = self.latent_layer.build_loss(loss_mask, tf.reduce_mean)
            for loss_key, loss_op in latent_loss_dict.items():
                self.loss += loss_op
                self.summary_ops[loss_key] = tf.summary.scalar(str(loss_key), loss_op, collections=[self.mode+"/model_summary"])