"""
SPL: training and evaluation of neural networks with a structured prediction layer.
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
import tensorflow as tf

from spl.model.base_model import BaseModel


class ZeroVelocityBaseline(BaseModel):
    """Repeats the last known frame for as many frames necessary.
    
    From Martinez et al. (https://arxiv.org/abs/1705.02445).
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(ZeroVelocityBaseline, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Dummy variable
        self._dummy = tf.Variable(0.0, name="dummy_variable")
        
        # Extract the seed and target inputs.
        with tf.name_scope("inputs"):
            self.prediction_inputs = self.data_inputs[:, self.source_seq_len - 1:-1]
            self.prediction_targets = self.data_inputs[:, self.source_seq_len:]

    def build_network(self):
        # Don't do anything, just repeat the last known pose.
        last_known_pose = self.prediction_inputs[:, 0:1]
        return tf.tile(last_known_pose, [1, self.target_seq_len, 1])

    def build_loss(self):
        # Build a loss operation so that training script doesn't complain.
        d = self._dummy - self._dummy
        return tf.reduce_mean(tf.reduce_sum(d*d))
        
    def summary_routines(self):
        # Build a summary operation so that training script doesn't complain.
        tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])
        self.summary_update = tf.summary.merge_all(self.mode + "/model_summary")
    
    def optimization_routines(self):
        pass
    
    def step(self, session):
        output_feed = [self.loss,
                       self.summary_update,
                       self.outputs]
        outputs = session.run(output_feed)
        return outputs[0], outputs[1], outputs[2]
    
    def sampled_step(self, session):
        assert self.is_eval, "Only works in sampling mode."
        prediction, targets, seed_sequence, data_id = session.run([self.outputs,
                                                                   self.prediction_targets,
                                                                   self.data_inputs[:, :self.source_seq_len],
                                                                   self.data_ids])
        return prediction, targets, seed_sequence, data_id

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
        
            config['num_epochs'] = 0
        else:
            config = from_config
    
        config["experiment_id"] = str(int(time.time()))
        experiment_name_format = "{}-{}-{}_{}-b{}-in{}_out{}"
        experiment_name = experiment_name_format.format(config["experiment_id"],
                                                        config["model_type"],
                                                        "h36m" if config["use_h36m"] else "amass",
                                                        config["data_type"],
                                                        config["batch_size"],
                                                        config["source_seq_len"],
                                                        config["target_seq_len"])
        return config, experiment_name
