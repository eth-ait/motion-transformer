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
import numpy as np
import cv2
import quaternion
import tensorflow as tf
import copy

from common.conversions import is_valid_rotmat, rotmat2euler, aa2rotmat
from common.conversions import get_closest_rotmat, sparse_to_full, local_rot_to_global


def pck(predictions, targets, thresh):
    """
    Percentage of correct keypoints.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a np array of shape (..., len(threshs))

    """
    dist = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))
    pck = np.mean(np.array(dist <= thresh, dtype=np.float32), axis=-1)
    return pck


def angle_diff(predictions, targets):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an np array of shape (..., n_joints)
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)


def positional(predictions, targets):
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as an np array of shape (..., n_joints)
    """
    return np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))


def euler_diff(predictions, targets):
    """
    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euler angle error an np array of shape (..., )
    """
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    ori_shape = predictions.shape[:-3]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    euler_preds = rotmat2euler(preds)  # (N, 3)
    euler_targs = rotmat2euler(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints*3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints*3])

    # l2 error on euler angles
    idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]
    euc_error = np.power(euler_targs[:, idx_to_use] - euler_preds[:, idx_to_use], 2)
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # (-1, ...)

    # reshape to original
    return np.reshape(euc_error, ori_shape)


class MetricsEngine(object):
    """
    Compute and aggregate various motion metrics. It keeps track of the metric values per frame, so that we can
    evaluate them for different sequence lengths.
    """
    def __init__(self, fk_engine, target_lengths, force_valid_rot, rep, which=None, pck_threshs=None, is_sparse=True):
        """
        Initializer.
        Args:
            fk_engine: An object of type `ForwardKinematics` used to compute positions.
            target_lengths: List of target sequence lengths that should be evaluated.
            force_valid_rot: If True, the input rotation matrices might not be valid rotations and so it will find
              the closest rotation before computing the metrics.
            rep: Which representation to use, "quat" or "rotmat".
            which: Which metrics to compute. Options are [positional, joint_angle, pck, euler], defaults to all.
            pck_threshs: List of thresholds for PCK evaluations.
            is_sparse:  If True, `n_joints` is assumed to be 15, otherwise the full SMPL skeleton is assumed. If it is
              sparse, the metrics are only calculated on the given joints.
        """
        self.which = which if which is not None else ["positional", "joint_angle", "pck", "euler"]
        self.target_lengths = target_lengths
        self.force_valid_rot = force_valid_rot
        self.fk_engine = fk_engine
        self.pck_threshs = pck_threshs if pck_threshs is not None else [0.2]
        self.is_sparse = is_sparse
        self.all_summaries_op = None
        self.n_samples = 0
        self._should_call_reset = False  # a guard to avoid stupid mistakes
        self.rep = rep
        assert self.rep in ["rotmat", "quat", "aa"]
        assert is_sparse, "at the moment we expect sparse input; if that changes, " \
                          "the metrics values may not be comparable anymore"

        # treat pck_t as a separate metric
        if "pck" in self.which:
            self.which.pop(self.which.index("pck"))
            for t in self.pck_threshs:
                self.which.append("pck_{}".format(int(t*100) if t*100 >= 1 else t*100))
        self.metrics_agg = {k: None for k in self.which}
        self.summaries = {k: {t: None for t in target_lengths} for k in self.which}

    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics_agg = {k: None for k in self.which}
        self.n_samples = 0
        self._should_call_reset = False  # now it's again safe to compute new values

    def create_summaries(self):
        """
        Create placeholders and summary ops for each metric and target length that we want to evaluate.
        """
        for m in self.summaries:
            for t in self.summaries[m]:
                assert self.summaries[m][t] is None
                # placeholder to feed metric value
                pl = tf.placeholder(tf.float32, name="{}_{}_summary_pl".format(m, t))
                # summary op to store in tensorboard
                smry = tf.summary.scalar(name="{}/until_{}".format(m, t),
                                         tensor=pl,
                                         collections=["all_metrics_summaries"])
                # store as tuple (summary, placeholder)
                self.summaries[m][t] = (smry, pl)
        # for convenience, so we don't have to list all summaries we want to request
        self.all_summaries_op = tf.summary.merge_all('all_metrics_summaries')

    def get_summary_feed_dict(self, final_metrics):
        """
        Compute the metrics for the target sequence lengths and return the feed dict that can be used in a call to
        `sess.run` to retrieve the Tensorboard summary ops.
        Args:
            final_metrics: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.

        Returns:
            The feed dictionary filled with values per summary.
        """
        feed_dict = dict()
        for m in self.summaries:
            for t in self.summaries[m]:
                pl = self.summaries[m][t][1]
                if m.startswith("pck"):
                    # does not make sense to sum up for pck
                    val = np.mean(final_metrics[m][:t])
                else:
                    val = np.sum(final_metrics[m][:t])
                feed_dict[pl] = val
        return feed_dict

    def compute_rotmat(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the chosen metrics. Predictions and targets are assumed to be in rotation matrix format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*9)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        """
        assert predictions.shape[-1] % 9 == 0, "predictions are not rotation matrices"
        assert targets.shape[-1] % 9 == 0, "targets are not rotation matrices"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 9
        n_joints = len(self.fk_engine.major_joints) if self.is_sparse else self.fk_engine.n_joints
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]
        assert n_joints*dof == predictions.shape[-1], "unexpected number of joints"

        # first reshape everything to (-1, n_joints * 9)
        pred = np.reshape(predictions, [-1, n_joints*dof]).copy()
        targ = np.reshape(targets, [-1, n_joints*dof]).copy()

        # enforce valid rotations
        if self.force_valid_rot:
            pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
            pred = get_closest_rotmat(pred_val)
            pred = np.reshape(pred, [-1, n_joints*dof])

        # check that the rotations are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid'
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]))
        assert targ_are_valid, 'target rotation matrices are not valid'

        # add potentially missing joints
        if self.is_sparse:
            pred = sparse_to_full(pred, self.fk_engine.major_joints, self.fk_engine.n_joints, rep="rotmat")
            targ = sparse_to_full(targ, self.fk_engine.major_joints, self.fk_engine.n_joints, rep="rotmat")

        # make sure we don't consider the root orientation
        assert pred.shape[-1] == self.fk_engine.n_joints*dof
        assert targ.shape[-1] == self.fk_engine.n_joints*dof
        pred[:, 0:9] = np.eye(3, 3).flatten()
        targ[:, 0:9] = np.eye(3, 3).flatten()

        metrics = dict()

        if "positional" in self.which or "pck" in self.which:
            # need to compute positions - only do this once for efficiency
            pred_pos = self.fk_engine.from_rotmat(pred)  # (-1, full_n_joints, 3)
            targ_pos = self.fk_engine.from_rotmat(targ)  # (-1, full_n_joints, 3)
        else:
            pred_pos = targ_pos = None

        select_joints = self.fk_engine.major_joints if self.is_sparse else list(range(self.fk_engine.n_joints))
        reduce_fn_np = np.mean if reduce_fn == "mean" else np.sum

        for metric in self.which:
            if metric.startswith("pck"):
                thresh = float(metric.split("_")[-1]) / 100.0
                v = pck(pred_pos[:, select_joints], targ_pos[:, select_joints], thresh=thresh)  # (-1, )
                metrics[metric] = np.reshape(v, [batch_size, seq_length])
            elif metric == "positional":
                v = positional(pred_pos[:, select_joints], targ_pos[:, select_joints])  # (-1, n_joints)
                v = np.reshape(v, [batch_size, seq_length, n_joints])
                metrics[metric] = reduce_fn_np(v, axis=-1)
            elif metric == "joint_angle":
                # compute the joint angle diff on the global rotations, not the local ones, which is a harder metric
                pred_global = local_rot_to_global(pred, self.fk_engine.parents, left_mult=self.fk_engine.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                targ_global = local_rot_to_global(targ, self.fk_engine.parents, left_mult=self.fk_engine.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
                v = np.reshape(v, [batch_size, seq_length, n_joints])
                metrics[metric] = reduce_fn_np(v, axis=-1)
            elif metric == "euler":
                # compute the euler angle error on the local rotations, which is how previous work does it
                pred_local = np.reshape(pred, [-1, self.fk_engine.n_joints, 3, 3])
                targ_local = np.reshape(targ, [-1, self.fk_engine.n_joints, 3, 3])
                v = euler_diff(pred_local[:, select_joints], targ_local[:, select_joints])  # (-1, )
                metrics[metric] = np.reshape(v, [batch_size, seq_length])
            else:
                raise ValueError("metric '{}' unknown".format(metric))

        return metrics

    def compute_quat(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the chosen metrics. Predictions and targets are assumed to be quaternions.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*4)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        """
        assert predictions.shape[-1] % 4 == 0, "predictions are not quaternions"
        assert targets.shape[-1] % 4 == 0, "targets are not quaternions"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 4
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert quaternions to rotation matrices
        pred_q = quaternion.from_float_array(np.reshape(predictions, [batch_size, seq_length, -1, dof]))
        targ_q = quaternion.from_float_array(np.reshape(targets, [batch_size, seq_length, -1, dof]))
        pred_rots = quaternion.as_rotation_matrix(pred_q)
        targ_rots = quaternion.as_rotation_matrix(targ_q)

        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute_aa(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the chosen metrics. Predictions and targets are assumed to be in angle-axis format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*3)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        """
        assert predictions.shape[-1] % 3 == 0, "predictions are not quaternions"
        assert targets.shape[-1] % 3 == 0, "targets are not quaternions"
        assert reduce_fn in ["mean", "sum"]
        assert not self._should_call_reset, "you should reset the state of this class after calling `finalize`"
        dof = 3
        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        # for simplicity we just convert angle-axis to rotation matrices
        pred_aa = np.reshape(predictions, [batch_size, seq_length, -1, dof])
        targ_aa = np.reshape(targets, [batch_size, seq_length, -1, dof])
        pred_rots = aa2rotmat(pred_aa)
        targ_rots = aa2rotmat(targ_aa)
        preds = np.reshape(pred_rots, [batch_size, seq_length, -1])
        targs = np.reshape(targ_rots, [batch_size, seq_length, -1])
        return self.compute_rotmat(preds, targs, reduce_fn)

    def compute(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the chosen metrics. Predictions and targets can be in rotation matrix or quaternion format.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        """
        if self.rep == "rotmat":
            return self.compute_rotmat(predictions, targets, reduce_fn)
        elif self.rep == "quat":
            return self.compute_quat(predictions, targets, reduce_fn)
        else:
            return self.compute_aa(predictions, targets, reduce_fn)

    def aggregate(self, new_metrics):
        """
        Aggregate the metrics.
        Args:
            new_metrics: Dictionary of new metric values to aggregate. Each entry is expected to be a numpy array
            of shape (batch_size, seq_length). For PCK values there might be more than 2 dimensions.
        """
        assert isinstance(new_metrics, dict)
        assert list(new_metrics.keys()) == list(self.metrics_agg.keys())

        # sum over the batch dimension
        for m in new_metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = np.sum(new_metrics[m], axis=0)
            else:
                self.metrics_agg[m] += np.sum(new_metrics[m], axis=0)

        # keep track of the total number of samples processed
        batch_size = new_metrics[list(new_metrics.keys())[0]].shape[0]
        self.n_samples += batch_size

    def compute_and_aggregate(self, predictions, targets, reduce_fn="mean"):
        """
        Computes the metric values and aggregates them directly.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].
        """
        new_metrics = self.compute(predictions, targets, reduce_fn)
        self.aggregate(new_metrics)

    def get_final_metrics(self):
        """
        Finalize and return the metrics - this should only be called once all the data has been processed.
        Returns:
            A dictionary of the final aggregated metrics per time step.
        """
        self._should_call_reset = True  # make sure to call `reset` before new values are computed
        assert self.n_samples > 0

        for m in self.metrics_agg:
            self.metrics_agg[m] = self.metrics_agg[m] / self.n_samples

        # return a copy of the metrics so that the class can be re-used again immediately
        return copy.deepcopy(self.metrics_agg)

    @classmethod
    def get_summary_string(cls, metric_results, at_mode=False):
        """
        Create a summary string (e.g. for printing to the console) from the given metrics for the entire sequence.
        Args:
            metric_results: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.
            at_mode: If true will report the numbers at the last frame rather then until the last frame.

        Returns:
            A summary string.
        """
        seq_length = metric_results[list(metric_results.keys())[0]].shape[0]
        s = "metrics until {}:".format(seq_length)
        for m in sorted(metric_results):
            if m.startswith("pck"):
                continue
            val = metric_results[m][seq_length - 1] if at_mode else np.sum(metric_results[m])
            s += "   {}: {:.3f}".format(m, val)

        # print pcks last
        pck_threshs = [5, 10, 15]
        for t in pck_threshs:
            m_name = "pck_{}".format(t)
            val = metric_results[m_name][seq_length - 1] if at_mode else np.mean(metric_results[m_name])
            s += "   {}: {:.3f}".format(m_name, val)
        return s

    @classmethod
    def get_summary_string_all(cls, metric_results, target_lengths, pck_thresholds, at_mode=False, report_pck=False):
        """
        Create a summary string for given lengths. Note that this yeilds results reported in the paper.
        Args:
            metric_results: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.
            target_lengths: Metrics at these time-steps are reported.
            pck_thresholds: PCK for this threshold values is reported.
            at_mode: If true will report the numbers at the last frame rather then until the last frame.
            report_pck: Whether to print all PCK values or not.

        Returns:
            A summary string.
        """
        s = ""
        for seq_length in sorted(target_lengths):
            s += "\nMetrics until {:<2}:".format(seq_length)
            for m in sorted(metric_results):
                if m.startswith("pck"):
                    continue
                val = metric_results[m][seq_length - 1] if at_mode else np.sum(metric_results[m][:seq_length])
                s += "   {}: {:.3f}".format(m, val)
        
            pck_values = []
            for threshold in sorted(pck_thresholds):
                t = threshold*100  # Convert pck value in float to a str compatible name.
                m_name = "pck_{}".format(t if t < 1 else (int(t)))
                val = metric_results[m_name][seq_length - 1] if at_mode else np.mean(metric_results[m_name][:seq_length])
                if report_pck:
                    s += "   {}: {:.3f}".format(m_name, val)
                pck_values.append(val)

            auc = cls.calculate_auc(pck_values, pck_thresholds, seq_length)
            s += "   AUC: {:.3f}".format(auc)
            
        return s
    
    @classmethod
    def calculate_auc(cls, pck_values, pck_thresholds, target_length):
        """Calculate area under a curve (AUC) metric for PCK.
        
        If the sequence length is shorter, we ignore some of the high-tolerance PCK values in order to have less
        saturated AUC.
        Args:
            pck_values (list): PCK values.
            pck_thresholds (list): PCK threshold values.
            target_length (int): determines for which time-step we calculate AUC.
        Returns:
        """
        # Due to the saturation effect, we consider a limited number of PCK thresholds in AUC calculation.
        if target_length < 6:
            n_pck = 6
        elif target_length < 12:
            n_pck = 7
        elif target_length < 18:
            n_pck = 8
        else:
            n_pck = len(pck_thresholds)
            
        norm_factor = np.diff(pck_thresholds[:n_pck]).sum()
        auc_values = []
        for i in range(n_pck - 1):
            auc = (pck_values[i] + pck_values[i + 1]) / 2 * (pck_thresholds[i + 1] - pck_thresholds[i])
            auc_values.append(auc)
        return np.array(auc_values).sum() / norm_factor
