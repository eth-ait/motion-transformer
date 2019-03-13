import numpy as np
import cv2
import quaternion
import tensorflow as tf
import copy

from smpl import SMPL_NR_JOINTS, SMPL_MAJOR_JOINTS
from smpl import smpl_sparse_to_full, smpl_rot_to_global
from smpl import SMPLForwardKinematicsNP


def rad2deg(v):
    """Convert from radians to degrees."""
    return v * 180.0 / np.pi


def deg2rad(v):
    """Convert from degrees to radians."""
    return v * np.pi / 180.0


def rx(angle):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), -np.sin(angle)],
                     [0.0, np.sin(angle), np.cos(angle)]])


def ry(angle):
    return np.array([[np.cos(angle), 0.0, np.sin(angle)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(angle), 0.0, np.cos(angle)]])


def rz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0.0],
                     [np.sin(angle), np.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def is_valid_rotmat(rotmats, thresh=1e-6):
    """
    Checks that the rotation matrices are valid, i.e. R*R' == I and det(R) == 1
    Args:
        rotmats: A np array of shape (..., 3, 3).
        thresh: Numerical threshold.

    Returns:
        True if all rotation matrices are valid, False if at least one is not valid.
    """
    # check we have a valid rotation matrix
    rotmats_t = np.transpose(rotmats, tuple(range(len(rotmats.shape[:-2]))) + (-1, -2))
    is_orthogonal = np.all(np.abs(np.matmul(rotmats, rotmats_t) - eye(3, rotmats.shape[:-2])) < thresh)
    det_is_one = np.all(np.abs(np.linalg.det(rotmats) - 1.0) < thresh)
    return is_orthogonal and det_is_one


def rotmat2euler(rotmats):
    """
    Converts rotation matrices to euler angles. This is an adaptation of Martinez et al.'s code to work with batched
    inputs. Original code can be found here:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L12

    Args:
        rotmats: An np array of shape (..., 3, 3)

    Returns:
        An np array of shape (..., 3) containing the Euler angles for each rotation matrix in `rotmats`
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3
    orig_shape = rotmats.shape[:-2]
    rs = np.reshape(rotmats, [-1, 3, 3])
    n_samples = rs.shape[0]

    # initialize to zeros
    e1 = np.zeros([n_samples])
    e2 = np.zeros([n_samples])
    e3 = np.zeros([n_samples])

    # find indices where we need to treat special cases
    is_one = rs[:, 0, 2] == 1
    is_minus_one = rs[:, 0, 2] == -1
    is_special = np.logical_or(is_one, is_minus_one)

    e1[is_special] = np.arctan2(rs[is_special, 0, 1], rs[is_special, 0, 2])
    e2[is_minus_one] = np.pi/2
    e2[is_one] = -np.pi/2

    # normal cases
    is_normal = ~np.logical_or(is_one, is_minus_one)
    # clip inputs to arcsin
    in_ = np.clip(rs[is_normal, 0, 2], -1, 1)
    e2[is_normal] = -np.arcsin(in_)
    e2_cos = np.cos(e2[is_normal])
    e1[is_normal] = np.arctan2(rs[is_normal, 1, 2]/e2_cos,
                               rs[is_normal, 2, 2]/e2_cos)
    e3[is_normal] = np.arctan2(rs[is_normal, 0, 1]/e2_cos,
                               rs[is_normal, 0, 0]/e2_cos)

    eul = np.stack([e1, e2, e3], axis=-1)
    eul = np.reshape(eul, np.concatenate([orig_shape, eul.shape[1:]]))
    return eul


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).

    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


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
    def __init__(self, smpl_model_path, target_lengths, force_valid_rot, rep, which=None, pck_threshs=None, is_sparse=True):
        """
        Initializer.
        Args:
            smpl_model_path: Path to the SMPL pickle file.
            target_lengths: List of target sequence lengths that should be evaluated.
            force_valid_rot: If True, the input rotation matrices might not be valid rotations and so it will find
              the closest rotation before computing the metrics.
            rep: Which representation to use, "quat" or "rot_mat".
            which: Which metrics to compute. Options are [positional, joint_angle, pck, euler], defaults to all.
            pck_threshs: List of thresholds for PCK evaluations.
            is_sparse:  If True, `n_joints` is assumed to be 15, otherwise the full SMPL skeleton is assumed. If it is
              sparse, the metrics are only calculated on the given joints.
        """
        self.which = which if which is not None else ["positional", "joint_angle", "pck", "euler"]
        self.target_lengths = target_lengths
        self.force_valid_rot = force_valid_rot
        self.smpl_m = SMPLForwardKinematicsNP(smpl_model_path)
        self.pck_threshs = pck_threshs if pck_threshs is not None else [0.2]
        self.is_sparse = is_sparse
        self.all_summaries_op = None
        self.n_samples = 0
        self._should_call_reset = False  # a guard to avoid stupid mistakes
        self.rep = rep
        assert self.rep in ["rot_mat", "quat"]
        assert is_sparse, "at the moment we expect sparse input; if that changes, " \
                          "the metrics values may not be comparable anymore"

        # treat pck_t as a separate metric
        if "pck" in self.which:
            self.which.pop(self.which.index("pck"))
            for t in self.pck_threshs:
                self.which.append("pck_{}".format(int(t*100)))
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
        n_joints = len(SMPL_MAJOR_JOINTS) if self.is_sparse else SMPL_NR_JOINTS
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
            pred = smpl_sparse_to_full(pred, sparse_joints_idxs=SMPL_MAJOR_JOINTS, rep="rot_mat")
            targ = smpl_sparse_to_full(targ, sparse_joints_idxs=SMPL_MAJOR_JOINTS, rep="rot_mat")

        # make sure we don't consider the root orientation
        assert pred.shape[-1] == SMPL_NR_JOINTS*dof
        assert targ.shape[-1] == SMPL_NR_JOINTS*dof
        pred[:, 0:9] = np.eye(3, 3).flatten()
        targ[:, 0:9] = np.eye(3, 3).flatten()

        metrics = dict()

        if "positional" in self.which or "pck" in self.which:
            # need to compute positions - only do this once for efficiency
            pred_pos = self.smpl_m.from_rotmat(pred)  # (-1, SMPL_NR_JOINTS, 3)
            targ_pos = self.smpl_m.from_rotmat(targ)  # (-1, SMPL_NR_JOINTS, 3)
        else:
            pred_pos = targ_pos = None

        select_joints = SMPL_MAJOR_JOINTS if self.is_sparse else list(range(SMPL_NR_JOINTS))
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
                pred_global = smpl_rot_to_global(pred, rep="rot_mat")  # (-1, SMPL_NR_JOINTS, 3, 3)
                targ_global = smpl_rot_to_global(targ, rep="rot_mat")  # (-1, SMPL_NR_JOINTS, 3, 3)
                v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
                v = np.reshape(v, [batch_size, seq_length, n_joints])
                metrics[metric] = reduce_fn_np(v, axis=-1)
            elif metric == "euler":
                # compute the euler angle error on the local rotations, which is how previous work does it
                pred_local = np.reshape(pred, [-1, SMPL_NR_JOINTS, 3, 3])
                targ_local = np.reshape(targ, [-1, SMPL_NR_JOINTS, 3, 3])
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
        if self.rep == "rot_mat":
            return self.compute_rotmat(predictions, targets, reduce_fn)
        else:
            return self.compute_quat(predictions, targets, reduce_fn)

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
    def get_summary_string(cls, final_metrics):
        """
        Create a summary string from the given metrics, e.g. for printing to the console.
        Args:
            final_metrics: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.

        Returns:
            A summary string.
        """
        seq_length = final_metrics[list(final_metrics.keys())[0]].shape[0]
        s = "metrics until {}:".format(seq_length)
        for m in sorted(final_metrics):
            if m.startswith("pck"):
                continue
            val = np.sum(final_metrics[m])
            s += "   {}: {:.3f}".format(m, val)

        # print pcks last
        pck_threshs = [5, 10, 15]
        for t in pck_threshs:
            m_name = "pck_{}".format(t)
            val = np.mean(final_metrics[m_name])
            s += "   {}: {:.3f}".format(m_name, val)

        return s

    @classmethod
    def get_summary_glogger(cls, final_metrics, is_validation=True, until=None):
        """
        Create a summary that can be written into glogger.
        Args:
            final_metrics: Dictionary of metric values, expects them to be in shape (seq_length, ) except for PCK.
            is_validation: If the given metrics are from the validation set, otherwise it's assumed they're from test.
            until: Until which time step to compute the metrics.

        Returns:
            A dictionary that can be written into glogger
        """
        pck_thresh = 10  # print this one as "pck"
        t = until if until is not None else final_metrics[list(final_metrics.keys())[0]].shape[0]
        glog_data = dict()
        for m in final_metrics:
            if m == "pck_{}".format(pck_thresh):
                # store under "pck"
                key = "val pck" if is_validation else "test pck"
                val = np.mean(final_metrics[m][:t])
                glog_data[key] = [float(val)]

            key = "val {}".format(m) if is_validation else "test {}".format(m)
            val = np.mean(final_metrics[m][:t]) if m.startswith("pck") else np.sum(final_metrics[m][:t])
            glog_data[key] = [float(val)]
        return glog_data


def _test_angle_diff():
    # test random angle diffs
    random_preds = np.random.rand(1000, 3)
    random_targs = np.random.rand(1000, 3)

    preds = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(random_preds))
    targs = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(random_targs))

    import time
    start = time.time()
    diffs1 = rad2deg(angle_diff(preds, targs))
    elapsed = time.time() - start
    print("elapsed time: {} secs, {} msecs per sample".format(elapsed, elapsed * 1000.0 / random_preds.shape[0]))

    print('max diff: ', np.amax(diffs1))
    print('min diff: ', np.amin(diffs1))

    # test some specific cases
    preds = np.stack([rx(0.0),
                      rx(np.pi / 4),
                      rx(np.pi / 2),
                      rx(np.pi),
                      rx(3 * np.pi / 2.0),
                      rx(-np.pi / 4),
                      rx(3 * np.pi / 4),
                      rx(3 * np.pi / 4),
                      rx(3 * np.pi / 4)])
    targs = np.stack([rx(0.0),
                      rx(0.0),
                      rx(0.0),
                      rx(0.0),
                      rx(0.0),
                      rx(0.0),
                      rx(np.pi / 4),
                      rx(5 * np.pi / 4),
                      rx(-3 * np.pi / 4)])

    diff = angle_diff(preds, targs)
    expected_diff = np.array([0.0,
                              np.pi / 4,
                              np.pi / 2,
                              np.pi,
                              np.pi / 2,
                              np.pi / 4,
                              np.pi / 2,
                              np.pi / 2,
                              np.pi / 2])

    print(np.linalg.norm(diff - expected_diff))
    print("actual vs expected")
    print(np.stack([diff, expected_diff], axis=1))


def _test_rotmat2euler():
    from data_utils import rotmat2euler as rotmat2euler_martinez

    # some random rotation matrices
    rs_random = np.zeros([1000, 3, 3])
    martinez_out = []
    for i in range(rs_random.shape[0]):
        xyz = np.random.rand(3) * np.pi*2 - np.pi
        rs_random[i] = np.matmul(rx(xyz[0]), np.matmul(ry(xyz[1]), rz(xyz[2])))
        martinez_out.append(rotmat2euler_martinez(rs_random[i]))
    martinez_out = np.stack(martinez_out)
    ours = rotmat2euler(rs_random)
    print("random: ", np.linalg.norm(martinez_out - ours))

    # some manual cases
    rs = np.stack([rx(np.pi/4),
                   ry(np.pi),
                   ry(np.pi/2)])
    martinez_out = []
    for r in rs:
        martinez_out.append(rotmat2euler_martinez(r))
    martinez_out = np.stack(martinez_out)
    ours = rotmat2euler(rs)
    print("manual: ", np.linalg.norm(martinez_out - ours))


if __name__ == '__main__':
    _test_rotmat2euler()
