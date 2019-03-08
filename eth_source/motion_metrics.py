import numpy as np
import cv2
import quaternion

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
    e2[is_normal] = -np.arcsin(rs[is_normal, 0, 2])
    e2_cos = np.cos(e2[is_normal])
    e1[is_normal] = np.arctan2(rs[is_normal, 1, 2]/e2_cos,
                               rs[is_normal, 2, 2]/e2_cos)
    e3[is_normal] = np.arctan2(rs[is_normal, 0, 1]/e2_cos,
                               rs[is_normal, 0, 0]/e2_cos)

    eul = np.stack([e1, e2, e3], axis=-1)
    eul = np.reshape(eul, np.concatenate([orig_shape, eul.shape[1:]]))
    return eul


def pck(predictions, targets, thresh):
    """
    Percentage of correct keypoints.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a np array of shape (..., )

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


def compute_metrics(predictions, targets, which=None, is_sparse=True):
    """
    Compute the chosen metrics. Predictions and targets are assumed to be in rotation matrix format.
    Args:
        predictions: An np array of shape (n, seq_length, n_joints*9)
        targets: An np array of the same shape as `predictions`
        which: Which metrics to compute. Options are [positional, joint_angle, pck, euler], all be default.
        is_sparse: If True, `n_joints` is assumed to be 15, otherwise the full SMPL skeleton is assumed. If it is
          sparse, the metrics are only calculated on the given joints.

    Returns:
        A dictionary {metric_name -> values} where the values are given per batch entry, frame and joint in an
        array of shape (n, seq_length, n_joints). If a metric cannot be reported per joint (e.g. PCK), it is returned
        as an array of shape (n, seq_length) respectively.
    """
    # TODO(kamanuel) may be have a reduce funtion to [sum, average] over the joints
    assert predictions.shape[-1] % 9 == 0, "currently we can only handle rotation matrices"
    assert targets.shape[-1] % 9 == 0, "currently we can only handle rotation matrices"
    assert is_sparse, "at the moment we expect sparse input; if that changes, the metrics values may not be comparable"
    dof = 9
    n_joints = len(SMPL_MAJOR_JOINTS) if is_sparse else SMPL_NR_JOINTS
    batch_size = predictions.shape[0]
    seq_length = predictions.shape[1]
    assert n_joints*dof == predictions.shape[-1], "unexpected number of joints"
    which = which if which is None else ["positional", "joint_angle", "pck", "euler"]

    # first reshape everything to (-1, n_joints * 9)
    pred = np.reshape(predictions, [-1, n_joints*dof]).copy()
    targ = np.reshape(targets, [-1, n_joints*dof]).copy()

    if is_sparse:
        pred = smpl_sparse_to_full(pred, sparse_joints_idxs=SMPL_MAJOR_JOINTS, rep="rot_mat")
        targ = smpl_sparse_to_full(targ, sparse_joints_idxs=SMPL_MAJOR_JOINTS, rep="rot_mat")

    # make sure we don't consider the root
    pred[:, 0:9] = np.eye(3, 3).flatten()
    targ[:, 0:9] = np.eye(3, 3).flatten()

    metrics = dict()

    if "positional" in which or "pck" in which:
        # need to compute positions - only do this once for efficiency
        # TODO(kamanuel) creating this object in every call is inefficient
        smpl_m = SMPLForwardKinematicsNP('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
        pred_pos = smpl_m.from_rotmat(pred)  # (-1, SMPL_NR_JOINTS, 3)
        targ_pos = smpl_m.from_rotmat(targ)  # (-1, SMPL_NR_JOINTS, 3)
    else:
        pred_pos = targ_pos = None

    select_joints = SMPL_MAJOR_JOINTS if is_sparse else list(range(SMPL_NR_JOINTS))

    for metric in which:
        if metric == "positional":
            v = positional(pred_pos[:, select_joints], targ_pos[:, select_joints])  # (-1, n_joints)
            metrics[metric] = np.reshape(v, [batch_size, seq_length, n_joints])
        elif metric == "pck":
            # TODO(kamanuel) how to choose threshold?
            v = pck(pred_pos[:, select_joints], targ_pos[:, select_joints], thresh=0.2)  # (-1, )
            metrics[metric] = np.reshape(v, [batch_size, seq_length])
        elif metric == "joint_angle":
            # compute the joint angle diff on the global rotations, not the local ones, which is a harder metric
            pred_global = smpl_rot_to_global(pred, rep="rot_mat")  # (-1, SMPL_NR_JOINTS, 3, 3)
            targ_global = smpl_rot_to_global(targ, rep="rot_mat")  # (-1, SMPL_NR_JOINTS, 3, 3)
            v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
            metrics[metric] = np.reshape(v, [batch_size, seq_length, n_joints])
        elif metric == "euler":
            # compute the euler angle error on the local rotations, which is how previous work does it
            pred_local = np.reshape(pred, [-1, n_joints, 3, 3])
            targ_local = np.reshape(targ, [-1, n_joints, 3, 3])
            v = euler_diff(pred_local[:, select_joints], targ_local[:, select_joints])  # (-1, )
            metrics[metric] = np.reshape(v, [batch_size, seq_length])
        else:
            raise ValueError("metric '{}' unknown".format(metric))

    return metrics


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
