import numpy as np
import cv2
import quaternion


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


def angle_diff(predictions, targets, use_quat=False):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`
        use_quat: If True, we use the quaternion library to extract the angle of the difference quaternion

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
    if use_quat:
        aa = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(r))
        angles = np.linalg.norm(aa, axis=-1)  # (N, )
    else:
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


if __name__ == '__main__':
    # test speed of angle_diff
    random_preds = np.random.rand(1000, 3)
    random_targs = np.random.rand(1000, 3)

    preds = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(random_preds))
    targs = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(random_targs))

    import time
    start = time.time()
    diffs1 = angle_diff(preds, targs)
    print("opencv: {} secs".format(time.time() - start))

    start = time.time()
    diffs2 = angle_diff(preds, targs, use_quat=True)
    print("quat: {} secs".format(time.time() - start))

    # TODO, why is this difference so big???
    print(np.linalg.norm(diffs1 - diffs2))
