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
import quaternion
import cv2


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


def quat2euler(quats, epsilon=0):
    """
    PSA: This function assumes Tait-Bryan angles, i.e. consecutive rotations rotate around the rotated coordinate
    system. Use at your own peril.

    Adopted from QuaterNet; only supports order == 'xyz'. Original source code found here:
    https://github.com/facebookresearch/QuaterNet/blob/ce2d8016f749d265da9880a8dcb20a9be1a6d69c/common/quaternion.py#L53
    Args:
        quats: numpy array of shape (..., 4)

    Returns:
        A numpy array of shape (..., 3)
    """
    assert quats.shape[-1] == 4

    orig_shape = list(quats.shape)
    orig_shape[-1] = 3
    quats = np.reshape(quats, [-1, 4])

    q0 = quats[:, 0]
    q1 = quats[:, 1]
    q2 = quats[:, 2]
    q3 = quats[:, 3]

    x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
    z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

    eul = np.stack([x, y, z], axis=-1)
    return np.reshape(eul, orig_shape)


def aa2rotmat(angle_axes):
    """
    Convert angle-axis to rotation matrices using opencv's Rodrigues formula.
    Args:
        angle_axes: A np array of shape (..., 3)

    Returns:
        A np array of shape (..., 3, 3)
    """
    orig_shape = angle_axes.shape[:-1]
    aas = np.reshape(angle_axes, [-1, 3])
    rots = np.zeros([aas.shape[0], 3, 3])
    for i in range(aas.shape[0]):
        rots[i] = cv2.Rodrigues(aas[i])[0]
    return np.reshape(rots, orig_shape + (3, 3))


def rotmat2aa(rotmats):
    """
    Convert rotation matrices to angle-axis using opencv's Rodrigues formula.
    Args:
        rotmats: A np array of shape (..., 3, 3)

    Returns:
        A np array of shape (..., 3)
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3 and len(rotmats.shape) >= 3, 'invalid input dimension'
    orig_shape = rotmats.shape[:-2]
    rots = np.reshape(rotmats, [-1, 3, 3])
    aas = np.zeros([rots.shape[0], 3])
    for i in range(rots.shape[0]):
        aas[i] = np.squeeze(cv2.Rodrigues(rots[i])[0])
    return np.reshape(aas, orig_shape + (3,))


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


def sparse_to_full(joint_angles_sparse, sparse_joints_idxs, tot_nr_joints, rep="rotmat"):
    """
    Pad the given sparse joint angles with identity elements to retrieve a full skeleton with `tot_nr_joints`
    many joints.
    Args:
        joint_angles_sparse: An np array of shape (N, len(sparse_joints_idxs) * dof)
          or (N, len(sparse_joints_idxs), dof)
        sparse_joints_idxs: A list of joint indices pointing into the full skeleton given by range(0, tot_nr_joints)
        tot_nr_jonts: Total number of joints in the full skeleton.
        rep: Which representation is used, rotmat or quat

    Returns:
        The padded joint angles as an array of shape (N, tot_nr_joints*dof)
    """
    joint_idxs = sparse_joints_idxs
    assert rep in ["rotmat", "quat", "aa"]
    dof = 9 if rep == "rotmat" else 4 if rep == "quat" else 3
    n_sparse_joints = len(sparse_joints_idxs)
    angles_sparse = np.reshape(joint_angles_sparse, [-1, n_sparse_joints, dof])

    # fill in the missing indices with the identity element
    smpl_full = np.zeros(shape=[angles_sparse.shape[0], tot_nr_joints, dof])  # (N, tot_nr_joints, dof)
    if rep == "quat":
        smpl_full[..., 0] = 1.0
    elif rep == "rotmat":
        smpl_full[..., 0] = 1.0
        smpl_full[..., 4] = 1.0
        smpl_full[..., 8] = 1.0
    else:
        pass  # nothing to do for angle-axis

    smpl_full[:, joint_idxs] = angles_sparse
    smpl_full = np.reshape(smpl_full, [-1, tot_nr_joints * dof])
    return smpl_full


def local_rot_to_global(joint_angles, parents, rep="rotmat", left_mult=False):
    """
    Converts local rotations into global rotations by "unrolling" the kinematic chain.
    Args:
        joint_angles: An np array of rotation matrices of shape (N, nr_joints*dof)
        parents: A np array specifying the parent for each joint
        rep: Which representation is used for `joint_angles`
        left_mult: If True the local matrix is multiplied from the left, rather than the right

    Returns:
        The global rotations as an np array of rotation matrices in format (N, nr_joints, 3, 3)
    """
    assert rep in ["rotmat", "quat", "aa"]
    n_joints = len(parents)
    if rep == "rotmat":
        rots = np.reshape(joint_angles, [-1, n_joints, 3, 3])
    elif rep == "quat":
        rots = quaternion.as_rotation_matrix(quaternion.from_float_array(
            np.reshape(joint_angles, [-1, n_joints, 4])))
    else:
        rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(
            np.reshape(joint_angles, [-1, n_joints, 3])))

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if parents[j] < 0:
            # root rotation
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., parents[j], :, :]
            local_rot = rots[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[..., j, :, :] = np.matmul(lm, rm)
    return out
