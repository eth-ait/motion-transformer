import tensorflow as tf
import pickle as pkl
import numpy as np
import quaternion
import cv2

import sys

from tf_rot_conversions import quat2rotmat, aa2rotmat

try:
    sys.path.append('../external/smpl_py3')
    from smpl_webuser.serialization import load_model
except:
    print("SMPL model not available.")


SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SMPL_NR_JOINTS = 24
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
               'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder',
               'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}


def smpl_rot_to_global(joint_angles, rep="rot_mat"):
    """
    Converts local smpl rotations into global rotations by "unrolling" the kinematic chain.
    Args:
        joint_angles: An np array of rotation matrices of shape (N, SMPL_NR_JOINTS*dof)
        rep: Which representation is used for `joint_angles`

    Returns:
        The global rotations as an np array of rotation matrices in format (N, SMPL_NR_JOINTS, 3, 3)
    """
    assert rep in ["rot_mat", "quat", "aa"]
    if rep == "rot_mat":
        rots = np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3, 3])
    elif rep == "quat":
        rots = quaternion.as_rotation_matrix(quaternion.from_float_array(
            np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 4])))
    else:
        rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(
            np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3])))

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if SMPL_PARENTS[j] < 0:
            # root rotation
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., SMPL_PARENTS[j], :, :]
            local_rot = rots[..., j, :, :]
            out[..., j, :, :] = np.matmul(parent_rot, local_rot)

    return out


def smpl_sparse_to_full(joint_angles_sparse, sparse_joints_idxs=None, rep="rot_mat"):
    """
    Pad the given sparse joint angles with identity elements to retrieve a full SMPL skeleton with SMPL_NR_JOINTS
    many joints.
    Args:
        joint_angles_sparse: An np array of shape (N, len(sparse_joints_idxs) * dof)
          or (N, len(sparse_joints_idxs), dof)
        sparse_joints_idxs: A list of joint indices pointing into the full SMPL skeleton, defaults to SMPL_MAJOR_JOINTS
        rep: Which representation is used, rot_mat or quat

    Returns:
        The padded joint angles as an array of shape (N, SMPL_NR_JOINTS*dof)
    """
    joint_idxs = sparse_joints_idxs if sparse_joints_idxs is not None else SMPL_MAJOR_JOINTS
    assert rep in ["rot_mat", "quat"]
    dof = 9 if rep == "rot_mat" else 4
    n_sparse_joints = len(sparse_joints_idxs)
    angles_sparse = np.reshape(joint_angles_sparse, [-1, n_sparse_joints, dof])

    # fill in the missing indices with the identity element
    smpl_full = np.zeros(shape=[angles_sparse.shape[0], SMPL_NR_JOINTS, dof])  # (N, SMPL_NR_JOINTS, dof)
    if rep == "quat":
        smpl_full[..., 0] = 1.0
    else:
        smpl_full[..., 0] = 1.0
        smpl_full[..., 4] = 1.0
        smpl_full[..., 8] = 1.0

    smpl_full[:, joint_idxs] = angles_sparse
    smpl_full = np.reshape(smpl_full, [-1, SMPL_NR_JOINTS * dof])
    return smpl_full


class SMPLForwardKinematics(object):
    """
    Computes the joint positions using the SMPL model and code. This is slow because SMPL does not support a
    batched version, so we have to loop over every frame. However, this class can be used to get some "ground-truth"
    data.
    """
    def __init__(self, smpl_model_path):
        self.model = load_model(smpl_model_path)

    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in angle axis format.
        Args:
            joint_angles: np array of shape (N, SMPL_NR_JOINTS*3)

        Returns:
            The 3D joint positions as a an array of shape (N, SMPL_NR_JOINTS, 3)
        """
        assert joint_angles.shape[-1] == SMPL_NR_JOINTS*3
        # must loop, SMPL code does not support batches
        positions = np.zeros(shape=[joint_angles.shape[0], SMPL_NR_JOINTS, 3])
        for idx in range(joint_angles.shape[0]):
            self.model.pose[:] = joint_angles[idx]
            positions[idx] = self.model.J_transformed.r
        return positions

    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, SMPL_NR_JOINTS*3).
        """
        return self.fk(joint_angles)

    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, SMPL_NR_JOINTS*3*3).
        """
        angles = np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3, 3])
        angles_aa = np.zeros(angles.shape[:-1])
        # for now do this with opencv, i.e. we have to loop over all frames and joints, might be slow, but this
        # whole class is slow anyways
        for i in range(angles.shape[0]):
            for j in range(SMPL_NR_JOINTS):
                aa = cv2.Rodrigues(angles[i, j])[0]
                angles_aa[i, j] = aa[:, 0]  # for broadcasting
        return self.fk(np.reshape(angles_aa, [-1, SMPL_NR_JOINTS * 3]))

    def from_quat(self, joint_angles):
        """
        Get joint positions from quaternion representations in shape (N, SMPL_NR_JOINTS*4)
        """
        qs = quaternion.from_float_array(np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 4]))
        aa = quaternion.as_rotation_vector(qs)
        return self.fk(np.reshape(aa, [-1, SMPL_NR_JOINTS * 3]))

    def from_sparse(self, joint_angles_sparse, sparse_joints_idxs=None, rep="rot_mat", return_sparse=True):
        """
        Get joint positions from reduced set of SMPL joints.
        Args:
            joint_angles_sparse: np array of shape (N, len(sparse_joint_idxs) * dof))
            sparse_joints_idxs: List of indices into `SMPL_JOINTS` pointing out which SMPL joints are used in
              `pose_sparse`. If None defaults to `SMPL_MAJOR_JOINTS`.
            rep: "rot_mat" or "quat", which representation is used for the angles in `joint_angles_sparse`
            return_sparse: if True it will return only the positions of the joints given in `sparse_joint_idxs`

        Returns:
            The joint positions as an array of shape (N, len(sparse_joint_idxs), 3) if `return_sparse` is True
            otherwise (N, SMPL_NR_JOINTS, 3).
        """
        assert rep in ["rot_mat", "quat"]
        joint_idxs = sparse_joints_idxs if sparse_joints_idxs is not None else SMPL_MAJOR_JOINTS
        smpl_full = smpl_sparse_to_full(joint_angles_sparse, joint_idxs, rep)
        fk_func = self.from_quat if rep == "quat" else self.from_rotmat
        positions = fk_func(smpl_full)
        if return_sparse:
            positions = positions[:, joint_idxs]
        return positions


# noinspection PyMissingConstructor
class SMPLForwardKinematicsNP(SMPLForwardKinematics):
    """
    A numpy implementation of the SMPL forward kinematics functions, which is faster as it supports batched inputs.
    """
    def __init__(self, smpl_model_path):
        data = pkl.load(open(smpl_model_path, 'rb'), encoding='latin1')
        self.J = np.array(data['J'], dtype=np.float32)
        self.parent_ids = data['kintree_table'][0].astype(int)

    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be rotation matrices.
        Args:
            joint_angles: np array of shape (N, SMPL_NR_JOINTS*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, SMPL_NR_JOINTS, 3)
        """
        assert joint_angles.shape[-1] == SMPL_NR_JOINTS * 3 * 3
        rots = np.reshape(joint_angles, [-1, 24, 3, 3])
        n_samples = rots.shape[0]

        def with_zeros(x):
            """Make batch of (N, M, 3, 4) matrices to (4, 4) matrices by adding a [0, 0, 0, 1] row to bottom."""
            zeros = np.zeros(shape=x.shape[:-2] + (1, 4))
            zeros[:, :, 0, -1] = 1.0
            return np.concatenate([x, zeros], axis=2)

        a_global = np.zeros(shape=[n_samples, SMPL_NR_JOINTS, 4, 4])

        # insert global rotation for root
        j0 = np.tile(np.reshape(self.J[0], [1, 1, 3, 1]), [n_samples, 1, 1, 1])
        r_with_t = np.concatenate([rots[:, 0:1], j0], axis=-1)  # add translation to right of rotation matrix
        root_r = with_zeros(r_with_t)  # shape (n, 1, 4, 4)

        a_global[:, 0:1] = root_r

        for idx, pid in enumerate(self.parent_ids[1:]):
            # compute 4-by-4 matrix for this joint
            j = np.tile(np.reshape(self.J[idx+1] - self.J[pid], [1, 1, 3, 1]), [n_samples, 1, 1, 1])
            r_with_t = np.concatenate([rots[:, idx + 1:idx + 2], j], axis=-1)  # add translation to right of rotation matrix
            r = with_zeros(r_with_t)  # shape (n, 1, 4, 4)

            # multiply with its parent
            a_glob_r = np.matmul(a_global[:, pid:pid+1], r)

            # update a_global
            a_global[:, idx+1:idx+2] = a_glob_r

        # extract joint positions
        joints = a_global[:, :, :3, 3]

        # bring back in original shape
        joints = np.reshape(joints, [n_samples, SMPL_NR_JOINTS, 3])
        return joints

    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, SMPL_NR_JOINTS*3).
        """
        angles = np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3])
        angles_rotmat = np.zeros(angles.shape + (3, ))
        # TODO(kamanuel) use own function instead of opencv for improved speed
        for i in range(angles.shape[0]):
            for j in range(SMPL_NR_JOINTS):
                angles_rotmat[i, j] = cv2.Rodrigues(angles[i, j])[0]
        return self.fk(np.reshape(angles_rotmat, [-1, SMPL_NR_JOINTS * 3 * 3]))

    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, SMPL_NR_JOINTS*3*3).
        """
        return self.fk(joint_angles)

    def from_quat(self, joint_angles):
        """
        Get joint positions from quaternion representations in shape (N, SMPL_NR_JOINTS*4)
        """
        qs = quaternion.from_float_array(np.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 4]))
        rotmat = quaternion.as_rotation_matrix(qs)
        return self.fk(np.reshape(rotmat, [-1, SMPL_NR_JOINTS * 3 * 3]))


# noinspection PyMissingConstructor
class SMPLForwardKinematicsTF(SMPLForwardKinematics):
    """
    A TensorFlow implementation of the SMPL forward kinematics functions.
    """
    def __init__(self, smpl_model_path):
        data = pkl.load(open(smpl_model_path, 'rb'), encoding='latin1')
        self.J = tf.constant(data['J'], dtype=tf.float32)
        self.parent_ids = data['kintree_table'][0].astype(int)

    def from_sparse(self, joint_angles_sparse, sparse_joints_idxs=None, rep="rot_mat", return_sparse=True):
        """
        Get joint positions from reduced set of SMPL joints.
        Args:
            joint_angles_sparse: np array of shape (N, len(sparse_joint_idxs) * dof))
            sparse_joints_idxs: List of indices into `SMPL_JOINTS` pointing out which SMPL joints are used in
              `pose_sparse`. If None defaults to `SMPL_MAJOR_JOINTS`.
            rep: "rot_mat" or "quat", which representation is used for the angles in `joint_angles_sparse`
            return_sparse: if True it will return only the positions of the joints given in `sparse_joint_idxs`

        Returns:
            The joint positions as an array of shape (N, len(sparse_joint_idxs), 3) if `return_sparse` is True
            otherwise (N, SMPL_NR_JOINTS, 3).
        """
        assert rep in ["quat", "rot_mat"]
        joint_idxs = sparse_joints_idxs if sparse_joints_idxs is not None else SMPL_MAJOR_JOINTS
        dof = 9 if rep == "rot_mat" else 4
        n_sparse_joints = len(joint_idxs)
        pose_sparse_r = tf.reshape(joint_angles_sparse, [-1, n_sparse_joints, dof])
        batch_size = tf.shape(pose_sparse_r)[0]

        # Distribute the known sparse orientations to the full pose. For this we must use scatter_nd because array
        # indexing is not yet supported in TF. `indices` must have shape (batch_size, nr_joints, 2), the last two
        # dimensions are the indices into batch element and joint index
        u, v = tf.meshgrid(tf.range(0, batch_size), joint_idxs, indexing='ij')
        indices = tf.stack([u, v], axis=-1)

        # `updates` must have shape (batch_size, nr_joints, dof)
        updates = pose_sparse_r

        # pose_full will have shape (batch_size, SMPL_NR_JOINTS, dof)
        pose_full = tf.scatter_nd(indices, updates, [batch_size, SMPL_NR_JOINTS, dof])

        # the unused joints are just 0 everywhere, must make them the identity element
        unused_joints = [i for i in range(SMPL_NR_JOINTS) if i not in joint_idxs]
        u, v = tf.meshgrid(tf.range(0, batch_size), unused_joints, indexing='ij')
        indices2 = tf.stack([u, v], axis=-1)

        if rep == "quat":
            # for quats insert (1.0, 0.0, 0.0, 0.0), i.e. indices are triplets (i, j, 0)
            indices2 = tf.concat([indices2, tf.zeros([batch_size, len(unused_joints), 1], dtype=tf.int32)], axis=-1)
            updates2 = tf.ones([batch_size, len(unused_joints)])
            iden = tf.scatter_nd(indices2, updates2, [batch_size, SMPL_NR_JOINTS, dof])
            pose_full = pose_full + iden
            fk_fn = self.from_quat
        else:
            # for rotation matrices insert (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            updates2 = tf.ones([batch_size, len(unused_joints)])
            for i in [0, 4, 8]:
                idx = tf.fill([batch_size, len(unused_joints), 1], tf.constant(i, dtype=tf.int32))
                indices_idx = tf.concat([indices2, idx], axis=-1)
                iden = tf.scatter_nd(indices_idx, updates2, [batch_size, SMPL_NR_JOINTS, dof])
                pose_full = pose_full + iden
            fk_fn = self.from_rotmat

        # forward kinematics
        joints_full = fk_fn(tf.reshape(pose_full, [-1, SMPL_NR_JOINTS*dof]))

        # may be select sparse joints again
        if return_sparse:
            joints_full = tf.gather_nd(joints_full, indices)
        return joints_full

    def from_aa(self, joint_angles):
        """
        Get joint positions from angle axis representations in shape (N, SMPL_NR_JOINTS*3).
        """
        aa = tf.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3])
        return self.fk(aa2rotmat(aa))

    def from_rotmat(self, joint_angles):
        """
        Get joint positions from rotation matrix representations in shape (N, SMPL_NR_JOINTS*3*3).
        """
        joint_rotmat = tf.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 3, 3])
        return self.fk(joint_rotmat)

    def from_quat(self, joint_angles):
        """
        Perform forward kinematics given relative quaternions.
        Args:
            joint_angles: A tensor of shape (..., SMPL_NR_JOINTS * 4).

        Returns:
            A tensor of shape (..., SMPL_NR_JOINTS, 3) containing the global joint positions.
        """
        pose_rot = quat2rotmat(tf.reshape(joint_angles, [-1, SMPL_NR_JOINTS, 4]))
        return self.fk(pose_rot)

    def fk(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.

        Args:
            joint_angles: A tensor of shape (..., SMPL_NR_JOINTS, 3, 3).

        Returns:
            A tensor of shape (..., SMPL_NR_JOINTS, 3) containing the global joint positions.
        """
        assert joint_angles.get_shape()[-1].value == 3
        assert joint_angles.get_shape()[-2].value == 3
        assert joint_angles.get_shape()[-3].value == SMPL_NR_JOINTS
        ori_shape = tf.shape(joint_angles)[:-3]
        rots = tf.reshape(joint_angles, [-1, 24, 3, 3])
        batch_dim = tf.shape(rots)[0]

        def with_zeros(x):
            """Make batch of (N, M, 3, 4) matrices to (4, 4) matrices by adding a [0, 0, 0, 1] row to bottom."""
            batch_shape = tf.shape(x)[:-2]
            zeros = tf.reshape(tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32), [1, 1, 1, 4])
            zeros = tf.tile(zeros, multiples=tf.concat([batch_shape, [1, 1]], axis=0))
            return tf.concat([x, zeros], axis=2)

        a_global = tf.fill([batch_dim, 24, 4, 4], tf.constant(0.0, dtype=tf.float32))
        # insert global rotation for root
        j0 = tf.tile(tf.reshape(self.J[0], [1, 1, 3, 1]), [batch_dim, 1, 1, 1])
        r_with_t = tf.concat([rots[:, 0:1], j0], axis=-1)  # add translation to right of rotation matrix
        root_r = with_zeros(r_with_t)  # shape (n, 1, 4, 4)

        # update a_global for the whole batch
        def get_indices(joint_idx):
            b = tf.range(batch_dim, dtype=tf.int32)
            i = tf.tile([[joint_idx]], multiples=tf.stack([batch_dim, 1]))
            return tf.concat([tf.expand_dims(b, axis=-1), i], axis=1)

        def update_a_global(joint_idx, rot_, a_global_):
            indices = get_indices(joint_idx)
            scatter = tf.scatter_nd(indices=indices,
                                    updates=tf.squeeze(rot_, axis=1),
                                    shape=[batch_dim, 24, 4, 4])  # shape (n, 24, 4, 4)
            return a_global_ + tf.cast(scatter, tf.float32)

        a_global = update_a_global(0, root_r, a_global)

        for idx, pid in enumerate(self.parent_ids[1:]):
            # compute 4-by-4 matrix for this joint
            j = tf.tile(tf.reshape(self.J[idx+1] - self.J[pid], [1, 1, 3, 1]), [batch_dim, 1, 1, 1])
            r_with_t = tf.concat([rots[:, idx + 1:idx + 2], j], axis=-1)  # add translation to right of rotation matrix
            r = with_zeros(r_with_t)  # shape (n, 1, 4, 4)

            # multiply with its parent
            a_glob_r = tf.matmul(a_global[:, pid:pid+1], r)

            # update a_global
            a_global = update_a_global(idx+1, a_glob_r, a_global)

        # extract joint positions
        joints = a_global[:, :, :3, 3]

        # bring back in original shape
        joints = tf.reshape(joints, tf.concat([ori_shape, [24, 3]], axis=0))
        return joints


def _test_smpl_fk():
    # from angle axis
    m = SMPLForwardKinematics('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    random_pose = np.random.rand(100, 72) * .3
    positions = m.fk(random_pose)

    # from quaternion
    random_pose_aa = np.reshape(random_pose, [-1, SMPL_NR_JOINTS, 3])
    random_pose_quat = quaternion.as_float_array(quaternion.from_rotation_vector(random_pose_aa))
    random_pose_quat = np.reshape(random_pose_quat, [-1, SMPL_NR_JOINTS*4])
    positions_quat = m.from_quat(random_pose_quat)

    # from rotation matrices
    random_pose_rot_mat = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(random_pose_aa))
    random_pose_rot_mat = np.reshape(random_pose_rot_mat, [-1, SMPL_NR_JOINTS*9])
    positions_rotmat = m.from_rotmat(random_pose_rot_mat)

    from viz import visualize_positions
    visualize_positions([positions, positions_quat, positions_rotmat], SMPL_PARENTS, overlay=True)


def _test_np_fk():
    import time
    # get ground truth SMPL pose
    m = SMPLForwardKinematics('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    random_pose = np.random.rand(100, 72) * .3

    start = time.time()
    positions = m.fk(random_pose)
    print("SMPL: {} seconds".format(time.time() - start))

    mnp = SMPLForwardKinematicsNP('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    start = time.time()
    positions_np = mnp.from_aa(random_pose)
    print("Numpy: {} seconds".format(time.time() - start))

    from viz import visualize_positions
    visualize_positions([positions, positions_np], SMPL_PARENTS, overlay=True)


def _test_fk_sparse():
    m = load_model('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    n = 100
    random_poses = np.random.rand(n, 72) * .3
    smpl_poses = []
    for i in range(n):
        pose = np.reshape(random_poses[i], [24, 3])
        pose_full = np.zeros([24, 3])
        pose_full[SMPL_MAJOR_JOINTS] = pose[SMPL_MAJOR_JOINTS]
        m.pose[:] = np.reshape(pose_full, [-1])
        smpl_joints = m.J_transformed.r
        smpl_poses.append(smpl_joints)
    smpl_poses = np.stack(smpl_poses)

    tf_m = SMPLForwardKinematicsTF('../external/smpl_py3/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    pose_r = np.reshape(random_poses, [-1, SMPL_NR_JOINTS, 3])[:, SMPL_MAJOR_JOINTS]
    pose_r = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(pose_r))
    pose_r = np.reshape(pose_r, [n, -1])
    pose_r = tf.constant(pose_r, dtype=tf.float32)
    tf_joints = tf_m.from_sparse(pose_r, rep="rot_mat", return_sparse=False)
    with tf.Session() as sess:
        tf_jointse = sess.run(tf_joints)
        print(np.linalg.norm(smpl_poses - tf_jointse))

    from viz import visualize_positions
    visualize_positions([smpl_poses, tf_jointse], SMPL_PARENTS, overlay=True)


if __name__ == '__main__':
    _test_fk_sparse()
