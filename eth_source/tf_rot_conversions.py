import tensorflow as tf


def quat2rotmat(quats):
    """
    Converts quaternions to the respective rotation matrices. Assumes, that the quaternions are unit.
    Implemented after this algorithm:
    http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix
    Args:
        quats: A tensor of shape (..., 4) containing unit quaternions

    Returns:
        A tensor of shape (..., 3, 3) containing the converted rotation matrices
    """
    w, x, y, z = tf.unstack(quats, axis=-1)

    w2 = tf.multiply(w, w)
    x2 = tf.multiply(x, x)
    y2 = tf.multiply(y, y)
    z2 = tf.multiply(z, z)

    rxx = x2 - y2 - z2 + w2
    ryy = -x2 + y2 - z2 + w2
    rzz = -x2 - y2 + z2 + w2

    t1 = tf.multiply(x, y)
    t2 = tf.multiply(z, w)
    ryx = 2.0 * (t1 + t2)
    rxy = 2.0 * (t1 - t2)

    t1 = tf.multiply(x, z)
    t2 = tf.multiply(y, w)
    rzx = 2.0 * (t1 - t2)
    rxz = 2.0 * (t1 + t2)

    t1 = tf.multiply(y, z)
    t2 = tf.multiply(x, w)
    rzy = 2.0 * (t1 + t2)
    ryz = 2.0 * (t1 - t2)

    r = [[rxx, rxy, rxz],
         [ryx, ryy, ryz],
         [rzx, rzy, rzz]]

    return tf.stack([tf.stack(r[i], axis=-1) for i in range(3)], axis=-2)


def aa2rotmat(angle_axis):
    """
    Implementation of Rodrigues' formula in TensorFlow to convert angle axis to rotation matrices.
    Following this formulas: http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    Args:
        angle_axis: A tensor of shape (..., 3).

    Returns:
        A tensor of shape (..., 3, 3) containing the converted rotation matrices.
    """
    assert angle_axis.get_shape()[-1].value == 3
    ori_shape = tf.shape(angle_axis)[:-1]
    aa = tf.reshape(angle_axis, [-1, 3])  # (N, 3)
    batch_size = tf.shape(aa)[0]

    theta = tf.sqrt(tf.reduce_sum(aa*aa, axis=-1, keepdims=True))  # (N, 1)
    r = aa / theta  # (N, 3)
    is_zero = tf.equal(tf.squeeze(theta), 0.0)

    # assemble skew symmetric matrix
    zero = tf.zeros([batch_size, 1])
    row1 = tf.concat([zero, -r[:, 2:3], r[:, 1:2]], axis=-1)  # (N, 3)
    row2 = tf.concat([r[:, 2:3], zero, -r[:, 0:1]], axis=-1)  # (N, 3)
    row3 = tf.concat([-r[:, 1:2], r[:, 0:1], zero], axis=-1)  # (N, 3)
    rskew = tf.stack([row1, row2, row3], axis=1)  # (N, 3, 3)

    iden = tf.eye(3, batch_shape=[batch_size])
    r1 = tf.expand_dims(tf.sin(theta), axis=-1) * rskew
    r2 = tf.expand_dims(1.0 - tf.cos(theta), axis=-1) * tf.matmul(rskew, rskew)
    rot = iden + r1 + r2

    # insert identity matrix where theta was zero
    rot = tf.where(is_zero, tf.eye(3, batch_shape=[batch_size]), rot)

    # reshape back to input shape
    rot = tf.reshape(rot, tf.concat([ori_shape, [3, 3]], axis=0))
    return rot


def rotmat2aa(rotmats):
    """
    Implementation of Rodrigues' formula in TensorFlow to convert angle axis to rotation matrices. This function
    essentially computes log(R) = angle * axis.
    Following this formulas:
    Args:
        rotmats: A tensor of shape (..., 3, 3)

    Returns:
        A tensor of shape (..., 3) containig the converted rotation matrices
    """
    rr2 = (rotmats - tf.matrix_transpose(rotmats)) / 2.0
    a1 = -rr2[..., 1, 2]
    a2 = rr2[..., 0, 2]
    a3 = -rr2[..., 0, 1]
    a = tf.stack([a1, a2, a3], axis=-1)  # (..., 3)

    # the norm of a is equal to sin(theta), so we normalize a and then multiply with theta = arcsin(norm(a))
    norm_a = tf.sqrt(tf.reduce_sum(a * a, axis=-1, keepdims=True))  # (..., 1)
    is_zero = tf.equal(norm_a, 0.0)  # TODO(kamanuel) what to do in this case?
    aa = a / norm_a * tf.asin(norm_a)  # TODO(kamanuel) may be clip `norma` to valid values

    # insert the zero-vector where norm was zero
    aa = tf.where(tf.squeeze(is_zero, axis=-1), tf.zeros(tf.shape(rotmats)[:-1]), aa)
    raise NotImplementedError("This function does not treat edge cases well - use at your own risk")


def _test_quat2rot():
    import numpy as np
    import quaternion
    # sample random quaternions and check if numpy and tf conversion give the same results
    quats = np.random.uniform(-1, 1.0, (100, 4))
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    rots_np = quaternion.as_rotation_matrix(quaternion.from_float_array(quats))

    quats_tf = tf.constant(quats)
    rots_tf = quat2rotmat(quats_tf)

    with tf.Session() as sess:
        rots_tfe = sess.run(rots_tf)
        print(np.linalg.norm(rots_tfe - rots_np))


def _test_aa2rot():
    import numpy as np
    import cv2
    # sample random angle axis representations and check if opencv gives the same results as tf conversion
    aa = np.random.randn(100, 3).astype(np.float32)
    # add zero vector
    aa = np.concatenate([aa, [[0.0, 0.0, 0.0]]], axis=0).astype(np.float32)

    rot = aa2rotmat(tf.constant(aa))
    with tf.Session() as sess:
        rot_tf = sess.run(rot)

    mats = []
    for i, a in enumerate(aa):
        mat, _ = cv2.Rodrigues(a)
        mats.append(mat)
        print("{} TensorFlow: ".format(i), rot_tf[i])
        print("{} Rodrigues : ".format(i), mat)
    mats = np.stack(mats)

    print(np.linalg.norm(mats - rot_tf))


def _test_rotmat2aa():
    import numpy as np
    import cv2
    import quaternion

    # sample random angle axis representations and check if opencv gives the same results as tf conversion
    aa_ori = np.random.randn(100, 3).astype(np.float32)
    # add zero vector
    aa_ori = np.concatenate([aa_ori, [[0.0, 0.0, 0.0]]], axis=0).astype(np.float32)
    # add rotation of 180 degrees around x
    aa_ori = np.concatenate([aa_ori, [[np.pi, 0.0, 0.0]]], axis=0).astype(np.float32)

    rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(aa_ori)).astype(np.float32)

    aa_tf = rotmat2aa(tf.constant(rots))
    with tf.Session() as sess:
        aa_np = sess.run(aa_tf)

    aa_cv = []
    for i, rot in enumerate(rots):
        aa, _ = cv2.Rodrigues(rot)
        aa_cv.append(aa[:, 0])
        print("{} TensorFlow: ".format(i), aa_np[i])
        print("{} Rodrigues : ".format(i), aa_cv[-1])
    aa_cv = np.stack(aa_cv)

    print("dist to opencv: ", np.linalg.norm(aa_np - aa_cv))
    print("dist to ori: ", np.linalg.norm(aa_np - aa_ori))


if __name__ == '__main__':
    _test_aa2rot()
    # _test_quat2rot()
    # _test_rotmat2aa()
