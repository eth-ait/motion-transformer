import tensorflow as tf


def quaternion_dist_l2(q1, q2):
    """ Computes min { || q1 - q2 ||, || q1 + q2 || }. """
    assert q1.get_shape()[-1].value == 4
    assert q2.get_shape()[-1].value == 4

    def norm(q):
        dot = tf.reduce_sum(q*q, axis=-1)
        dot = tf.maximum(dot, 1e-12)
        return tf.sqrt(dot)

    a = norm(q1 - q2)
    b = norm(q1 + q2)
    return tf.minimum(a, b)


def quaternion_dist_cos(q1, q2):
    """ Computes arccos( | dot(q1, q2) | )"""
    assert q1.get_shape()[-1].value == 4
    assert q2.get_shape()[-1].value == 4
    dot = tf.reduce_sum(q1 * q2, axis=-1)
    dot = tf.minimum(1.0, tf.abs(dot))
    return tf.acos(dot)


def quaternion_dist_cos_approx_square(q1, q2):
    """ Computes 1 - dot(q1, q2)^2"""
    assert q1.get_shape()[-1].value == 4
    assert q2.get_shape()[-1].value == 4
    dot = tf.reduce_sum(q1 * q2, axis=-1)
    dist = 1.0 - dot*dot
    return dist


def quaternion_dist_cos_approx_abs(q1, q2):
    """ Computes 1 - | q1, q2 |"""
    assert q1.get_shape()[-1].value == 4
    assert q2.get_shape()[-1].value == 4
    dot = tf.reduce_sum(q1 * q2, axis=-1)
    dist = 1.0 - tf.abs(dot)
    return dist


def quaternion_dist_deviation_from_identity(q1, q2):
    """ Computes 2 * sqrt( 2 * (1 - | dot(q1, q2) |^2 ) which is equivalent to fro( I - R1*R2^T ) if we would use
    rotation matrices."""
    assert q1.get_shape()[-1].value == 4
    assert q2.get_shape()[-1].value == 4
    dot = tf.abs(tf.reduce_sum(q1 * q2, axis=-1))
    dot = 2.0 * (1.0 - dot*dot)
    dot = tf.maximum(dot, 1e-12)
    return 2.0 * tf.sqrt(dot)


def quaternion_loss(q1, q2, loss_fn):
    """
    Assuming q1 and q2 are valid unit quaternions, computes quaternion-based distance metric between q1 and q2.
    Args:
        q1: Tensor of shape (..., K*4)
        q2: Tensor of shape (..., K*4)
        loss_fn: string specifying which distance metric to use.

    Returns:
        Tensor of shape (..., ) containing angular error summed up over all K joints.
    """
    # assert q1.get_shape()[-1].value % 4 == 0
    # assert q2.get_shape()[-1].value % 4 == 0
    # n_joints = q1.get_shape()[-1].value // 4

    new_shape = tf.concat([tf.shape(q1)[:-1], [-1, 4]], axis=0)
    q1_r = tf.reshape(q1, new_shape)
    q2_r = tf.reshape(q2, new_shape)

    if loss_fn == 'quat_l2':
        dist = quaternion_dist_l2(q1_r, q2_r)
    elif loss_fn == 'quat_cos':
        dist = quaternion_dist_cos(q1_r, q2_r)
    elif loss_fn == 'quat_cos_approx_square':
        dist = quaternion_dist_cos_approx_square(q1_r, q2_r)
    elif loss_fn == 'quat_cos_approx_abs':
        dist = quaternion_dist_cos_approx_abs(q1_r, q2_r)
    elif loss_fn == 'quat_dev_identity':
        dist = quaternion_dist_deviation_from_identity(q1_r, q2_r)
    else:
        raise ValueError("Loss function '{}' unknown".format(loss_fn))

    return tf.reduce_sum(dist, axis=-1)


def quaternion_norm(q):
    """
    Computes how far away the given quaternions are from having unit norm. More specifically it computes the quantity
    (w^2 + x^2 + y^2 + z^2 - 1)^2
    Args:
        q: Tensor of shape (..., K*4)

    Returns:
        Tensor of shape (..., ) containing the average distance to unit norm over all K joints
    """
    last_dim = q.get_shape()[-1].value
    assert last_dim % 4 == 0
    n_joints = last_dim // 4
    q_r = tf.reshape(q, tf.concat([tf.shape(q)[:-1], [n_joints, 4]], axis=0))
    norms = tf.reduce_sum(q_r*q_r, axis=-1) - 1.0
    norms = norms * norms
    return tf.reduce_mean(norms, axis=-1)
