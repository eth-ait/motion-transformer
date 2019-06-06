# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements TensorFlow quaternion utility functions.
A quaternion is written as $$q =  xi + yj + zk + w$$, where $$i,j,k$$ forms the
three bases of the imaginary part. The functions implemented in this file
use the Hamilton convention where $$i^2 = j^2 = k^2 = ijk = -1$$. A quaternion
is stored in a 4-D vector $$[x, y, z, w]^T$$.
More details about Hamiltonian quaternions can be found on [this page.]
(https://en.wikipedia.org/wiki/Quaternion)
Note: Some of the functions expect normalized quaternions as inputs where
$$x^2 + y^2 + z^2 + w^2 = 1$$.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_tr_math_vector
import tf_tr_rotmat
import tf_tr_utils_assert
import tf_tr_utils_safe_ops
import tf_tr_utils_shape


def from_rotation_matrix(rotation_matrix, name=None):
    """Converts a rotation matrix representation to a quaternion.
    Warning:
      This function is not smooth everywhere.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      rotation_matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
        dimensions represent a rotation matrix.
      name: A name for this op that defaults to "quaternion_from_rotation_matrix".
    Returns:
      A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
      a normalized quaternion.
    Raises:
      ValueError: If the shape of `rotation_matrix` is not supported.
    """

    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

    tf_tr_utils_shape.check_static(
        tensor=rotation_matrix,
        tensor_name="rotation_matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3), (-2, 3)))

    rotation_matrix = tf_tr_rotmat.assert_rotation_matrix_normalized(
        rotation_matrix)

    trace = tf.linalg.trace(rotation_matrix)
    eps_addition = tf_tr_utils_assert.select_eps_for_addition(rotation_matrix.dtype)
    rows = tf.unstack(rotation_matrix, axis=-2)
    entries = [tf.unstack(row, axis=-1) for row in rows]

    def tr_positive():
        sq = tf.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = tf_tr_utils_safe_ops.safe_unsigned_div(entries[2][1] - entries[1][2], sq)
        qy = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][2] - entries[2][0], sq)
        qz = tf_tr_utils_safe_ops.safe_unsigned_div(entries[1][0] - entries[0][1], sq)
        return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_1():
        sq = tf.sqrt(1.0 + entries[0][0] - entries[1][1] - entries[2][2] +
                     eps_addition) * 2.  # sq = 4 * qx.
        qw = tf_tr_utils_safe_ops.safe_unsigned_div(entries[2][1] - entries[1][2], sq)
        qx = 0.25 * sq
        qy = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][1] + entries[1][0], sq)
        qz = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][2] + entries[2][0], sq)
        return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_2():
        sq = tf.sqrt(1.0 + entries[1][1] - entries[0][0] - entries[2][2] +
                     eps_addition) * 2.  # sq = 4 * qy.
        qw = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][2] - entries[2][0], sq)
        qx = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][1] + entries[1][0], sq)
        qy = 0.25 * sq
        qz = tf_tr_utils_safe_ops.safe_unsigned_div(entries[1][2] + entries[2][1], sq)
        return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_3():
        sq = tf.sqrt(1.0 + entries[2][2] - entries[0][0] - entries[1][1] +
                     eps_addition) * 2.  # sq = 4 * qz.
        qw = tf_tr_utils_safe_ops.safe_unsigned_div(entries[1][0] - entries[0][1], sq)
        qx = tf_tr_utils_safe_ops.safe_unsigned_div(entries[0][2] + entries[2][0], sq)
        qy = tf_tr_utils_safe_ops.safe_unsigned_div(entries[1][2] + entries[2][1], sq)
        qz = 0.25 * sq
        return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_idx(cond):
        cond = tf.expand_dims(cond, -1)
        cond = tf.tile(cond, [1] * (rotation_matrix.shape.ndims - 2) + [4])
        return cond

    where_2 = tf.where(
        cond_idx(entries[1][1] > entries[2][2]), cond_2(), cond_3())
    where_1 = tf.where(
        cond_idx((entries[0][0] > entries[1][1])
                 & (entries[0][0] > entries[2][2])), cond_1(), where_2)

    quat = tf.where(cond_idx(trace > 0), tr_positive(), where_1)
    return tf.convert_to_tensor(value=quat)


def normalize(quaternion, eps=1e-12, name=None):
    """Normalizes a quaternion.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      quaternion:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a quaternion.
      eps: A lower bound value for the norm that defaults to 1e-12.
      name: A name for this op that defaults to "quaternion_normalize".
    Returns:
      A N-D tensor of shape `[?, ..., ?, 1]` where the quaternion elements have
      been normalized.
    Raises:
      ValueError: If the shape of `quaternion` is not supported.
    """
    quaternion = tf.convert_to_tensor(value=quaternion)

    tf_tr_utils_shape.check_static(tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))

    return tf.math.l2_normalize(quaternion, axis=-1, epsilon=eps)


def relative_angle(quaternion1, quaternion2, name=None):
    r"""Computes the unsigned relative rotation angle between 2 unit quaternions.
    Given two normalized quanternions $$\mathbf{q}_1$$ and $$\mathbf{q}_2$$, the
    relative angle is computed as
    $$\theta = 2\arccos(\mathbf{q}_1^T\mathbf{q}_2)$$.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      quaternion1: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
      quaternion2: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
      name: A name for this op that defaults to "quaternion_relative_angle".
    Returns:
      A tensor of shape `[A1, ..., An, 1]` where the last dimension represents
      rotation angles in the range [0.0, pi].
    Raises:
      ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
    """
    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)

    tf_tr_utils_shape.check_static(
        tensor=quaternion1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    tf_tr_utils_shape.check_static(
        tensor=quaternion2, tensor_name="quaternion2", has_dim_equals=(-1, 4))
    quaternion1 = tf_tr_utils_assert.assert_normalized(quaternion1)
    quaternion2 = tf_tr_utils_assert.assert_normalized(quaternion2)

    dot_product = tf_tr_math_vector.dot(quaternion1, quaternion2, keepdims=False)
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 4.0 * tf_tr_utils_assert.select_eps_for_addition(dot_product.dtype)
    dot_product = tf_tr_utils_safe_ops.safe_shrink(dot_product,
                                                   -1.0,
                                                   1.0,
                                                   False,
                                                   eps=eps_dot_prod)
    return 2.0 * tf.acos(tf.abs(dot_product))


def from_axis_angle(axis_angle, name=None):
    """Converts an axis-angle representation to a quaternion.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a normalized axis.
      angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
        represents an angle.
      name: A name for this op that defaults to "quaternion_from_axis_angle".
    Returns:
      A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
      a normalized quaternion.
    Raises:
      ValueError: If the shape of `axis` or `angle` is not supported.
    """
    axis_angle = tf.convert_to_tensor(value=axis_angle)

    angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    # NOTE it's not a good idea to use tf.linalg.l2_normalize here!!!
    axis = axis_angle / angle

    # when the angle is very small, we manually set it to zero and choose a random normalized axis, so that the
    # resulting rotation matrix will be the identity
    angle = tf.where(tf.less_equal(angle, 1e-12), tf.zeros_like(angle), angle)
    random_axis = tf.concat([tf.ones_like(angle), tf.zeros_like(angle), tf.zeros_like(angle)], axis=-1)
    axis = tf.where(tf.less_equal(tf.concat([angle] * 3, axis=-1), 1e-12), random_axis, axis)

    # angle = tf.norm(axis_angle, axis=-1, keepdims=True)
    # axis = tf.linalg.l2_normalize(axis_angle, axis=-1)

    tf_tr_utils_shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    tf_tr_utils_shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    tf_tr_utils_shape.compare_batch_dimensions(
        tensors=(axis, angle), last_axes=-2, broadcast_compatible=True)

    # it really should be normalized here since we do it at the beginning
    axis = tf_tr_utils_assert.assert_normalized(axis)

    half_angle = 0.5 * angle
    w = tf.cos(half_angle)
    xyz = tf.sin(half_angle) * axis
    return tf.concat((xyz, w), axis=-1)


def multiply(quaternion1, quaternion2, name=None):
    """Multiplies two quaternions.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      quaternion1:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a quaternion.
      quaternion2:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a quaternion.
      name: A name for this op that defaults to "quaternion_multiply".
    Returns:
      A tensor of shape `[A1, ..., An, 4]` representing quaternions.
    Raises:
      ValueError: If the shape of `quaternion1` or `quaternion2` is not supported.
    """

    quaternion1 = tf.convert_to_tensor(value=quaternion1)
    quaternion2 = tf.convert_to_tensor(value=quaternion2)

    tf_tr_utils_shape.check_static(
        tensor=quaternion1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    tf_tr_utils_shape.check_static(
        tensor=quaternion2, tensor_name="quaternion2", has_dim_equals=(-1, 4))

    x1, y1, z1, w1 = tf.unstack(quaternion1, axis=-1)
    x2, y2, z2, w2 = tf.unstack(quaternion2, axis=-1)
    x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return tf.stack((x, y, z, w), axis=-1)
