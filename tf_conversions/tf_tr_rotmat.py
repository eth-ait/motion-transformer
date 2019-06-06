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
"""This module implements TensorFlow 3d rotation matrix utility functions.
More details rotation matrices can be found on [this page.]
(https://en.wikipedia.org/wiki/Rotation_matrix)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_tr_rotmat_com
import tf_tr_utils_assert
import tf_tr_utils_shape


def assert_rotation_matrix_normalized(matrix, eps=1e-2, name=None):
    """Checks whether a matrix is a rotation matrix.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      matrix: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
        dimensions represent a 3d rotation matrix.
      eps: The absolute tolerance parameter.
      name: A name for this op that defaults to
        'assert_rotation_matrix_normalized'.
    Returns:
      The input matrix, with dependence on the assertion operator in the graph.
    Raises:
      tf.errors.InvalidArgumentError: If rotation_matrix_3d is not normalized.
    """
    matrix = tf.convert_to_tensor(value=matrix)

    tf_tr_utils_shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))

    is_matrix_normalized = is_valid(matrix, atol=eps)
    with tf.control_dependencies([
        tf.assert_equal(
            is_matrix_normalized,
            tf.ones_like(is_matrix_normalized, dtype=tf.bool))
    ]):
        return tf.identity(matrix)


def from_quaternion(quaternion, name=None):
    """Convert a quaternion to a rotation matrix.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
      name: A name for this op that defaults to
        "rotation_matrix_3d_from_quaternion".
    Returns:
      A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
      represent a 3d rotation matrix.
    Raises:
      ValueError: If the shape of `quaternion` is not supported.
    """

    quaternion = tf.convert_to_tensor(value=quaternion)

    tf_tr_utils_shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    quaternion = tf_tr_utils_assert.assert_normalized(quaternion)

    x, y, z, w = tf.unstack(quaternion, axis=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                       txy + twz, 1.0 - (txx + tzz), tyz - twx,
                       txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=quaternion)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


def is_valid(matrix, atol=1e-2, name=None):
    """Determines if a matrix is a valid rotation matrix.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      matrix: A tensor of shape `[A1, ..., An, 3,3]`, where the last two
        dimensions represent a matrix.
      atol: Absolute tolerance parameter.
      name: A name for this op that defaults to "rotation_matrix_3d_is_valid".
    Returns:
      A tensor of type `bool` and shape `[A1, ..., An, 1]` where False indicates
      that the input is not a valid rotation matrix.
    """
    matrix = tf.convert_to_tensor(value=matrix)

    tf_tr_utils_shape.check_static(
        tensor=matrix,
        tensor_name="matrix",
        has_rank_greater_than=1,
        has_dim_equals=((-2, 3), (-1, 3)))

    return tf_tr_rotmat_com.is_valid(matrix, atol)


def from_axis_angle(axis_angle, name=None):
    """Convert an axis-angle representation to a rotation matrix.
    Note:
      In the following, A1 to An are optional batch dimensions, which must be
      broadcast compatible.
    Args:
      axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a normalized axis.
      angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
        represents a normalized axis.
      name: A name for this op that defaults to
        "rotation_matrix_3d_from_axis_angle".
    Returns:
      A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
      represents a 3d rotation matrix.
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
    axis = tf.where(tf.less_equal(tf.concat([angle]*3, axis=-1), 1e-12), random_axis, axis)

    tf_tr_utils_shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    tf_tr_utils_shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    tf_tr_utils_shape.compare_batch_dimensions(
        tensors=(axis, angle),
        tensor_names=("axis", "angle"),
        last_axes=-2,
        broadcast_compatible=True)

    axis = tf_tr_utils_assert.assert_normalized(axis, eps=1e-12)

    sin_axis = tf.sin(angle) * axis
    cos_angle = tf.cos(angle)
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = tf.unstack(axis, axis=-1)
    cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
    sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
    tmp = cos1_axis_x * axis_y
    m01 = tmp - sin_axis_z
    m10 = tmp + sin_axis_z
    tmp = cos1_axis_x * axis_z
    m02 = tmp + sin_axis_y
    m20 = tmp - sin_axis_y
    tmp = cos1_axis_y * axis_z
    m12 = tmp - sin_axis_x
    m21 = tmp + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
    matrix = tf.stack((diag_x, m01, m02,
                       m10, diag_y, m12,
                       m20, m21, diag_z),
                      axis=-1)  # pyformat: disable
    output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)
