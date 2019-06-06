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
r"""This module implements axis-angle functionalities.
The axis-angle representation is defined as $$\theta\mathbf{a}$$, where
$$\mathbf{a}$$ is a unit vector indicating the direction of rotation and
$$\theta$$ is a scalar controlling the angle of rotation. It is important to
note that the axis-angle does not perform rotation by itself, but that it can be
used to rotate any given vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into
a vector $$\mathbf{v}'$$ using the Rodrigues' rotation formula:
$$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
+\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$
More details about the axis-angle formalism can be found on [this page.]
(https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)
Note: Some of the functions defined in the module expect
a normalized axis $$\mathbf{a} = [x, y, z]^T$$ as inputs where
$$x^2 + y^2 + z^2 = 1$$.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_tr_utils_assert
import tf_tr_utils_safe_ops
import tf_tr_utils_shape


def from_quaternion(quaternion, name=None):
    """Converts a quaternion to an axis-angle representation.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      quaternion: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
        represents a normalized quaternion.
      name: A name for this op that defaults to "axis_angle_from_quaternion".
    Returns:
      Tuple of a tensor of shape `[A1, ..., An, 3]`,
      where the first thre numbers normalized as vector represents the axis,
      and the length of the vector represents the angle.
    Raises:
      ValueError: If the shape of `quaternion` is not supported.
    """
    quaternion = tf.convert_to_tensor(value=quaternion)

    tf_tr_utils_shape.check_static(
        tensor=quaternion, tensor_name="quaternion", has_dim_equals=(-1, 4))
    quaternion = tf_tr_utils_assert.assert_normalized(quaternion)

    # This prevents zero norm xyz and zero w, and is differentiable.
    quaternion += tf_tr_utils_assert.select_eps_for_addition(quaternion.dtype)
    xyz, w = tf.split(quaternion, (3, 1), axis=-1)
    norm = tf.norm(tensor=xyz, axis=-1, keepdims=True)
    angle = 2.0 * tf.atan2(norm, tf.abs(w))
    axis = tf_tr_utils_safe_ops.safe_unsigned_div(tf_tr_utils_safe_ops.nonzero_sign(w) * xyz, norm)

    return axis * angle


def is_normalized(axis, angle, atol=1e-3, name=None):
    """Determines if the axis-angle is normalized or not.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a normalized axis.
      angle: A tensor of shape `[A1, ..., An, 1]` where the last dimension
        represents an angle.
      atol: The absolute tolerance parameter.
      name: A name for this op that defaults to "axis_angle_is_normalized".
    Returns:
      A tensor of shape `[A1, ..., An, 1]`, where False indicates that the axis is
      not normalized.
    """
    axis = tf.convert_to_tensor(value=axis)
    angle = tf.convert_to_tensor(value=angle)

    tf_tr_utils_shape.check_static(tensor=axis, tensor_name="axis", has_dim_equals=(-1, 3))
    tf_tr_utils_shape.check_static(
        tensor=angle, tensor_name="angle", has_dim_equals=(-1, 1))
    tf_tr_utils_shape.compare_batch_dimensions(
        tensors=(axis, angle),
        tensor_names=("axis", "angle"),
        last_axes=-2,
        broadcast_compatible=True)

    norms = tf.norm(tensor=axis, axis=-1, keepdims=True)
    return tf.abs(norms - 1.) < atol
