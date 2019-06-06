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
"""Assert functions to be used by various modules.
This module contains asserts that are intended to be used in TensorFlow
Graphics. These asserts will be activated only if the debug flag
TFG_ADD_ASSERTS_TO_GRAPH is set to True.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def assert_no_infs_or_nans(tensor, name=None):
    """Checks a tensor for NaN and Inf values.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      tensor: A tensor of shape `[A1, ..., An]` containing the values we want to
        check.
      name: A name for this op. Defaults to 'assert_no_infs_or_nans'.
    Raises:
      tf.errors.InvalidArgumentError: If any entry of the input is NaN or Inf.
    Returns:
      The input vector, with dependence on the assertion operator in the graph.
    """

    tensor = tf.convert_to_tensor(value=tensor)

    assert_ops = (tf.debugging.check_numerics(
        tensor, message='Inf or NaN detected.'),)
    with tf.control_dependencies(assert_ops):
        return tf.identity(tensor)


def assert_normalized(vector, axis=-1, eps=None, name=None):
    """Checks whether vector/quaternion is normalized in its last dimension.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      vector: A tensor of shape `[A1, ..., M, ..., An]`, where the axis of M
        contains the vectors.
      axis: The axis containing the vectors.
      eps: A `float` describing the tolerance used to determine if the norm is
        equal to `1.0`.
      name: A name for this op. Defaults to 'assert_normalized'.
    Raises:
      InvalidArgumentError: If the norm of `vector` is not `1.0`.
    Returns:
      The input vector, with dependence on the assertion operator in the graph.
    """

    vector = tf.convert_to_tensor(value=vector)
    if eps is None:
        eps = select_eps_for_division(vector.dtype)
    eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

    norm = tf.norm(tensor=vector, axis=axis)
    one = tf.constant(1.0, dtype=norm.dtype)
    with tf.control_dependencies([tf.assert_near(norm, one, atol=eps)]):
        return tf.identity(vector)


def select_eps_for_addition(dtype):
    """Returns 2 * machine epsilon based on `dtype`.
    This function picks an epsilon slightly greater than the machine epsilon,
    which is the upper bound on relative error. This value ensures that
    `1.0 + eps != 1.0`.
    Args:
      dtype: The `tf.DType` of the tensor to which eps will be added.
    Raises:
      ValueError: If `dtype` is not a floating type.
    Returns:
      A `float` to be used to make operations safe.
    """
    return 2.0 * np.finfo(dtype.as_numpy_dtype()).eps


def select_eps_for_division(dtype):
    """Selects default values for epsilon to make divisions safe based on dtype.
    This function returns an epsilon slightly greater than the smallest positive
    floating number that is representable for the given dtype. This is mainly used
    to prevent division by zero, which produces Inf values. However, if the
    nominator is orders of magnitude greater than `1.0`, eps should also be
    increased accordingly. Only floating types are supported.
    Args:
      dtype: The `tf.DType` of the tensor to which eps will be added.
    Raises:
      ValueError: If `dtype` is not a floating type.
    Returns:
      A `float` to be used to make operations safe.
    """
    return 10.0 * np.finfo(dtype.as_numpy_dtype()).tiny


def assert_all_in_range(vector, minval, maxval, open_bounds=False, name=None):
    """Checks whether all values of vector are between minval and maxval.
    This function checks if all the values in the given vector are in an interval
    `[minval, maxval]` if `open_bounds` is `False`, or in `]minval, maxval[` if it
    is set to `True`.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      vector: A tensor of shape `[A1, ..., An]` containing the values we want to
        check.
      minval: A `float` or a tensor of shape `[A1, ..., An]` representing the
        desired lower bound for the values in `vector`.
      maxval: A `float` or a tensor of shape `[A1, ..., An]` representing the
        desired upper bound for the values in `vector`.
      open_bounds: A `bool` indicating whether the range is open or closed.
      name: A name for this op. Defaults to 'assert_all_in_range'.
    Raises:
      tf.errors.InvalidArgumentError: If `vector` is not in the expected range.
    Returns:
      The input vector, with dependence on the assertion operator in the graph.
    """
    # if not FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value:
    #   return vector

    vector = tf.convert_to_tensor(value=vector)
    minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
    maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)

    if open_bounds:
        assert_op_1 = tf.Assert(tf.less(tf.reduce_max(vector), maxval), [vector])
        assert_op_2 = tf.Assert(tf.greater(tf.reduce_max(vector), minval), [vector])

    else:
        assert_op_1 = tf.Assert(tf.less_equal(tf.reduce_max(vector), maxval), [vector])
        assert_op_2 = tf.Assert(tf.greater_equal(tf.reduce_max(vector), minval), [vector])

    with tf.control_dependencies([assert_op_1, assert_op_2]):
        return tf.identity(vector)
