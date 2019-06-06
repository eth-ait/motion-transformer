#Copyright 2018 Google LLC
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
"""Safe divisions and inverse trigonometric functions.
  This module implements safety mechanisms to prevent NaN's and Inf's from
  appearing due to machine precision issues. These safety mechanisms ensure that
  the derivative is unchanged and the sign of the perturbation is unbiased.
  If the debug flag TFG_ADD_ASSERTS_TO_GRAPH is set to True, all affected
  functions also add assertions to the graph to ensure that the fix has worked
  as expected.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tf_tr_utils_assert

def nonzero_sign(x, name=None):
  """Returns the sign of x with sign(0) defined as 1 instead of 0."""

  x = tf.convert_to_tensor(value=x)

  one = tf.ones_like(x)
  return tf.where(tf.greater_equal(x, 0.0), one, -one)

def safe_unsigned_div(a, b, eps=None, name=None):
  """Calculates a/b with b >= 0 safely.
  If the tfg debug flag TFG_ADD_ASSERTS_TO_GRAPH defined in tfg_flags.py
  is set to True, this function adds assertions to the graph that check whether
  b + eps is greather than zero, and the division has no NaN or Inf values.
  Args:
    a: A `float` or a tensor of shape `[A1, ..., An]`, which is the nominator.
    b: A `float` or a tensor of shape `[A1, ..., An]`, which is the denominator.
    eps: A small `float`, to be added to the denominator. If left as `None`, its
      value is automatically selected using `b.dtype`.
    name: A name for this op. Defaults to 'safe_signed_div'.
  Raises:
     InvalidArgumentError: If tf-graphics debug flag is set and the division
       causes `NaN` or `Inf` values.
  Returns:
     A tensor of shape `[A1, ..., An]` containing the results of division.
  """
  a = tf.convert_to_tensor(value=a)
  b = tf.convert_to_tensor(value=b)
  if eps is None:
    eps = tf_tr_utils_assert.select_eps_for_division(b.dtype)
  eps = tf.convert_to_tensor(value=eps, dtype=b.dtype)

  return tf_tr_utils_assert.assert_no_infs_or_nans(a / (b + eps))


def safe_shrink(vector,
                minval=None,
                maxval=None,
                open_bounds=False,
                eps=None,
                name=None):
  """Shrinks vector by (1.0 - eps) based on its dtype.
  This function shrinks the input vector by a very small amount to ensure that
  it is not outside of expected range because of floating point precision
  of operations, e.g. dot product of a normalized vector with itself can
  be greater than `1.0` by a small amount determined by the `dtype` of the
  vector. This function can be used to shrink it without affecting its
  derivative (unlike tf.clip_by_value) and make it safe for other operations
  like `acos(x)`. If the tf-graphics debug flag is set to `True`, this function
  adds assertions to the graph that explicitly check that the vector is in the
  range `[minval, maxval]` when open_bounds is `False`, or in range `]minval,
  maxval[` when open_bounds is `True`.
  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.
  Args:
    vector: A tensor of shape `[A1, ..., An]`.
    minval: A `float` or a tensor of shape `[A1, ..., An]`, which contains the
      the lower bounds for tensor values after shrinking to test against. This
      is only used when both `minval` and `maxval` are not `None`.
    maxval: A `float` or a tensor of shape `[A1, ..., An]`, which contains the
      the upper bounds for tensor values after shrinking to test against. This
      is only used when both `minval` and `maxval` are not `None`.
    open_bounds: A `bool` indicating whether the assumed range is open or
      closed, only to be used when both `minval` and `maxval` are not `None`.
    eps: A `float` that is used to shrink the `vector`. If left as `None`, its
      value is automatically determined from the `dtype` of `vector`.
    name: A name for this op. Defaults to 'safe_shrink'.
  Raises:
    InvalidArgumentError: If tf-graphics debug flag is set and the vector is not
      inside the expected range.
  Returns:
    A tensor of shape `[A1, ..., An]` containing the shrinked values.
  """
  vector = tf.convert_to_tensor(value=vector)
  if eps is None:
     eps = tf_tr_utils_assert.select_eps_for_addition(vector.dtype)
  eps = tf.convert_to_tensor(value=eps, dtype=vector.dtype)

  vector *= (1.0 - eps)
  if minval is not None and maxval is not None:
     vector = tf_tr_utils_assert.assert_all_in_range(
         vector, minval, maxval, open_bounds=open_bounds)
  return vector
