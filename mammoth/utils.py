import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax
from tensorflow.keras import backend as k
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np


def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    if n_units is None:
        n_units = tf.shape(x)[-1] // 2

    return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])


@tf_export("nn.gelu", v1=[])
def gelu(features, approximate=False, name=None):
  with ops.name_scope(name, "Gelu", [features]):
    features = ops.convert_to_tensor(features, name="features")
    if not features.dtype.is_floating:
      raise ValueError(
          "`features.dtype` must be a floating point tensor."
          f"Received:features.dtype={features.dtype}")
    if approximate:
      coeff = math_ops.cast(0.044715, features.dtype)
      return 0.5 * features * (
          1.0 + math_ops.tanh(0.7978845608028654 *
                              (features + coeff * math_ops.pow(features, 3))))
    else:
      return 0.5 * features * (1.0 + math_ops.erf(
          features / math_ops.cast(1.4142135623730951, features.dtype)))


def sparsemax(logits, axis):
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe



class MultiDimSoftmax(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(MultiDimSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(inputs - tf.reduce_logsumexp(
                    inputs, axis=self.axis, keepdims=True))
            else:
                return tf.math.softmax(inputs, axis=self.axis[0])
        return tf.math.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def compute_normal_mean(tensor, masking, axis, keepdims = None):
    tensor *= masking
    
    normal_count = k.sum(masking, axis=axis, keepdims=keepdims)
    
    return tf.math.divide_no_nan(k.sum(tensor, axis=axis, keepdims=keepdims), normal_count)


def compute_normal_std(tensor, masking, axis, keepdims = None):
    tensor *= masking
    
    normal_mean = compute_normal_mean(tensor, masking, axis, keepdims)
    if not keepdims:
        normal_mean = tf.expand_dims(normal_mean, axis=axis)
    normal_count = k.sum(masking, axis=axis, keepdims=keepdims)
    
    return k.sqrt( tf.math.divide_no_nan(k.sum((tensor-normal_mean)**2, axis=axis, keepdims=keepdims), normal_count) )


def compute_normal_min(tensor, masking, axis, keepdims = None):
    tensor *= masking
    
    return k.min(tensor, axis=axis, keepdims=keepdims)


def compute_normal_max(tensor, masking, axis, keepdims = None):
    tensor *= masking
    
    return k.max(tensor, axis=axis, keepdims=keepdims)
        

def get_perc_horizon(encoder_type, hp):
    if encoder_type == 'WavenetEncoder':
        ks = hp.get('enc_kernel_size', 2)
        l = hp.get('n_enc_layers', 6)
        return ks ** l
    elif encoder_type == 'SciEncoder':
        ks = hp.get('enc_kernel_size', 5)
        nl = hp.get('n_levels', 2)
        ns = hp.get('n_splits', 2)
        return (ks+2) * (ns ** nl)
    else:
        return hp.get('perc_horizon')
    
    
def scatter_update(tensor, indices, updates, axis=-1):
    rank = len(tensor.shape)

    tr = [i for i in range(rank)]
    tr[0], tr[axis] = tr[axis], tr[0]

    tensor = tf.transpose(tensor, tr)
    updates = tf.transpose(updates, tr)

    indices = np.array(indices).reshape(-1, 1)

    tensor = tf.tensor_scatter_nd_update(tensor, indices, updates)


    return tf.transpose(tensor, tr)


def tf_ignore_warnings():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)