import tensorflow as tf
from tensorflow.keras import backend as k

def PearsonCoef(y_true, y_pred):
    mu_1 = k.mean(y_true, axis=1, keepdims=True)
    mu_2 = k.mean(y_pred, axis=1, keepdims=True)
    y_true -= mu_1
    y_pred -= mu_2
    std_1 = k.std(y_true, axis=1, keepdims=True)+1.0e-6
    std_2 = k.std(y_pred, axis=1, keepdims=True)+1.0e-6
    return (y_true*y_pred)/(std_1*std_2)


def DEC_KL_loss(y_true, y_pred):
    f = k.sum(y_pred, axis=0, keepdims=True)
    p = (y_pred**2/f) / (k.sum(y_pred**2, axis=-1, keepdims=True)/f)
    kl = k.sum(p*k.log(p)-p*k.log(y_pred), axis=-1, keepdims=True)
    return k.expand_dims(k.expand_dims(kl, axis=-1), axis=-1)


def cluster_var_loss(y_true, y_pred):
    mask = tf.math.divide_no_nan(y_pred, y_pred)
    nonzero_count = tf.math.count_nonzero(mask, axis=(1,2), keepdims=True, dtype=tf.float32)
    
    mae = k.abs(y_true - y_pred) * mask
    mu = tf.math.divide_no_nan( k.sum(mae, axis=(1,2), keepdims=True) , nonzero_count )
    mae = (mae - mu) * mask
    var = tf.math.divide_no_nan( k.sum(mae**2, axis=(1,2), keepdims=True) , nonzero_count )
    return k.mean(var, axis=-1, keepdims=True)


def WAPE(y_true, y_pred):
    y_true = k.sum(y_true, axis=1, keepdims=True)
    y_pred = k.sum(y_pred, axis=1, keepdims=True)
    
    return tf.math.divide_no_nan(k.abs(y_pred-y_true), y_true)