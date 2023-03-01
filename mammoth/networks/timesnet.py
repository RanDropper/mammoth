import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Conv3D, Concatenate, Dropout, Reshape, ZeroPadding2D, Softmax
from tensorflow.keras.regularizers import L1
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
import numpy as np

class TimesBlock(Layer):
    def __init__(self, nfreq, n_enc_filters, enc_kernel_size, enc_activation, enc_l1_regular, **kwargs):
        super(TimesBlock, self).__init__(**kwargs)
        self.nfreq = nfreq
        self.n_enc_filters = n_enc_filters
        self.enc_kernel_size = enc_kernel_size
        self.enc_activation = enc_activation
        self.enc_l1_regular = enc_l1_regular

    def build(self, input_shape):
        self.conv2d = Conv3D(filters = self.n_enc_filters,
                             kernel_size = (1, self.enc_kernel_size[0], self.enc_kernel_size[1]),
                             activation = self.enc_activation,
                             padding = 'same',
                             kernel_regularizer = L1(self.enc_l1_regular),
                             bias_regularizer = L1(self.enc_l1_regular))

    def call(self, tensor):
        B,H,T,F = tensor.shape

        freq_list = np.arange(1, T//2, (T//2)//self.nfreq)
        amptitude = tf.reduce_mean(
            tf.abs(
                tf.transpose(tf.signal.rfft(tf.transpose(tensor,[0,1,3,2])), [0,1,3,2])
            ), axis=-1
        )
        period_weight = tf.gather(amptitude, freq_list, axis=-1)

        res = []
        for freq in freq_list+1:
            padding_len = int(np.ceil(T/freq)*freq - T)
            tmp_tensor = ZeroPadding2D(((0,0), (0,padding_len)))(tensor)
            tmp_tensor = Reshape((H, freq, int(np.ceil(T/freq)), F))(tmp_tensor)
            tmp_tensor = self.conv2d(tmp_tensor)
            tmp_tensor = Reshape((H, int(np.ceil(T/freq)*freq), F))(tmp_tensor)
            res.append(tmp_tensor[:, :, :T, :])
        res = tf.stack(res, axis=-1)
        period_weight = Softmax(axis=-1)(period_weight)
        period_weight = tf.expand_dims(tf.expand_dims(period_weight, axis=-2), axis=-2)
        res = tf.reduce_sum(res*period_weight, axis=-1)
        res += tensor

        return res