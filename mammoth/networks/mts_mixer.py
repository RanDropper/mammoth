import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate, Dropout, Reshape, ZeroPadding2D
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L1
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
import numpy as np

class temporal_interactor(Layer):
    def __init__(self, hidden_dims, regular, dropout, **kwargs):
        super(temporal_interactor, self).__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.regular = regular
        self.dropout = dropout

    def build(self, input_shape):
        B, H, T, F = input_shape
        self.FFN = Sequential([EinsumDense(equation='bhtf,ts->bhsf',
                                                 output_shape=(H, self.hidden_dims, F),
                                                 activation='gelu',
                                                 kernel_regularizer=L1(self.regular),
                                                 bias_regularizer=L1(self.regular)),
                               Dropout(self.dropout),
                               EinsumDense(equation='bhtf,ts->bhsf',
                                           output_shape=(H, T, F),
                                           activation='gelu',
                                           kernel_regularizer=L1(self.regular),
                                           bias_regularizer=L1(self.regular))])
    def call(self, tensor):
        return self.FFN(tensor)


class MtsMixer(Layer):
    def __init__(self, n_sub_seqs, temp_hidden_dims, channel_hidden_dims, regular,
                 temp_dropout, channel_dropout, **kwargs):
        super(MtsMixer, self).__init__(**kwargs)
        self.n_sub_seqs = n_sub_seqs
        self.temp_hidden_dims = temp_hidden_dims
        self.channel_hidden_dims = channel_hidden_dims
        self.regular = regular
        self.temp_dropout = temp_dropout
        self.channel_dropout = channel_dropout

        self._build_from_signature()

    def _build_from_signature(self):
        self.FFN_list = [temporal_interactor(self.temp_hidden_dims, self.regular, self.temp_dropout) for _ in range(self.n_sub_seqs)]
        self.channel_interactor = Sequential([Dense(self.channel_hidden_dims,
                                                    activation='gelu',
                                                    kernel_regularizer=L1(self.regular),
                                                    bias_regularizer=L1(self.regular)),
                                              Dropout(self.channel_dropout),
                                              Dense(self.channel_hidden_dims,
                                                    kernel_regularizer=L1(self.regular),
                                                    bias_regularizer=L1(self.regular))])

    def call(self, tensor):
        B,H,T,F = tensor.shape
        padding_len = int(np.ceil(tensor.shape[2] / self.n_sub_seqs)) * self.n_sub_seqs - tensor.shape[2]
        tensor = ZeroPadding2D(((0, 0), (padding_len, 0)))(tensor)

        tensor_splits = [tensor[:, :, i::self.n_sub_seqs, :] for i in range(self.n_sub_seqs)]
        out_tensor_list = []
        for i, subseq in enumerate(tensor_splits):
            out_tensor_list.append(self.FFN_list[i](subseq))
        out_tensor = Reshape((H, T, F))(
            tf.transpose(tf.stack(out_tensor_list, axis=2), (0,1,3,2,4))
        )
        return self.channel_interactor(out_tensor + tensor)
