import tensorflow as tf
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras.layers import Concatenate, Dropout, Layer
from tensorflow.keras.regularizers import L1


class SpatialAttention(Layer):
    def __init__(self, num_heads, k_dim, v_dim, attn_l1_regular=0., attn_dropout=0.1, is_dynamic=True, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.attn_l1_regular = attn_l1_regular
        self.attn_dropout = attn_dropout
        self.is_dynamic = is_dynamic

    def build(self, input_shape):
        self._query_dense = EinsumDense(equation='abcd,def->abcef',
                                        output_shape=(None, input_shape[2], self.num_heads, self.k_dim),
                                        kernel_regularizer=L1(self.attn_l1_regular),
                                        name='query')
        self._key_dense = EinsumDense(equation='abcd,def->abcef',
                                      output_shape=(None, input_shape[2], self.num_heads, self.k_dim),
                                      kernel_regularizer=L1(self.attn_l1_regular),
                                      name='key')
        self._value_dense = EinsumDense(equation='abcd,def->abcef',
                                        output_shape=(None, input_shape[2], self.num_heads, self.v_dim),
                                        kernel_regularizer=L1(self.attn_l1_regular),
                                        name='value')
        if self.is_dynamic:
            self._graph_encode = EinsumDense(equation='tlhBb,h->tlBb',
                                             output_shape=(None, None, None),
                                             name='graph_encode')
        else:
            self._graph_encode = EinsumDense(equation='tlhBb,lh->tBb',
                                             output_shape=(None, None),
                                             name='graph_encode')

    def call(self, queries, keys, values, attn_mask=None):
        Q = self._query_dense(queries)
        K = self._key_dense(keys)
        V = self._value_dense(values)

        scale = 1. / tf.math.sqrt(tf.constant(self.k_dim, dtype=tf.float32))
        scores = tf.einsum('Btlhe,btlhe->tlhBb', Q, K)

        if scale is not None:
            scores *= scale
        if attn_mask is not None:
            scores *= attn_mask

        scores = Dropout(self.attn_dropout)(scores)
        scores = tf.math.softmax(self._graph_encode(scores), axis=-1)
        if self.is_dynamic:
            attn = tf.einsum('tlBb,btlhe->Btle', scores, V)
        else:
            attn = tf.einsum('tBb,btlhe->Btle', scores, V)

        return (attn, scores)

