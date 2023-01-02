import tensorflow as tf
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras.layers import Concatenate, Dropout
from tensorflow.keras import backend as k
from tensorflow.keras.regularizers import L1
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Ones, Zeros
import numpy as np

class ProbAttention(Layer):
    """
    The probability attention used in Informer.
    """
    def __init__(self, num_heads, k_dim, v_dim, factor=5, attn_l1_regular=0., attn_dropout=0.1, de_stationary=True, **kwargs):
        super(ProbAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.factor = factor
        self.attn_l1_regular = attn_l1_regular
        self.attn_dropout = attn_dropout
        self.de_stationary = de_stationary
        
        self._built_from_signature = False
        
    
    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "factor": self.factor,
            "attn_l1_regular": self.attn_l1_regular,
            "attn_dropout": self.attn_dropout,
            "query_shape": self.query_shape,
            "key_shape": self.key_shape,
            "value_shape": self.value_shape
        }
        base_config = super(ProbAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    
    def _build_from_signature(self, queries, keys, values):
        self._built_from_signature = True
        
        self.query_shape = tf.TensorShape(queries.shape)
        self.key_shape = tf.TensorShape(keys.shape)
        self.value_shape = tf.TensorShape(values.shape)
        
        self._query_dense = EinsumDense(equation = 'abcd,def->abcef',
                                        output_shape = (None, queries.shape[2], self.num_heads, self.k_dim),
                                        kernel_regularizer = L1(self.attn_l1_regular),
                                        name = 'query')
        self._key_dense = EinsumDense(equation = 'abcd,def->abcef',
                                      output_shape = (None, keys.shape[2], self.num_heads, self.k_dim),
                                      kernel_regularizer = L1(self.attn_l1_regular),
                                      name = 'key')
        self._value_dense = EinsumDense(equation = 'abcd,def->abcef',
                                        output_shape = (None, values.shape[2], self.num_heads, self.v_dim),
                                        kernel_regularizer = L1(self.attn_l1_regular),
                                        name = 'value')
        self._attn_output = EinsumDense(equation = 'bthle,h->btle',
                                        output_shape = (None, values.shape[2], self.v_dim),
                                        name = 'attn_out')
        
    def build(self, input_shape, **kwargs):
        super(ProbAttention, self).build(input_shape, **kwargs)
        
        self.tau = self.add_weight(
            name = 'tau',
            shape = (1,),
            initializer = Ones(),
            trainable = True,
            dtype=self.dtype
        )
        self.delta = self.add_weight(
            name = 'delta',
            shape = (1,),
            initializer = Zeros(),
            trainable = True,
            dtype=self.dtype
        )
        
    
    def _index_matrix(self, indices):
        T,H,Q = self.query_shape[1], self.num_heads, self.n_top
        id_m = tf.ones((1, T, H, Q), dtype=tf.float32)
        former_idx = tf.cast( tf.reshape(tf.where(id_m > 0)[:,:-1], (1, T, H, Q, 3)), tf.int32 )
        former_idx = tf.einsum('ab,bcdef->acdef', indices[:,0:1,0,0]*0+1, former_idx)
        
        return Concatenate()([former_idx, tf.expand_dims(indices, axis=-1)])
        
        
    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        B, T, L_K, H, E = K.shape
        _, _, L_Q, _, _ = Q.shape

        # calculate the sampled Q_K
        index_sample = np.random.choice(np.arange(L_K), sample_k, replace=False)
        K_sample = tf.gather(K, index_sample, axis=2)
        Q_K_sample = tf.einsum('btqhe,btkhe->bthqk', Q, K_sample)

        # find the Top_k query with sparisty measurement
        M = k.max(Q_K_sample, axis=-1) - k.sum(Q_K_sample, axis=-1)/L_K
        top_k = tf.math.top_k(M, n_top, sorted=False).indices
        M_top = self._index_matrix(top_k)

        # use the reduced Q to calculate Q_K
        Q_reduce = tf.gather_nd(tf.transpose(Q, [0,1,3,2,4]), indices = M_top) # factor*ln(L_q)
        Q_reduce = tf.transpose(Q_reduce, [0,1,3,2,4])
        Q_K = tf.einsum('btlhe,btshe->bthls', Q_reduce, K)

        return Q_K, M_top
    
    
    def _update_context(self, V, attn, index):
        ## V: b,t,l,h,e
        ## attn: b,t,h,n_top,e
        ## index: b,t,h,n_top,rank
        V = tf.transpose(V, [0,1,3,2,4])
        contex = k.cumsum(V, axis=-2)
        
        contex = tf.tensor_scatter_nd_update(contex, index, attn)
        
        return contex
        
    
    def call(self, queries, keys, values, attn_mask=None, input_shape=None):
        if not self._built_from_signature:
            self._build_from_signature(queries, keys, values)
            
        L_Q = queries.shape[2]
        L_K = keys.shape[2]

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        
        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        self.n_top = u
            
        Q = self._query_dense(queries)
        
        K = self._key_dense(keys)
        
        V = self._value_dense(values)
        
        scores_top, index = self._prob_QK(Q, V, sample_k=U_part, n_top=u) 
        if self.de_stationary:
            scores_top = scores_top * self.tau + self.delta

        # add scale factor
        scale = 1./tf.math.sqrt(tf.constant(self.k_dim, dtype=tf.float32))
        if scale is not None:
            scores_top *= scale
        if attn_mask is not None:
            scores_top *= attn_mask
        
        scores_top = Dropout(self.attn_dropout)(scores_top)
        
        scores_top = tf.math.softmax(scores_top, axis=-1)
        
        attn = tf.einsum('bthls,btshe->bthle', scores_top, V)
        attn = self._update_context(V, attn, index)
        
        attn_output = self._attn_output(attn)
        
        return attn_output
    
    
    
class FullAttention(Layer):
    def __init__(self, num_heads, k_dim, v_dim, attn_l1_regular=0., attn_dropout=0.1, de_stationary=True, **kwargs):
        super(FullAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.attn_l1_regular = attn_l1_regular
        self.attn_dropout = attn_dropout
        self.de_stationary = de_stationary
        
        self._built_from_signature = False
    
    
    def get_config(self):
        config = {
            "num_heads": self.num_heads,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "attn_l1_regular": self.attn_l1_regular,
            "attn_dropout": self.attn_dropout,
            "query_shape": self.query_shape,
            "key_shape": self.key_shape,
            "value_shape": self.value_shape
        }
        base_config = super(FullAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def build(self, input_shape, **kwargs):
        super(FullAttention, self).build(input_shape, **kwargs)
        
        self.tau = self.add_weight(
            name = 'tau',
            shape = (1,),
            initializer = Ones(),
            trainable = True,
            dtype=self.dtype
        )
        self.delta = self.add_weight(
            name = 'delta',
            shape = (1,),
            initializer = Zeros(),
            trainable = True,
            dtype=self.dtype
        )
        
    
    def _build_from_signature(self, queries, keys, values):
        self._built_from_signature = True
        
        self.query_shape = tf.TensorShape(queries.shape)
        self.key_shape = tf.TensorShape(keys.shape)
        self.value_shape = tf.TensorShape(values.shape)
        
        self._query_dense = EinsumDense(equation = 'abcd,def->abcef',
                                        output_shape = (None, queries.shape[2], self.num_heads, self.k_dim),
                                        kernel_regularizer = L1(self.attn_l1_regular),
                                        name = 'query')
        self._key_dense = EinsumDense(equation = 'abcd,def->abcef',
                                      output_shape = (None, keys.shape[2], self.num_heads, self.k_dim),
                                      kernel_regularizer = L1(self.attn_l1_regular),
                                      name = 'key')
        self._value_dense = EinsumDense(equation = 'abcd,def->abcef',
                                        output_shape = (None, values.shape[2], self.num_heads, self.v_dim),
                                        kernel_regularizer = L1(self.attn_l1_regular),
                                        name = 'value')
        self._attn_output = EinsumDense(equation = 'bthle,h->btle',
                                        output_shape = (None, values.shape[2], self.v_dim),
                                        name = 'attn_out')
        
    
    def call(self, queries, keys, values, attn_mask=None):
        if not self._built_from_signature:
            self._build_from_signature(queries, keys, values)
            
        Q = self._query_dense(queries)

        K = self._key_dense(keys)

        V = self._value_dense(values)

        scale = 1./tf.math.sqrt(tf.constant(self.k_dim, dtype=tf.float32))

        scores = tf.einsum('btlhe,btshe->bthls', Q, K)
        if self.de_stationary:
            scores = scores * self.tau + self.delta
        if scale is not None:
            scores *= scale
        if attn_mask is not None:
            scores *= attn_mask
            
        scores = Dropout(self.attn_dropout)(scores)
        
        scores = tf.math.softmax(scores, axis=-1)
        attn = tf.einsum('bthls,btshe->bthle', scores, V)
            
        attn_output = self._attn_output(attn)

        return attn_output