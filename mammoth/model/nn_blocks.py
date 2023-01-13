import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Lambda, MaxPool2D, Conv1D, Conv2D, \
    TimeDistributed, LayerNormalization, ZeroPadding2D, Concatenate, Layer
from tensorflow.keras.regularizers import L1
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras.layers import ZeroPadding1D
from mammoth.utils import sparsemax, scatter_update
from mammoth.networks.attention import FullAttention, ProbAttention
from mammoth.networks.scinet import SCINet
from mammoth.networks.tabnet import TabNet
from mammoth.networks.normalization import InstanceNormalization


class ModelBlock(Layer):
    def __init__(self, **kwargs):
        super(ModelBlock, self).__init__(**kwargs)
        self.added_loss = None

    @tf.autograph.experimental.do_not_convert
    def call(self, tensor, **kwargs):
        if len(tensor.shape) == 2:
            tensor_shape = (None, tensor.shape[-1])
        elif len(tensor.shape) == 3:
            tensor_shape = (None, None, tensor.shape[-1])
        elif len(tensor.shape) == 4:
            tensor_shape = (None, None, tensor.shape[-2], tensor.shape[-1])
        else:
            raise ValueError("The rank of {} input tensor should be <= 4, but recieve {}".format(self.name, tensor.shape))

        def inner_build(tensor):
            return self.forward(tensor, **kwargs)
        
        return inner_build(tensor)

    def forward(self, tensor, **kwargs):
        return tensor


class ForkTransform(ModelBlock):
    def __init__(self, hp, name='ForkTransform', **kwargs):
        super(ForkTransform, self).__init__(name=name, **kwargs)
        self.hp = hp.copy()

    def forward(self, tensor, **kwargs):
        perc_horizon = self.hp.get('perc_horizon')
        fcst_horizon = self.hp.get('fcst_horizon')
        window_freq = self.hp.get('window_freq')
        masking = kwargs.get('masking')
        dynamic_feat = kwargs.get('dynamic_feat')
        seq_target = kwargs.get('seq_target')
        enc_feat = kwargs.get('enc_feat')
        dec_feat = kwargs.get('dec_feat')
        is_fcst = kwargs.get('is_fcst')
        remainder = kwargs.get('remainder')

        enc_idx = [dynamic_feat.index(i) for i in seq_target + enc_feat]
        dec_idx = [dynamic_feat.index(i) for i in dec_feat]

        if is_fcst:
            tensor = tensor[:, -(perc_horizon + fcst_horizon):, :]
            masking = masking[:, -(perc_horizon + fcst_horizon):, :]
            enc_tensor = tf.gather(tensor[:, :perc_horizon, :], enc_idx, axis=-1)
            if len(dec_idx) > 0:
                dec_tensor = tf.gather(tf.expand_dims(tensor[:, -fcst_horizon:, :], axis=1), dec_idx, axis=-1)
            else:
                dec_tensor = None
            his_masking = masking[:, :perc_horizon, :]
            fut_masking = tf.expand_dims(masking[:, -fcst_horizon:, :], axis=1)
        else:
            enc_tensor = tf.gather(tensor[:, :-1, :], enc_idx, axis=-1)
            if len(dec_idx) > 0:
                dec_tensor = tf.gather(
                    tf.signal.frame(tensor[:, 1:, :], fcst_horizon, window_freq, axis=1),
                    dec_idx, axis=-1
                )
                if remainder is not None:
                    dec_tensor = dec_tensor[:, -remainder:, :, :]
            else:
                dec_tensor = None
            his_masking = masking[:, :-1, :]
            fut_masking = tf.signal.frame(masking[:, 1:, :], fcst_horizon, window_freq, axis=1)
            if remainder is not None:
                fut_masking = fut_masking[:, -remainder:, :, :]
        return enc_tensor, dec_tensor, his_masking, fut_masking
    

class MovingWindowTransform(ModelBlock):
    def __init__(self, hp, name='MovingWindowTransform', **kwargs):
        super(MovingWindowTransform, self).__init__(name=name, **kwargs)
        self.hp = hp.copy()

    def forward(self, tensor, **kwargs):
        perc_horizon = self.hp.get('perc_horizon')
        fcst_horizon = self.hp.get('fcst_horizon')
        window_freq = self.hp.get('window_freq')
        masking = kwargs.get('masking')
        dynamic_feat = kwargs.get('dynamic_feat')
        seq_target = kwargs.get('seq_target')
        enc_feat = kwargs.get('enc_feat')
        dec_feat = kwargs.get('dec_feat')
        is_fcst = kwargs.get('is_fcst')
        remainder = kwargs.get('remainder')

        enc_idx = [dynamic_feat.index(i) for i in seq_target + enc_feat]
        dec_idx = [dynamic_feat.index(i) for i in dec_feat]

        if is_fcst:
            padding_len = perc_horizon+fcst_horizon-tensor.shape[1]
            if padding_len > 0:
                tensor = ZeroPadding1D((padding_len, 0))(tensor)
                masking = ZeroPadding1D((padding_len, 0))(masking)
            tensor = tensor[:, -(perc_horizon+fcst_horizon):, :]
            masking = masking[:, -(perc_horizon+fcst_horizon):, :]
            tensor = tf.expand_dims(tensor, axis=1)
            masking = tf.expand_dims(masking, axis=1)
        else:
            tensor = ZeroPadding1D((perc_horizon-1, fcst_horizon-1))(tensor)
            tensor = tf.signal.frame(tensor, perc_horizon+fcst_horizon, window_freq, axis=1)

            masking = ZeroPadding1D((perc_horizon-1, fcst_horizon-1))(masking)
            masking = tf.signal.frame(masking, perc_horizon+fcst_horizon, window_freq, axis=1)

        enc_tensor = tf.gather(tensor[:, :, :perc_horizon, :], enc_idx, axis=-1)
        his_masking = masking[:, :, :perc_horizon, :]
        if remainder is None:
            fut_masking = masking[:, :, -fcst_horizon:, :]
        else:
            fut_masking = masking[:, -remainder:, -fcst_horizon:, :]

        if len(dec_idx) > 0:
            if remainder is None:
                dec_tensor = tf.gather(tensor[:, :, -fcst_horizon:, :], dec_idx, axis=-1)
            else:
                dec_tensor = tf.gather(tensor[:, -remainder:, -fcst_horizon:, :], dec_idx, axis=-1)
        else:
            dec_tensor = None

        return enc_tensor, dec_tensor, his_masking, fut_masking


class InstanceNorm(ModelBlock):
    def __init__(self, hp, name='InstanceNorm', **kwargs):
        super(InstanceNorm, self).__init__(name=name, **kwargs)
        self.hp = hp.copy()
        self.method = kwargs.get('method')
        self.norm_col_idx = kwargs.get('norm_col_idx')
        self.axis = kwargs.get('axis', -2)
        self.affine = kwargs.get('affine', True)
        self._built_from_signature()

    def _built_from_signature(self):
        self.IN = InstanceNormalization(axis=self.axis, method=self.method,
                                        affine=self.affine,
                                        name='{}_norm_on_feature{}'.format(self.method, self.built_times))


    def forward(self, tensor, **kwargs):
        masking = kwargs.get('masking')
        tensor_norm_update = self.IN(tf.gather(tensor, self.norm_col_idx, axis=-1), masking=masking)
        tensor = scatter_update(tensor, self.norm_col_idx, tensor_norm_update, axis=-1)
        return tensor


class SimpleEmbedding(ModelBlock):
    def __init__(self, hp, name='SimpleEmbedding', **kwargs):
        super(SimpleEmbedding, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        n_embed_layers = self.hp.get('n_embed_layers', 1)
        embed_out_dim = self.hp.get('embed_out_dim', input_shape[-1])
        embed_l1_regular = self.hp.get('embed_l1_regular', 0)

        self.dense_list = [Dense(embed_out_dim,
                                 use_bias=False,
                                 kernel_regularizer=L1(embed_l1_regular),
                                 name='{}_{}'.format(self.name, i)) for i in range(n_embed_layers)]

    
    def forward(self, tensor, **kwargs):
        for layer in self.dense_list:
            tensor = layer(tensor)
        return tensor
    
    
class AttentionStack(ModelBlock):
    def __init__(self, hp, name='AttentionStack', **kwargs):
        super(AttentionStack, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        rel_dim = self.hp.get('rel_dim', 4)
        k_clusters = self.hp.get('n_stack')
        stack_dropout = self.hp.get('stack_dropout', 0)

        self.Q_dense = Dense(rel_dim, use_bias=False, name = '{}_Q_dense'.format(self.name))
        self.K_dense = Dense(rel_dim, use_bias=False, name = '{}_K_dense'.format(self.name))
        self.V_dense = Dense(rel_dim, use_bias=False, name = '{}_V_dense'.format(self.name))
        self.dropout = Dropout(stack_dropout)
        self.prob_dense = Dense(k_clusters, use_bias=False, name='{}_prob_dense'.format(self.name))
    
    def forward(self, tensor, **kwargs):
        Q = self.Q_dense(tensor)
        K = self.K_dense(tensor)
        V = self.V_dense(tensor)
        
        rel_matrix = tf.einsum('Bte,bte->tBb', Q, K)
        rel_matrix = tf.math.reduce_mean(rel_matrix, axis=0)
        rel_matrix = self.dropout(rel_matrix)
        rel_matrix = tf.einsum('Bb,bte->Bte', rel_matrix, V)
        rel_matrix = tf.math.reduce_mean(rel_matrix, axis=1)
        
        cluster_prob = Lambda(lambda x:tf.nn.softmax(x, axis=-1), name='cluster_prob')(
            self.prob_dense(rel_matrix)
        )
        return cluster_prob
    
    
class GCNStack(ModelBlock):
    def __init__(self, hp, name='GCNStack', **kwargs):
        super(GCNStack, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        gcn_layers = self.hp.get('gcn_layers', 2)
        rel_dim = self.hp.get('rel_dim', 4)
        n_stack = self.hp.get('n_stack')
        gcn_pool_size = self.hp.get('gcn_pool_size', 4)

        self.maxpool_list = [MaxPool2D((gcn_pool_size, 1), 1, padding='same', name='{}_maxpooling_{}'.format(self.name, i))
                             for i in range(gcn_layers)]
        self.weight_dense_list = [Dense(rel_dim,
                                        use_bias=False,
                                        name = '{}_weight_matrix_{}'.format(self.name, i)) for i in range(gcn_layers)]
        self.prob_dense = Dense(n_stack, use_bias=False, name='{}_prob_dense'.format(self.name))

    
    def forward(self, tensor, **kwargs):
        for weight_dense, maxpool2d in zip(self.weight_dense_list, self.maxpool_list):
            adj_matrix = tf.einsum('Bte,bte->tBb', tensor, tensor)
            adj_matrix = tf.math.reduce_mean(adj_matrix, axis=0)
            adj_matrix = sparsemax(adj_matrix, axis=-1)
            adj_matrix = tf.einsum('Bb,bte->Bbte', adj_matrix, tensor)
            
            adj_matrix = maxpool2d(adj_matrix)
            adj_matrix = tf.math.reduce_sum(adj_matrix, axis=1)
            
            tensor = weight_dense(adj_matrix)
            tensor = tf.nn.relu(tensor)
        
        tensor = tf.math.reduce_mean(tensor, axis=1)
        
        cluster_prob = Lambda(lambda x:tf.nn.softmax(x, axis=-1), name='cluster_prob')(
            self.prob_dense(tensor)
        )
        return cluster_prob
    

    
class WavenetEncoder(ModelBlock):
    def __init__(self, hp, name='WavenetEncoder', **kwargs):
        super(WavenetEncoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        n_enc_layers = self.hp.get('n_enc_layers', 6)
        n_enc_filters = self.hp.get('n_enc_filters', 16)
        enc_kernel_size = self.hp.get('enc_kernel_size', 2)
        enc_channel_groups = self.hp.get('enc_channel_groups', 1)
        enc_activation = self.hp.get('enc_activation', 'relu')
        enc_l1_regular = self.hp.get('enc_l1_regular', 0)
        is_fork = self.hp.get('is_fork', False)

        if is_fork:
            self.conv_list = [Conv1D(filters = n_enc_filters,
                                     kernel_size = enc_kernel_size,
                                     padding = 'causal',
                                     dilation_rate = enc_kernel_size**i,
                                     groups = enc_channel_groups,
                                     activation = enc_activation,
                                     kernel_regularizer = L1(enc_l1_regular),
                                     bias_regularizer = L1(enc_l1_regular),
                                     name = '{}_{}'.format(self.name, i)) for i in range(n_enc_layers)]
        else:
            self.conv_list = [TimeDistributed(Conv1D(filters = n_enc_filters,
                                              kernel_size = enc_kernel_size,
                                              padding = 'causal',
                                              dilation_rate = enc_kernel_size**i,
                                              groups = enc_channel_groups,
                                              activation = enc_activation,
                                              kernel_regularizer = L1(enc_l1_regular),
                                              bias_regularizer = L1(enc_l1_regular)),
                                       name = '{}_{}'.format(self.name, i)) for i in range(n_enc_layers)]

    def forward(self, tensor, **kwargs):
        for conv in self.conv_list:
            tensor = conv(tensor)
        return tensor
    

    
class TransformerEncoder(ModelBlock):
    def __init__(self, hp, name='TransformerEncoder', **kwargs):
        super(TransformerEncoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        num_heads = self.hp.get('enc_num_heads', 4)
        k_dim = self.hp.get('enc_k_dim', 4)
        v_dim = self.hp.get('enc_v_dim', 4)
        enc_attn_layers = self.hp.get('enc_attn_layers', 1)
        attn_l1_regular = self.hp.get('attn_l1_regular', 0)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)

        self.full_attn_list = [FullAttention(num_heads,
                                             k_dim,
                                             v_dim,
                                             attn_l1_regular,
                                             attn_dropout,
                                             de_stationary = de_stationary,
                                             name = '{}_full_attn_{}'.format(self.name, i))
                               for i in range(enc_attn_layers)]
        self.dense_list = [Dense(input_shape[-1],
                                 use_bias = False,
                                 name = '{}_reshape_dense_{}'.format(self.name, i))
                           for i in range(enc_attn_layers-1)]
        self.LN_list = [LayerNormalization(name = '{}_LN_{}'.format(self.name, i))
                        for i in range(enc_attn_layers-1)]


    def forward(self, tensor, **kwargs):
        for L, full_attn in enumerate(self.full_attn_list):
            attn_output = full_attn(tensor, tensor, tensor)
                
            if L < len(self.full_attn_list) - 1:
                attn_output = self.dense_list[L](attn_output)
                tensor += self.LN_list[L](attn_output)
            else:
                return attn_output
    
    
    
class InformerEncoder(ModelBlock):
    def __init__(self, hp, name='InformerEncoder', **kwargs):
        super(InformerEncoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        num_heads = self.hp.get('enc_num_heads', 4)
        k_dim = self.hp.get('enc_k_dim', 4)
        v_dim = self.hp.get('enc_v_dim', 4)
        factor = self.hp.get('factor', 5)
        enc_attn_layers = self.hp.get('enc_attn_layers', 3)
        attn_l1_regular = self.hp.get('attn_l1_regular', 0)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)
        downconv_size = self.hp.get('downconv_size', 2)

        self.prob_attn_list = [ProbAttention(num_heads,
                                             k_dim,
                                             v_dim,
                                             factor,
                                             attn_l1_regular,
                                             attn_dropout,
                                             de_stationary = de_stationary,
                                             name = '{}_prob_attn_{}'.format(self.name, i))
                               for i in range(enc_attn_layers)]
        self.zero_padding = ZeroPadding2D(((0,0),(downconv_size-1,0)))
        self.conv_list = [Conv2D(filters = input_shape[-1],
                                 kernel_size = (1, downconv_size),
                                 strides = (1, 1),
                                 padding = 'valid',
                                 activation = 'elu',
                                 name = '{}_conv2d_{}'.format(self.name, i))
                          for i in range(enc_attn_layers)]
        self.maxpool_list = [MaxPool2D(pool_size = (1, downconv_size),
                                       strides = (1, downconv_size),
                                       padding = 'valid',
                                       name = '{}_maxpool_{}'.format(self.name, i))
                             for i in range(enc_attn_layers)]
        self.LN_list = [LayerNormalization(name = '{}_LN_{}'.format(self.name, i))
                        for i in range(enc_attn_layers-1)]

    def forward(self, tensor, **kwargs):
        for L, prob_attn in enumerate(self.prob_attn_list):
            attn_output = prob_attn(tensor, tensor, tensor)
            attn_output = self.zero_padding(attn_output)
            attn_output = self.conv_list[L](attn_output)
            attn_output = self.maxpool_list[L](attn_output)
            
            if L < len(self.prob_attn_list) - 1:
                tensor = self.LN_list[L](attn_output)
            else:
                return attn_output
    

class SciEncoder(ModelBlock):
    def __init__(self, hp, name='SciEncoder', **kwargs):
        super(SciEncoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        n_splits = self.hp.get('n_splits', 2)
        n_levels = self.hp.get('n_levels', 4)
        n_blocks = self.hp.get('n_blocks', 1)
        hidden_size = self.hp.get('hidden_size', 4)
        kernel_size = self.hp.get('enc_kernel_size', 5)
        groups = self.hp.get('enc_channel_groups', 1)
        regular = self.hp.get('enc_l1_regular', 0)
        dropout = self.hp.get('enc_dropout', 0.5)
        is_fork = self.hp.get('is_fork', False)

        self.scinet_list = [SCINet(n_splits,
                                   n_levels,
                                   hidden_size,
                                   kernel_size,
                                   groups,
                                   regular,
                                   dropout,
                                   is_fork,
                                   name='{}_{}'.format(self.name, i))
                            for i in range(n_blocks)]


    def forward(self, tensor, **kwargs):
        stacked = []
        for scinet in self.scinet_list:
            single_out = scinet(tensor)
            stacked.append(single_out)
        
        if len(stacked) > 1:
            return Concatenate()(stacked)
        else:
            return stacked[0]
        
        
        
class DenseDecoder(ModelBlock):
    def __init__(self, hp, name='DenseDecoder', **kwargs):
        super(DenseDecoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        fcst_len = self.hp.get('fcst_horizon')
        activation = self.hp.get('dec_activation', 'swish')

        self.dense = EinsumDense(equation = 'btle,lf->btfe',
                                 output_shape = (None, fcst_len, input_shape[-1]),
                                 activation = activation,
                                 name = '{}_out'.format(self.name))
        
    def forward(self, tensor, **kwargs):
        tensor = self.dense(tensor)
        return tensor

    
class CaCtDecoder(ModelBlock):
    def __init__(self, hp, name='CaCtDecoder', **kwargs):
        super(CaCtDecoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        fcst_len = self.hp.get('fcst_horizon')
        ca_dim = self.hp.get('ca_dim', 16)
        ca_activation = self.hp.get('ca_activation', 'swish')
        ca_l1_regular = self.hp.get('ca_l1_regular', 0)
        ct_dim = self.hp.get('ct_dim', 4)
        ct_activation = self.hp.get('ct_activation', 'swish')
        ct_l1_regular = self.hp.get('ct_l1_regular', 0)

        self.ca_dense = EinsumDense(equation = 'btle,lk->btk',
                                    output_shape = (None, ca_dim),
                                    activation = ca_activation,
                                    kernel_regularizer = L1(ca_l1_regular),
                                    name = '{}_ca'.format(self.name))
        self.ct_dense = EinsumDense(equation = 'btle,lfk->btfk',
                                    output_shape = (None, fcst_len, ct_dim),
                                    activation = ct_activation,
                                    kernel_regularizer = L1(ct_l1_regular),
                                    name = '{}_ct'.format(self.name))


    def forward(self, tensor, **kwargs):
        fcst_len = self.hp.get('fcst_horizon')

        ca_block = self.ca_dense(tensor)
        ca_block = tf.tile( tf.expand_dims(ca_block, axis=2), [1,1,fcst_len,1] )
        
        ct_block = self.ct_dense(tensor)

        return Concatenate()([ca_block, ct_block])

    
    
class ConvRecoder(ModelBlock):
    def __init__(self, hp, name='ConvRecoder', **kwargs):
        super(ConvRecoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        n_rec_layers = self.hp.get('n_rec_layers', 3)
        n_rec_filters = self.hp.get('n_rec_filters', 8)
        rec_kernel_size = self.hp.get('rec_kernel_size', 2)
        rec_channel_groups = self.hp.get('rec_channel_groups', 1)
        rec_activation = self.hp.get('rec_activation', 'swish')
        rec_l1_regular = self.hp.get('rec_l1_regular', 0)

        self.zero_padding = ZeroPadding2D(((0,0),(0,rec_kernel_size-1)))
        self.conv_list = [Conv2D(filters = n_rec_filters,
                                 kernel_size = (1, rec_kernel_size),
                                 padding = 'valid',
                                 groups = rec_channel_groups,
                                 activation = rec_activation,
                                 kernel_regularizer = L1(rec_l1_regular),
                                 bias_regularizer = L1(rec_l1_regular),
                                 name = '{}_conv_{}'.format(self.name, i))
                          for i in range(n_rec_layers)]
    
    def forward(self, tensor, **kwargs):
        if tensor.shape[-2] > 1:
            for conv in self.conv_list:
                tensor = self.zero_padding(tensor)
                tensor = conv(tensor)
        return tensor
    

    
class AttentionRecoder(ModelBlock):
    def __init__(self, hp, name='AttentionRecoder', **kwargs):
        super(AttentionRecoder, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
   
    def forward(self, tensor, **kwargs):
        pass
    

    
class MlpOutput(ModelBlock):
    def __init__(self, hp, name='MlpOutput', **kwargs):
        super(MlpOutput, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()
        self._build_from_signature()

    def _build_from_signature(self):
        mlp_dims = self.hp.get('mlp_dims', [16])
        mlp_activation = self.hp.get('mlp_activation', 'swish')
        mlp_l1_regular = self.hp.get('mlp_l1_regular', 0)

        self.dense_list = [Dense(mlp_dims[i],
                                 activation = mlp_activation,
                                 kernel_regularizer = L1(mlp_l1_regular),
                                 name = '{}_mlp_{}'.format(self.name, i))
                           for i in range(len(mlp_dims))]
        self.out_dense = Dense(1, use_bias=False, name='{}_output'.format(self.name))
        
    def forward(self, tensor, **kwargs):
        for dense in self.dense_list:
            tensor = dense(tensor)
        
        output = self.out_dense(tensor)
        
        return output
    

    
class AttentionOutput(ModelBlock):
    def __init__(self, hp, name='AttentionOutput', **kwargs):
        super(AttentionOutput, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        num_heads = self.hp.get('dec_num_heads', 8)
        k_dim = self.hp.get('dec_k_dim', 16)
        dec_attn_layers = self.hp.get('dec_attn_layers', 1)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)

        self.full_attn_list = [FullAttention(num_heads, k_dim, input_shape[-1],
                                             attn_dropout = attn_dropout,
                                             de_stationary = de_stationary,
                                             name = '{}_output_attn_{}'.format(self.name, i))
                               for i in range(dec_attn_layers)]
        self.LN_list = [LayerNormalization(name = '{}_LN_{}'.format(self.name, i))
                        for i in range(dec_attn_layers)]
        self.out_dense = Dense(1, use_bias=False, name='{}_output'.format(self.name))

    def forward(self, tensor, **kwargs):
        for L, full_attn in enumerate(self.full_attn_list):
            x = full_attn(tensor, tensor, tensor)
            tensor += self.LN_list[L](x)
        
        output = self.out_dense(tensor)
        
        return output
    

class TabnetOutput(ModelBlock):
    def __init__(self, hp, name='TabnetOutput', **kwargs):
        super(TabnetOutput, self).__init__(name = name, **kwargs)
        self.hp = hp.copy()

    def build(self, input_shape):
        tab_feat_dim = self.hp.get('tab_feat_dim', 16)
        decision_out_dim = self.hp.get('decision_out_dim', 16)
        num_decision_steps = self.hp.get('num_decision_steps', 5)
        relaxation_factor = self.hp.get('relaxation_factor', 1.5)
        sparsity_coef = self.hp.get('sparsity_coef', 1.0e-5)
        norm_type = self.hp.get('norm_type', 'group')
        num_groups = self.hp.get('num_groups', 2)
        virtual_batch_size = self.hp.get('virtual_batch_size', 8)

        self.tabnet = TabNet(feature_dim = tab_feat_dim,
                             output_dim = decision_out_dim,
                             num_features = input_shape[-1],
                             num_decision_steps = num_decision_steps,
                             relaxation_factor =  relaxation_factor,
                             sparsity_coefficient = sparsity_coef,
                             norm_type = norm_type,
                             num_groups = num_groups,
                             virtual_batch_size = virtual_batch_size,
                             name = '{}_tabnet'.format(self.name))
        self.out_dense = Dense(1, use_bias=False, name='{}_output'.format(self.name))

        
    def forward(self, tensor, **kwargs):
        tab_output, entropy_loss = self.tabnet(tensor)
        
        output = self.out_dense(tab_output)
        
        self.added_loss = entropy_loss
        
        return output