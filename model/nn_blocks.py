import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Lambda, MaxPool2D, Conv1D, Conv2D, \
    TimeDistributed, LayerNormalization, ZeroPadding2D, Concatenate
from tensorflow.keras.regularizers import L1
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras import backend as k
import abc
from mammoth.utils import sparsemax
from mammoth.networks.attention import FullAttention, ProbAttention
from mammoth.networks.scinet import SCINet
from mammoth.networks.tabnet import TabNet


class ModelBlock(metaclass = abc.ABCMeta):
    def __init__(self):
        self.added_loss = None
    
    def __call__(self, tensor):
        if len(tensor.shape) == 2:
            tensor_shape = (None, tensor.shape[-1])
        elif len(tensor.shape) == 3:
            tensor_shape = (None, None, tensor.shape[-1])
        elif len(tensor.shape) == 4:
            tensor_shape = (None, None, tensor.shape[-2], tensor.shape[-1])
        else:
            raise ValueError("The rank of {} input tensor should be <= 4, but recieve {}".format(self.name, tensor.shape))
        
        def inner_build(tensor):
            return self.build(tensor)
        
        return inner_build(tensor)
    
    @abc.abstractmethod
    def build(self, tensor):
        return tensor
    


class SimpleEmbedding(ModelBlock):
    def __init__(self, hp, name='SimpleEmbedding', **kwargs):
        super(SimpleEmbedding, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
    
    def build(self, tensor):
        n_embed_layers = self.hp.get('n_embed_layers', 1)
        embed_out_dim = self.hp.get('embed_out_dim', tensor.shape[-1])
        embed_l1_regular = self.hp.get('embed_l1_regular', 0)
        
        for i in range(n_embed_layers):
            tensor = Dense(embed_out_dim,
                           use_bias = False, 
                           kernel_regularizer = L1(embed_l1_regular),
                           name = '{}_{}'.format(self.name, i))(tensor)
        return tensor
    
    
class AttentionStack(ModelBlock):
    def __init__(self, hp, name='AttentionStack', **kwargs):
        super(AttentionStack, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
    
    def build(self, tensor):
        rel_dim = self.hp.get('rel_dim', 4)
        k_clusters = len(self.hp.get('base_models'))
        horizon = self.hp.get('horizon')
        stack_dropout = self.hp.get('stack_dropout', 0)
        
        Q = Dense(rel_dim, use_bias=False, name = '{}_Q_dense'.format(self.name))(tensor)
        K = Dense(rel_dim, use_bias=False, name = '{}_K_dense'.format(self.name))(tensor)
        V = Dense(rel_dim, use_bias=False, name = '{}_V_dense'.format(self.name))(tensor)
        
        rel_matrix = tf.einsum('Bte,bte->tBb', Q, K)
        rel_matrix = tf.math.reduce_mean(rel_matrix, axis=0)
        rel_matrix = Dropout(stack_dropout)(rel_matrix)
        rel_matrix = tf.einsum('Bb,bte->Bte', rel_matrix, V)
        rel_matrix = tf.math.reduce_mean(rel_matrix, axis=1)
        
        cluster_prob = Lambda(lambda x:tf.nn.softmax(x, axis=-1), name='cluster_prob')(
            Dense(k_clusters, use_bias=False, name='{}_prob_dense'.format(self.name))(rel_matrix)
        )
        return cluster_prob
    
    
class GCNStack(ModelBlock):
    def __init__(self, hp, name='GCNStack', **kwargs):
        super(GCNStack, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
    
    def build(self, tensor):
        gcn_layers = self.hp.get('gcn_layers', 2)
        rel_dim = self.hp.get('rel_dim', 4)
        n_stack = self.hp.get('n_stack')
        gcn_pool_size = self.hp.get('gcn_pool_size', 4)
        
        for i in range(gcn_layers):
            adj_matrix = tf.einsum('Bte,bte->tBb', tensor, tensor)
            adj_matrix = tf.math.reduce_mean(adj_matrix, axis=0)
            adj_matrix = sparsemax(adj_matrix, axis=-1)
            adj_matrix = tf.einsum('Bb,bte->Bbte', adj_matrix, tensor)
            
            adj_matrix = MaxPool2D((gcn_pool_size, 1), 1, padding='same', name='{}_maxpooling_{}'.format(self.name, i))(adj_matrix)
            adj_matrix = tf.math.reduce_sum(adj_matrix, axis=1)
            
            tensor = Dense(rel_dim, 
                           use_bias=False, 
                           name = '{}_weight_matrix_{}'.format(self.name, i))(adj_matrix)
            tensor = tf.nn.relu(tensor)
        
        tensor = tf.math.reduce_mean(tensor, axis=1)
        
        cluster_prob = Lambda(lambda x:tf.nn.softmax(x, axis=-1), name='cluster_prob')(
            Dense(n_stack, use_bias=False, name='{}_prob_dense'.format(self.name))(tensor)
        )
        return cluster_prob
    

    
class WavenetEncoder(ModelBlock):
    def __init__(self, hp, name='WavenetEncoder', **kwargs):
        super(WavenetEncoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
        
    def build(self, tensor):
        n_enc_layers = self.hp.get('n_enc_layers', 6)
        n_enc_filters = self.hp.get('n_enc_filters', 16)
        enc_kernel_size = self.hp.get('enc_kernel_size', 2)
        enc_channel_groups = self.hp.get('enc_channel_groups', 1)
        enc_activation = self.hp.get('enc_activation', 'relu')
        enc_l1_regular = self.hp.get('enc_l1_regular', 0)
        
        for i in range(n_enc_layers):
            conv = TimeDistributed(Conv1D(filters = n_enc_filters,
                                          kernel_size = enc_kernel_size,
                                          padding = 'causal',
                                          dilation_rate = enc_kernel_size**i,
                                          groups = enc_channel_groups, 
                                          activation = enc_activation,
                                          kernel_regularizer = L1(enc_l1_regular),
                                          bias_regularizer = L1(enc_l1_regular)),
                                   name = '{}_{}'.format(self.name, i))
            tensor = conv(tensor)
        
        return tensor[:, :, -1:, :]
    

    
class TransformerEncoder(ModelBlock):
    def __init__(self, hp, name='TransformerEncoder', **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    
    def build(self, tensor):
        num_heads = self.hp.get('enc_num_heads', 4)
        k_dim = self.hp.get('enc_k_dim', 4)
        v_dim = self.hp.get('enc_v_dim', 4)
        enc_attn_layers = self.hp.get('enc_attn_layers', 1)
        attn_l1_regular = self.hp.get('attn_l1_regular', 0)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)
        
        for L in range(enc_attn_layers):
            attn_output = FullAttention(num_heads, 
                                        k_dim, 
                                        v_dim, 
                                        attn_l1_regular, 
                                        attn_dropout,
                                        de_stationary = de_stationary,
                                        name = '{}_full_attn_{}'.format(self.name, L))(tensor, tensor, tensor)
                
            if L < enc_attn_layers - 1:
                attn_output = Dense(tensor.shape[-1],
                                    use_bias = False,
                                    name = '{}_reshape_dense_{}'.format(self.name, L))(attn_output)
                tensor += LayerNormalization(name = '{}_LN_{}'.format(self.name, L))( attn_output )
            else:
                return attn_output
    
    
    
class InformerEncoder(ModelBlock):
    def __init__(self, hp, name='InformerEncoder', **kwargs):
        super(InformerEncoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
            
    
    def build(self, tensor):
        num_heads = self.hp.get('enc_num_heads', 4)
        k_dim = self.hp.get('enc_k_dim', 4)
        v_dim = self.hp.get('enc_v_dim', 4)
        factor = self.hp.get('factor', 5)
        enc_attn_layers = self.hp.get('enc_attn_layers', 3)
        attn_l1_regular = self.hp.get('attn_l1_regular', 0)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)
        downconv_size = self.hp.get('downconv_size', 2)
        
        for L in range(enc_attn_layers):
            attn_output = ProbAttention(num_heads, 
                                        k_dim, 
                                        v_dim, 
                                        factor, 
                                        attn_l1_regular, 
                                        attn_dropout,
                                        de_stationary = de_stationary,
                                        name = '{}_prob_attn_{}'.format(self.name, L))(tensor, tensor, tensor)
            attn_output = ZeroPadding2D(((0,0),(downconv_size-1,0)))(attn_output)
            attn_output = Conv2D(filters = attn_output.shape[-1],
                                 kernel_size = (1, downconv_size),
                                 strides = (1, 1),
                                 padding = 'valid',
                                 activation = 'elu',
                                 name = '{}_conv2d_{}'.format(self.name, L))(attn_output)
            attn_output = MaxPool2D(pool_size = (1, downconv_size), 
                                    strides = (1, downconv_size),
                                    padding = 'valid',
                                    name = '{}_maxpool_{}'.format(self.name, L))(attn_output)
            
            if L < enc_attn_layers - 1:
                tensor = LayerNormalization(name = '{}_LN_{}'.format(self.name, L))( attn_output )
            else:
                return attn_output
    

class SciEncoder(ModelBlock):
    def __init__(self, hp, name='SciEncoder', **kwargs):
        super(SciEncoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        

    def build(self, tensor):
        n_splits = self.hp.get('n_splits', 2)
        n_levels = self.hp.get('n_levels', 4)
        n_blocks = self.hp.get('n_blocks', 1)
        hidden_size = self.hp.get('hidden_size', 4)
        kernel_size = self.hp.get('enc_kernel_size', 5)
        groups = self.hp.get('enc_channel_groups', 1)
        regular = self.hp.get('enc_l1_regular', 0)
        dropout = self.hp.get('enc_dropout', 0.5)
        
        stacked = []
        for n in range(n_blocks):
            scinet = SCINet(n_splits, 
                            n_levels, 
                            hidden_size, 
                            kernel_size, 
                            groups, 
                            regular, 
                            dropout,
                            name = '{}_{}'.format(self.name, n))(tensor)
            stacked.append(scinet)
        
        if len(stacked) > 1:
            return Concatenate()(stacked)
        else:
            return stacked[0]
        
        
        
class DenseDecoder(ModelBlock):
    def __init__(self, hp, name='DenseDecoder', **kwargs):
        super(DenseDecoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    def build(self, tensor):
        fcst_len = self.hp.get('fcst_horizon')
        activation = self.hp.get('dec_activation', 'swish')
        
        tensor = EinsumDense(equation = 'btle,lf->btfe',
                             output_shape = (None, fcst_len, tensor.shape[-1]),
                             activation = activation,
                             name = '{}_out'.format(self.name))(tensor)
        return tensor
    

    
class CaCtDecoder(ModelBlock):
    def __init__(self, hp, name='CaCtDecoder', **kwargs):
        super(CaCtDecoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    
    def build(self, tensor):
        fcst_len = self.hp.get('fcst_horizon')
        ca_dim = self.hp.get('ca_dim', 16)
        ca_activation = self.hp.get('ca_activation', 'swish')
        ca_l1_regular = self.hp.get('ca_l1_regular', 0)
        ct_dim = self.hp.get('ct_dim', 4)
        ct_activation = self.hp.get('ct_activation', 'swish')
        ct_l1_regular = self.hp.get('ct_l1_regular', 0)
        
        ca_block = EinsumDense(equation = 'btle,lk->btk',
                               output_shape = (None, ca_dim),
                               activation = ca_activation,
                               kernel_regularizer = L1(ca_l1_regular),
                               name = '{}_ca'.format(self.name))(tensor)
        ca_block = tf.tile( tf.expand_dims(ca_block, axis=2), [1,1,fcst_len,1] )
        
        ct_block = EinsumDense(equation = 'btle,lfk->btfk',
                               output_shape = (None, fcst_len, ct_dim),
                               activation = ct_activation,
                               kernel_regularizer = L1(ct_l1_regular),
                               name = '{}_ct'.format(self.name))(tensor)

        return Concatenate()([ca_block, ct_block])

    
    
class ConvRecoder(ModelBlock):
    def __init__(self, hp, name='ConvRecoder', **kwargs):
        super(ConvRecoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    
    def build(self, tensor):
        n_rec_layers = self.hp.get('n_rec_layers', 3)
        n_rec_filters = self.hp.get('n_rec_filters', 8)
        rec_kernel_size = self.hp.get('rec_kernel_size', 2)
        rec_channel_groups = self.hp.get('rec_channel_groups', 1)
        rec_activation = self.hp.get('rec_activation', 'swish')
        rec_l1_regular = self.hp.get('rec_l1_regular', 0)
        
        if tensor.shape[2] > 1:
            for i in range(n_rec_layers):
                tensor = ZeroPadding2D(((0,0),(0,rec_kernel_size-1)))(tensor)

                tensor = Conv2D(filters = n_rec_filters, 
                                kernel_size = (1, rec_kernel_size),
                                padding = 'valid',
                                groups = rec_channel_groups, 
                                activation = rec_activation,
                                kernel_regularizer = L1(rec_l1_regular),
                                bias_regularizer = L1(rec_l1_regular),
                                name = '{}_conv_{}'.format(self.name, i))(tensor)
        return tensor
    

    
class AttentionRecoder(ModelBlock):
    def __init__(self, hp, name='AttentionRecoder', **kwargs):
        super(AttentionRecoder, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
   
    def build(self, tensor):
        pass
    

    
class MlpOutput(ModelBlock):
    def __init__(self, hp, name='MlpOutput', **kwargs):
        super(MlpOutput, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    def build(self, tensor):
        mlp_dims = self.hp.get('mlp_dims', [16])
        mlp_activation = self.hp.get('mlp_activation', 'swish')
        mlp_l1_regular = self.hp.get('mlp_l1_regular', 0)
        
        for i in range(len(mlp_dims)):
            tensor = Dense(mlp_dims[i],
                           activation = mlp_activation,
                           kernel_regularizer = L1(mlp_l1_regular),
                           name = '{}_mlp_{}'.format(self.name, i))(tensor)
        
        output = Dense(1, use_bias=False, name='{}_output'.format(self.name))(tensor)
        
        return output
    

    
class AttentionOutput(ModelBlock):
    def __init__(self, hp, name='AttentionOutput', **kwargs):
        super(AttentionOutput, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name 
    

    def build(self, tensor):
        num_heads = self.hp.get('dec_num_heads', 8)
        k_dim = self.hp.get('dec_k_dim', 16)
        dec_attn_layers = self.hp.get('dec_attn_layers', 1)
        attn_dropout = self.hp.get('attn_dropout', 0)
        de_stationary = self.hp.get('de_stationary', True)

        for L in range(dec_attn_layers):
            x = FullAttention(num_heads, k_dim, tensor.shape[-1], 
                              attn_dropout = attn_dropout, 
                              de_stationary = de_stationary,
                              name = '{}_output_attn_{}'.format(self.name, L))(tensor, tensor, tensor)
            tensor += LayerNormalization(name = '{}_LN_{}'.format(self.name, L))(x)
        
        output = Dense(1, use_bias=False, name='{}_output'.format(self.name))(tensor)
        
        return output
    

    
class TabnetOutput(ModelBlock):
    def __init__(self, hp, name='TabnetOutput', **kwargs):
        super(TabnetOutput, self).__init__(**kwargs)
        self.hp = hp.copy()
        self.name = name
        
    def build(self, tensor):
        tab_feat_dim = self.hp.get('tab_feat_dim', 16)
        decision_out_dim = self.hp.get('decision_out_dim', 16)
        num_decision_steps = self.hp.get('num_decision_steps', 5)
        relaxation_factor = self.hp.get('relaxation_factor', 1.5)
        sparsity_coef = self.hp.get('sparsity_coef', 1.0e-5)
        norm_type = self.hp.get('norm_type', 'group')
        num_groups = self.hp.get('num_groups', 2)
        virtual_batch_size = self.hp.get('virtual_batch_size', 8)
        
        tabnet = TabNet(feature_dim = tab_feat_dim,
                        output_dim = decision_out_dim,
                        num_features = tensor.shape[-1],
                        num_decision_steps = num_decision_steps,
                        relaxation_factor =  relaxation_factor,
                        sparsity_coefficient = sparsity_coef,
                        norm_type = norm_type,
                        num_groups = num_groups,
                        virtual_batch_size = virtual_batch_size,
                        name = '{}_tabnet'.format(self.name))
        tab_output =  tabnet(tensor)
        
        output = Dense(1, use_bias=False, name='{}_output'.format(self.name))(tab_output)
        
        self.added_loss = tabnet.entropy_loss
        
        return output