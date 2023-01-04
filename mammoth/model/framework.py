import tensorflow as tf
from tensorflow.keras.layers import Concatenate, ZeroPadding1D, Lambda, Reshape
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import EarlyStopping
from mammoth.networks.normalization import RevIN, InstanceNormalization
from mammoth.utils import scatter_update, compute_normal_mean, compute_normal_std, compute_normal_max, compute_normal_min
from mammoth.losses import PearsonCoef
from mammoth.model.pipline import ModelPipeline
from mammoth.model.nn_blocks import *


class ModelBase(ModelPipeline):
    def __init__(self, input_settings, hyper_params, train_settings, name = None):
        super(ModelBase, self).__init__()
        
        self.input_settings = input_settings
        self.hyper_params = hyper_params
        self.train_settings = train_settings
        self.name = name

        self.strategy = tf.distribute.MirroredStrategy(self.distribute_gpu())
        self._default_train_settings()
        self.added_loss = 0
        self.is_pretrain = False
        self.is_stack = False
        
        self._update_hp()
        self._update_model_name()
        
        
    def _default_train_settings(self):
        _default_dict = {'optimizer': 'adam',
                         'learning_rate': 0.01,
                         'loss_func': 'mae',
                         'loss_weights': None,
                         'weighted_metrics': ['mae', PearsonCoef],
                         'batch_size':16,
                         'epochs':1000,
                         'callbacks':[EarlyStopping(monitor='loss', patience=10)],
                         'verbose': 1,
                         'validation_steps': None,
                         'validation_batch_size': None,
                         'validation_freq': 1,
                         'use_multiprocessing': True,
                         'shuffle': True}
        
        for key, value in _default_dict.items():
            if self.train_settings.get(key) is None:
                self.train_settings[key] = value
    
              
    def _build_model_blocks(self, hp):
        flow_blocks = hp.get('flow_blocks')
        
        self._Encoder = self._init_model_block(flow_blocks.get('Encoder'), hp)
        self._Decoder = self._init_model_block(flow_blocks.get('Decoder'), hp)
        self._Recoder = self._init_model_block(flow_blocks.get('Recoder'), hp)
        self._Output = self._init_model_block(flow_blocks.get('Output'), hp)
    

    def single_tsmodel(self, hp, seq_input, embed_input, masking, remainder, is_fcst):
        fcst_len = hp.get('fcst_horizon')
        perc_horizon = hp.get('perc_horizon')
        
        if perc_horizon > self.input_settings.get('perc_horizon'):
            print("warnings: The tsmodel '{}' has perception horizon {}, which is larger than that defined in data processing.".format(self.name, perc_horizon))
            
        enc_input,\
        dec_input,\
        his_masking,\
        fut_masking = self.seq_input_transform(seq_input, masking, perc_horizon, remainder, is_fcst, hp)
        
        revin = RevIN(name='RevIN_{}'.format(self.built_times))
        enc_scaled, \
        y_mean, \
        y_std = revin(enc_input, his_masking)

        if self._Encoder is not None:
            enc_output = self._Encoder(enc_scaled*his_masking)
            enc_output = enc_output[:, -fut_masking.shape[1]:, :]
        else:
            enc_output = enc_scaled[:, -fut_masking.shape[1]:, :]

        if self._Decoder is not None:
            enc_decoder = self._Decoder(enc_output)
        else:
            enc_decoder = enc_output
        
        outlayer_input = [enc_decoder]
            
        if dec_input is not None:
            if self._Recoder is None:
                outlayer_input.append(dec_input*fut_masking)
            else:
                outlayer_input.append(self._Recoder(dec_input*fut_masking))
        
        if embed_input is not None:
            _, T, F, E = fut_masking.shape
            embed_input = tf.tile( Reshape((1,1,embed_input.shape[-1]))(embed_input), [1, T, F, 1] )
            outlayer_input.append(embed_input)
                
        if len(outlayer_input) > 1:
            outlayer_input = Concatenate()(outlayer_input)
        else:
            outlayer_input = outlayer_input[0]
        
        scaled_output = self._Output(outlayer_input*fut_masking)

        output = revin.denormalize(scaled_output, y_mean, y_std)
        
        return output
 
    def NN(self, seq_input, embed_input, masking, remainder = None, is_fcst = False):
        self._Embedding = self._init_model_block(self.hyper_params['flow_blocks'].get('Embedding'), self.hyper_params)
        self._build_model_blocks(self.hyper_params)
        
        if embed_input is not None:
            if self._Embedding is not None:
                embed_tensor = self._Embedding(embed_input)
            else:
                embed_tensor = embed_input
        else:
            embed_tensor = None

        output = self.single_tsmodel(self.hyper_params, seq_input, embed_tensor, masking, remainder, is_fcst)
        output = Lambda(lambda x:x, name='output')(output)
        
        return output
    
    
    def seq_input_transform(self, tensor, masking, perc_horizon, remainder, is_fcst, hp):
        seq_target = self.input_settings['seq_target']
        dynamic_feat = self.input_settings['dynamic_feat']
        enc_feat = self.input_settings['enc_feat']
        dec_feat = self.input_settings['dec_feat']
        fcst_horizon = self.input_settings['fcst_horizon']
        window_freq = self.input_settings['window_freq']
        norm_feat = hp.get('norm_feat')
        
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
        
        ## Instance Normalization
        if norm_feat is not None:
            if not isinstance(norm_feat, dict):
                raise TypeError("""The dtype of hyper parameter 'norm_feat' should be 'dict', 
                and in the format as follows: {method: [col_1, col_2, ..., col_n]}.
                """)

            for method, cols in norm_feat.items():
                norm_col_idx = [dynamic_feat.index(i) for i in cols]
                tensor_norm_update = InstanceNormalization(axis = 2,
                                                           method = method, 
                                                           name = '{}_norm_on_feature{}'.format(method, self.built_times)
                                                          )(tf.gather(tensor, norm_col_idx, axis=-1),
                                                            masking = masking)
                tensor = scatter_update(tensor, norm_col_idx, tensor_norm_update, axis=-1)
        
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
    
    
    def _update_hp(self):
        if self.hyper_params.get('perc_horizon') is None:
            self.hyper_params['perc_horizon'] = self.input_settings['perc_horizon']
        self.hyper_params['fcst_horizon'] = self.input_settings['fcst_horizon']
        self.hyper_params['enc_feat_dim'] = len(self.input_settings.get('enc_feat'))
        self.hyper_params['dec_feat_dim'] = len(self.input_settings.get('dec_feat'))
        self.hyper_params['embed_feat_dim'] = len(self.input_settings.get('embed_feat'))
        
        
    def _update_model_name(self):
        if self.name is None:
            encoder = self.hyper_params['flow_blocks'].get('Encoder')
            if isinstance(encoder, str):
                self.name = encoder.replace('Encoder', '')
            else:
                self.name = encoder.name.replace('Encoder', '')
    
    
    def _init_model_block(self, mb, hp):
        if mb is None:
            return None
        
        if isinstance(mb, str):
            tmp_block = eval(mb)
        else:
            tmp_block = mb
            
        tmp_block = tmp_block(hp, name='{}{}'.format(tmp_block.__name__, self.built_times))
            
        if tmp_block.added_loss is not None:
            self.added_loss += tmp_block.added_loss
        
        return tmp_block
    
    
    
class ModelStack(ModelBase):
    def __init__(self, input_settings, train_settings, model_lists, stack_hp, is_pretrain = True, pretrain_epoch = 10, name = None):
        self.model_lists = model_lists
        self.hyper_params = stack_hp
        self.n_stack = len(model_lists)
        self.name = name
        self._update_model_name()
        
        super(ModelStack, self).__init__(input_settings, stack_hp, train_settings)
        
        self.is_pretrain = is_pretrain
        self.pretrain_epoch = pretrain_epoch
        
    
    def NN(self, seq_input, embed_input, masking, remainder = None, is_fcst = False):
        self.built_times = 0
        
        if embed_input is not None:
            _Embedding = self.model_lists[0]._Embedding
            if _Embedding is not None:
                embed_tensor = _Embedding(embed_input)
            else:
                embed_tensor = embed_input
        else:
            embed_tensor = None

        seq_scaled_input = self._encoder_y_scaled(seq_input, masking)
        
        stack_name = list(self.hyper_params.keys())[0]
        hp = self.hyper_params[stack_name]
        hp.update({'n_stack':self.n_stack})
        stack = self._init_model_block(stack_name, hp)
        stack_prob = stack(seq_scaled_input*masking)
        
        output = []
        for i, tsm in enumerate(self.model_lists):
            self.built_times = tsm.built_times
            self._build_model_blocks(tsm.hyper_params)
            self.hyper_params['single_model_{}'.format(i)] = tsm.hyper_params
            
            single_output = self.single_tsmodel(tsm.hyper_params, seq_input, embed_tensor, masking, remainder, is_fcst)
            
            output.append( single_output )
            
        stack_prob = Lambda(lambda x:x[0]*x[1], 
                            name = 'stack_prob')([Concatenate()(output), 
                                                  Reshape((1,1,self.n_stack))(stack_prob)])
        output = k.sum( stack_prob, axis=-1, keepdims=True )
            
        output = Lambda(lambda x:x, name='output')(output)
        
        return output
    

    def _encoder_y_scaled(self, seq_input, masking=None):
        y_mean = compute_normal_mean(seq_input[:,:,0:1], masking, axis=1, keepdims=True)
        y_std = compute_normal_std(seq_input[:,:,0:1], masking, axis=1, keepdims=True)

        seq_y_scaled = tf.math.divide_no_nan(seq_input[:,:,0:1]-y_mean, y_std)
        seq_scaled_input = Concatenate()([seq_y_scaled, seq_input[:,:,1:]])
        return seq_scaled_input
    
    
    def _update_model_name(self):
        if self.name is None:
            self.name = 'Stack_' + '_'.join( [i.name for i in self.model_lists] )