from tensorflow.keras.layers import Concatenate, ZeroPadding1D, Lambda, Reshape, Multiply
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as k
from tensorflow.keras.callbacks import EarlyStopping
from mammoth.networks.normalization import RevIN
from mammoth.utils import compute_normal_mean, compute_normal_std
from mammoth.losses import PearsonCoef
from mammoth.model.pipline import ModelPipeline
from mammoth.model.nn_blocks import *
from mammoth.model.preprocess import MovingWindowTransform, TSInstanceNormalization, SplitHisFut


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
        norm_feat = hp.get('norm_feat')
        is_fork = hp.get('is_fork', False)

        if not is_fork:
            self._MwTransform = self._init_model_block('MovingWindowTransform', hp)
        else:
            self._MwTransform = None
        if norm_feat is not None:
            self._TSInNorms = []
            for method in norm_feat.keys():
                self._TSInNorms.append(
                    self._init_model_block('TSInstanceNormalization', hp,
                                           name_suffix=method,
                                           norm_method=method)
                )
        else:
            self._TSInNorms = None
        self._SplitHisFut = self._init_model_block('SplitHisFut', hp)
        self._Encoder = self._init_model_block(flow_blocks.get('Encoder'), hp)
        self._Decoder = self._init_model_block(flow_blocks.get('Decoder'), hp)
        self._Recoder = self._init_model_block(flow_blocks.get('Recoder'), hp)
        self._Output = self._init_model_block(flow_blocks.get('Output'), hp)
    

    def single_tsmodel(self, hp, seq_input, embed_input, masking, remainder, is_fcst):
        perc_horizon = hp.get('perc_horizon')
        
        if perc_horizon > self.input_settings.get('perc_horizon'):
            print("warnings: The tsmodel '{}' has perception horizon {}, which is larger than that defined in data processing.".format(self.name, perc_horizon))

        if self._MwTransform is not None:
            seq_input, masking = self._MwTransform(seq_input, masking=masking, is_fcst=is_fcst)

        if self._TSInNorms is not None:
            for TSIN in self._TSInNorms:
                seq_input = TSIN(seq_input, masking=masking)

        enc_input, \
        dec_input, \
        his_masking, \
        fut_masking = self._SplitHisFut(seq_input, masking=masking,
                                        dynamic_feat=self.input_settings['dynamic_feat'],
                                        seq_target=self.input_settings['seq_target'],
                                        enc_feat=self.input_settings['enc_feat'],
                                        dec_feat=self.input_settings['dec_feat'],
                                        is_fcst=is_fcst,
                                        remainder=remainder)
        
        revin = RevIN(hp.get('is_y_affine', True), name='RevIN_{}'.format(self.built_times))
        enc_scaled, y_mean, y_std = revin(enc_input, his_masking)

        Sliced = Lambda(lambda x:x[:, -fut_masking.shape[1]:, :], name='slice_encoder_output_{}'.format(self.built_times))
        enc_scaled = Multiply(
            name='encoder_input_multiply_masking_{}'.format(self.built_times)
        )([enc_scaled, his_masking])
        if self._Encoder is not None:
            enc_output = self._Encoder(enc_scaled, is_fcst = is_fcst)
            enc_output = Sliced(enc_output)
        else:
            enc_output = Sliced(enc_scaled)

        if len(enc_output.shape) == 3:
            enc_output = tf.expand_dims(enc_output, axis=2)

        if self._Decoder is not None:
            enc_decoder = self._Decoder(enc_output, is_fcst = is_fcst)
        else:
            enc_decoder = enc_output
        
        outlayer_input = [enc_decoder]
            
        if dec_input is not None:
            dec_input = Multiply(
                name='decoder_input_multiply_masking_{}'.format(self.built_times)
            )([dec_input, fut_masking])
            if self._Recoder is None:
                outlayer_input.append(dec_input)
            else:
                outlayer_input.append(self._Recoder(dec_input, is_fcst = is_fcst))
        
        if embed_input is not None:
            _, T, F, E = fut_masking.shape
            embed_input = tf.tile(Reshape(
                (1,1,embed_input.shape[-1]), name='expand_embed_input_dims_{}'.format(self.built_times)
            )(embed_input), [1, T, F, 1], name = 'repeat_embed_feat_{}'.format(self.built_times))
            outlayer_input.append(embed_input)
                
        if len(outlayer_input) > 1:
            outlayer_input = Concatenate(
                name='concat_all_decoder_feats_{}'.format(self.built_times)
            )(outlayer_input)
        else:
            outlayer_input = outlayer_input[0]

        outlayer_input = Multiply(
            name='outlayer_input_multiply_masking_{}'.format(self.built_times)
        )([outlayer_input, fut_masking])
        scaled_output = self._Output(outlayer_input, is_fcst = is_fcst)

        output = revin.denormalize(scaled_output, y_mean, y_std)
        
        return output


    @tf.autograph.experimental.do_not_convert
    def NN(self, seq_input, embed_input, masking, remainder = None, is_fcst = False):
        self._Embedding = self._init_model_block(self.hyper_params['flow_blocks'].get('Embedding'), self.hyper_params)
        self._build_model_blocks(self.hyper_params)
        
        if embed_input is not None:
            if self._Embedding is not None:
                embed_tensor = self._Embedding(embed_input, is_fcst = is_fcst)
            else:
                embed_tensor = embed_input
        else:
            embed_tensor = None

        output = self.single_tsmodel(self.hyper_params, seq_input, embed_tensor, masking, remainder, is_fcst)
        output = Lambda(lambda x:x, name='output')(output)
        
        return output
    
    
    def _update_hp(self):
        if self.hyper_params.get('perc_horizon') is None:
            self.hyper_params['perc_horizon'] = self.input_settings['perc_horizon']
        self.hyper_params['fcst_horizon'] = self.input_settings['fcst_horizon']
        self.hyper_params['enc_feat_dim'] = len(self.input_settings.get('enc_feat'))
        self.hyper_params['dec_feat_dim'] = len(self.input_settings.get('dec_feat'))
        self.hyper_params['embed_feat_dim'] = len(self.input_settings.get('embed_feat'))
        self.hyper_params['window_freq'] = self.input_settings['window_freq']

        norm_feat = self.hyper_params.get('norm_feat')
        dynamic_feat = self.input_settings['dynamic_feat']
        if norm_feat is not None:
            if not isinstance(norm_feat, dict):
                raise TypeError("""The dtype of hyper parameter 'norm_feat' should be 'dict', 
                 and in the format as follows: {method: [col_1, col_2, ..., col_n]}.
                 """)
            for method, cols in norm_feat.items():
                norm_col_idx = [dynamic_feat.index(i) for i in cols]
                self.hyper_params['{}_norm_idx'.format(method)] = norm_col_idx


    def _update_model_name(self):
        if self.name is None:
            encoder = self.hyper_params['flow_blocks'].get('Encoder')
            if isinstance(encoder, str):
                self.name = encoder.replace('Encoder', '')
            else:
                self.name = encoder.name.replace('Encoder', '')
    
    
    def _init_model_block(self, mb, hp, name_suffix=None, **kwargs):
        if mb is None:
            return None
        
        if isinstance(mb, str):
            tmp_block = eval(mb)
        else:
            tmp_block = mb

        if name_suffix is not None:
            name = '{}_{}_{}'.format(tmp_block.__name__, name_suffix, self.built_times)
        else:
            name = '{}_{}'.format(tmp_block.__name__, self.built_times)
        tmp_block = tmp_block(hp, name=name, **kwargs)
            
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

    @tf.autograph.experimental.do_not_convert
    def NN(self, seq_input, embed_input, masking, remainder = None, is_fcst = False):
        self.built_times = 0
        
        if embed_input is not None:
            _Embedding = self.model_lists[0]._Embedding
            if _Embedding is not None:
                embed_tensor = _Embedding(embed_input, is_fcst = is_fcst)
            else:
                embed_tensor = embed_input
        else:
            embed_tensor = None

        seq_scaled_input = self._encoder_y_scaled(seq_input, masking)
        
        stack_name = list(self.hyper_params.keys())[0]
        hp = self.hyper_params[stack_name]
        hp.update({'n_stack':self.n_stack})
        stack = self._init_model_block(stack_name, hp)
        stack_prob = stack(seq_scaled_input*masking, is_fcst = is_fcst)
        
        output = []
        for i, tsm in enumerate(self.model_lists):
            self.built_times = tsm.built_times
            self._build_model_blocks(tsm.hyper_params)
            self.hyper_params['single_model_{}'.format(i)] = tsm.hyper_params
            
            single_output = self.single_tsmodel(tsm.hyper_params, seq_input, embed_tensor, masking, remainder, is_fcst)
            
            output.append( single_output )
            
        stack_prob = Lambda(lambda x:x[0]*x[1], 
                            name = 'stack_prob')([Concatenate(name='concat_all_model_outputs')(output),
                                                  Reshape((1,1,self.n_stack))(stack_prob)])
        output = k.sum( stack_prob, axis=-1, keepdims=True, name='stack_weighted_average' )
            
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