from mammoth.model.framework import ModelBase, ModelStack
from mammoth.data_loader import DatasetBuilder as dsbuilder
from mammoth.utils import tf_ignore_warnings

def single_tsm(input_settings, train_settings, 
               embedding = {'SimpleEmbedding':{}}, 
               encoder = {'WavenetEncoder':{}}, 
               decoder = {'DenseDecoder':{}}, 
               recoder = {'ConvRecoder':{}}, 
               output = {'MlpOutput':{}},
               norm_feat = None,
               perc_horizon = None,
               is_fork=False,
               is_y_affine=True,
               is_feat_affine=True):
    """ The single time series model, without stack.

    Arguments:
        input_settings: A dictionary contains the basic settings for preprocessing. For detail, please refer to 'dsbuilder' function.

        train_settings: A dictionary contains the basic settings for training. Alternative settings are as follows:
            * optimizer -> The optimizer function. It can either be a function in 'tensorflow.keras.optimizers' or 
                            the string representing the function. For detail, please refer to tensorflow.keras tutorial.
                            default: 'adam'.
            * learning_rate -> default: 0.01,
            * loss_func -> It can be a python func(refer to tensorflow.keras tutorial for how to build self-defined Loss Function)
                            ot string. Also it can be any iterable object(e.g. 'list' or 'tuple') containing different losses(e.g. ['mae','mse']).
                            default: 'mae'.
            * loss_weights -> Same parameter in 'tensorflow.keras.model.Model.compile'. default: None.
            * weighted_metrics -> Same parameter in 'tensorflow.keras.model.Model.compile'. default: ['mae', PearsonCoef].
            * batch_size -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: 16.
            * epochs -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1000.
            * callbacks -> Same parameter in 'tensorflow.keras.model.Model.fit'. For detail in defining your own callback functino,
                            please refer to 'tensorflow.keras.callbacks'. default: [EarlyStopping(monitor='loss', patience=10)].
            * verbose -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1.
            * validation_steps -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: None.
            * validation_batch_size -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: None.
            * validation_freq -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1.
            * use_multiprocessing -> Same parameter in 'tensorflow.keras.model.Model.fit'. Note that: 'multiprocessing' is not
                                        available when used GPU numbers < 2 or tensorflow version < 2.5.0. default: True. 
            * shuffle -> Same parameter in 'tensorflow.keras.model.Model.fit'. default: True.
        
        embedding: A dictionary whose key is the method of embedding, normally string type, and value is also a dictionary
                    which contains the hyperparameter for embedding(e.g. {'SimpleEmbedding':{'n_embed_layers':2}}).
                The builtin embedding is called 'SimpleEmbedding'. For customized embedding function, please refer to mammoth.model.nn_blocks.ModelBlock.
                default: {'SimpleEmbedding':{}}.
        
        encoder: A dictionary whose key is the method of time series encoding, normally string type, and value is also a dictionary
                    which contains the hyperparameter for encoding(e.g. {'WavenetEncoder':{'n_enc_filters':8, 'n_enc_layers':6}}).
                There are 4 builtin encoding methods:
            * WavenetEncoder -> https://arxiv.org/pdf/1609.03499v2.pdf
            * TransformerEncoder -> https://arxiv.org/pdf/1706.03762v5.pdf
            * InformerEncoder -> https://arxiv.org/pdf/2012.07436v3.pdf
            * SciEncoder -> https://arxiv.org/pdf/2106.09305v3.pdf
                For customized encoding function, please refer to mammoth.model.nn_blocks.ModelBlock. default: {'WavenetEncoder':{}}.
        
        decoder: A dictionary whose key is the method of time series decoding, normally string type, and value is also a dictionary
                    which contains the hyperparameter for decoding(e.g. {'DenseDecoder':{'dec_activation':'elu'}}).
                There are 2 builtin decoding methods:
            * DenseDecoder -> The most common decoding methods. Using fully connected layers after the encoder part to get decoder part.
            * CaCtDecoder -> https://arxiv.org/pdf/1711.11053v2.pdf
                For customized decoding function, please refer to mammoth.model.nn_blocks.ModelBlock. default: {'DenseDecoder':{}}.

        recoder: A dictionary whose key is the method of time series reverse encoding, normally string type, and value is also a dictionary
                    which contains the hyperparameter for reverse encoding(e.g. {'ConvRecoder':{'n_rec_layers':2}}).
                Reverse encoding is used in multi-horizon forecasting. When you have enough future information and use them as the
                    latent features, you can reverse encode the future information to its relative past time points.
                There are 1 builtin reverse encoding method:
            * ConvRecoder -> Use convolution to encode future information to now reversely.
                For customized reverse encoding function, please refer to mammoth.model.nn_blocks.ModelBlock. default: {'ConvRecoder':{}}.
        
        output: A dictionary whose key is the method of output, normally string type, and value is also a dictionary
                    which contains the hyperparameter for output(e.g. {'MlpOutput':{'mlp_dims':[16,8]}}).
                There are 3 builtin output methods:
            * MlpOutput: The most common output methods. Just use several fully connected layers to get the final output.
            * AttentionOutput: The attention mechanism applied on the feature dimension. Usually used in attention-related models, e.g. Informer.
            * TabnetOutput: Using TabNet for the final regression or classification. For TabNet, please refer to https://arxiv.org/pdf/1908.07442v5.pdf.

        norm_feat: A dictionary whose key is the normalization methods, and value is a iterable object that contains features you want to normalize. 
                    e.g. {'standard':['f1', 'f2']}. There are 4 normalization methods:
            * standard: $\frac{x-avg(x)}{std(x)}$
            * minmax: $\frac{x-min(x)}{max(x)-min(x)}$
            * mean: $\frac{x}{avg(x)}$
            * maxabs: $\frac{x}{max(abs(x))}$
                Note that: the target will be normalized using RevIN automatically. Thus you don't need to declare the target in 'norm_feat'.
                For RevIN, please refer to https://openreview.net/pdf?id=cGDAkQo1C0p. default: None.

        perc_horizon: The length of looking back window. If None, it will be the same with that defined in 'input_settings'.
                    If not None, the value will replace the original 'perc_horizon' defined in 'input_settings'. default: None.

        is_fork: Whether use fork training. Fork training is only available to RNN-based and RNN-based models, e.g. 'WaveNet' and 'SCINet'.
                    Fork training can save memory and speed up training process.

        is_y_affine: Whether add affine weight and bias when apply normalization to 'y'. default: True.

        is_feat_affine: Whether add affine weight and bias when apply normalization to exogenous features. default: True.
    
    Return:
        The untrained model with object type 'ModelBase'
    """
    hyper_params = {'is_fork':is_fork, 'is_y_affine':is_y_affine, 'is_feat_affine':is_feat_affine}
    flow_blocks = {}

    if not input_settings.get('is_dsbuilt', False):
        raise NotImplementedError('The input parameter "input_settings" is not passed through function "dsbuilder". '
                                  '"dsbuilder" function will return the variable "input_settings", which should be used here.')
    if norm_feat is not None:
        hyper_params.update({'norm_feat':norm_feat})
    if perc_horizon is not None:
        hyper_params.update({'perc_horizon':perc_horizon})

    for part_name, part in zip(['Embedding', 'Encoder', 'Decoder', 'Recoder', 'Output'],
                               [embedding, encoder, decoder, recoder, output]):
        if part is not None:
            hyper_params.update( list(part.values())[0] )
            flow_blocks[part_name] = list(part.keys())[0]
        else:
            flow_blocks[part_name] = None

    hyper_params['flow_blocks'] = flow_blocks
    
    tsm = ModelBase(input_settings, hyper_params, train_settings)
    tsm.model = tsm.build()
    
    return tsm


def stack_tsm(model_lists, stack_hp, is_pretrain = True, pretrain_epoch = 10):
    """ The stack of multiple time series models.

    Arguments:
        model_lists: A list of built time series models with object type 'ModelBase'.

        stack_hp: A dictionary whose key is the method of stack, normally string type, and value is also a dictionary
                    which contains the hyperparameter of stack(e.g. {'GCNStack':{'gcn_layers':3}}).
                There are 2 builtin stack methods:
            * GCNStack -> Using GCN to get weight for every single time series model. For GCN, please refer to https://arxiv.org/pdf/1609.02907v4.pdf.
            * AttentionStack -> Using attention to get weight for every single time series model.
                The stacked output is the weighted average of each single time series model.
        
        is_pretrain: Whether to pretrain every single time series model in 'model_lists'. default: True.

        pretrain_epoch: default: 10.
    """
    input_settings, train_settings = model_lists[0].input_settings, model_lists[0].train_settings
    
    tsms = ModelStack(input_settings, train_settings, model_lists, stack_hp, is_pretrain, pretrain_epoch)
    tsms.model = tsms.build()
    
    return tsms 
