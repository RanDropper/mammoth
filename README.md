```python
from mammoth import *
```

## I will build time series data by myself


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

seq_len = 1000
x = np.arange(seq_len)

series_1 = (np.log2(np.arange(seq_len)+1)+2*np.arange(seq_len)**0.5+
            4*np.sin(0.2*(x+1))+5*np.cos(x+12)+7*np.cos(0.04*(x-8))+
            np.random.random(seq_len)*5.5)
series_1 = pd.DataFrame({'series_key':['series_1']*seq_len, 
                         'time_idx':x,
                         'y':series_1,
                         'sin_wx':np.sin(0.2*(x+1)),
                         'cos_wx':np.cos(x+12)})

series_2 = (0.0002*(np.arange(seq_len)-400)**2+40+
            7*np.sin(0.5*(x+1))+10*np.cos(0.1*x)-4*np.cos(0.9*(x-20))+20*np.sin(0.02*(x+4))+
            np.random.random(seq_len)*5)
series_2 = pd.DataFrame({'series_key':['series_2']*seq_len, 
                         'time_idx':x,
                         'y':series_2,
                         'sin_wx':np.sin(0.02*(x+4)),
                         'cos_wx':np.cos(0.9*(x-20))})

series_3 = (np.log2(200-10*np.arange(seq_len)**0.2)**3/10+
            3*np.sin(0.05*(x-9))+2*np.sin(0.01*x)-np.cos(8*(x-10))-3*np.sin(0.2*(x+40))+
            np.random.random(seq_len)*4)
series_3 = pd.DataFrame({'series_key':['series_3']*seq_len, 
                         'time_idx':x,
                         'y':series_3,
                         'sin_wx':np.sin(0.01*x),
                         'cos_wx':np.cos(8*(x-10))})

data = pd.concat([series_1, series_2, series_3], ignore_index=True)
data['key_embed'] = data['series_key'].copy()
```


```python
plt.figure(figsize=(10,4), dpi=100)
for key in data['series_key'].unique():
    plt.plot(data[data['series_key']==key]['y'].values, label=key)
plt.legend()
plt.grid(alpha=0.3)
plt.title('self-made time series')
plt.show()
```


![png](output_3_0.png)


# Build dataset for training and forecasting
***

<font size=4>**dsbuilder** is the function to generate dataset for train, validation and forecast.</font>

***
<font size=3>***input_settings*** is a dictionary contains the basic settings for preprocessing.</font>

Required keys for 'input_settings' are as follows:

* `seq_key` $<List/string/integer>$ The primary key of time series.
* `seq_label` $<List/string/integer>$ The columns to identity the time sequence, such as calendar date or cumulative number.
* `seq_target` $<List/string/integer>$ The target to predict.
* `perc_horizon` $<integer>$ The size of looking back window, which means how long of historical data you will refer to.
* `fcst_horizon` $<integer>$ The prediction length.
        
Alternative keys for 'input_settings' are as follows:
* `enc_feat` $<List>$ The name of features used in the encoding part. default: [].
* `dec_feat` $<List>$ The name of features used in the decoding part. default: [].
* `embed_feat` $<List>$ The name of embedding features. default: [].
* `norm_feat` $<dictionary>$ The features you want to normalize. It is recommended to define 'norm_feat' within 'single_tsm' function. When you define 'norm_feat' here, there won't be trainable weights on mean and standard deviation. default: {}.
* `min_avail_perc_rate` $<float>$ The minimum rate of padding time points verus perception horizon. When exceeded, the data at that time point will be padded. default: 0.25.
* `min_avail_seq_rate` $<float>$ The minimum rate of padding time points verus each time series. Exceeded time series won't be trained. default: 0.5.
* `window_freq` $<integer>$ The strides of sliding window to get time series pieces. default: 1.
* `val_rate` $<float>$ The rate of validation part in train data. If None, no validation data will be built. default: None

***
<font size=3>***train_settings*** is a dictionary contains the basic settings for training.</font>

Alternative settings are as follows:

* `optimizer` The optimizer function. It can either be a function in 'tensorflow.keras.optimizers' or the string representing the function. For detail, please refer to tensorflow.keras tutorial. default: 'adam'.
* `learning_rate` default: 0.01.
* `loss_func` It can be a python func(refer to tensorflow.keras tutorial for how to build self-defined Loss Function or string. Also it can be any iterable object(e.g. 'list' or 'tuple') containing different losses(e.g. ['mae','mse']). default: 'mae'.
* `loss_weights` Same parameter in 'tensorflow.keras.model.Model.compile'. default: None.
* `weighted_metrics` Same parameter in 'tensorflow.keras.model.Model.compile'. default: ['mae', PearsonCoef].
* `batch_size` Same parameter in 'tensorflow.keras.model.Model.fit'. default: 16.
* `epochs` Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1000.
* `callbacks` Same parameter in 'tensorflow.keras.model.Model.fit'. For detail in defining your own callback functino, please refer to 'tensorflow.keras.callbacks'. default: [EarlyStopping(monitor='loss', patience=10)].
* `verbose` Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1.
* `validation_steps` Same parameter in 'tensorflow.keras.model.Model.fit'. default: None.
* `validation_batch_size` Same parameter in 'tensorflow.keras.model.Model.fit'. default: None.
* `validation_freq` Same parameter in 'tensorflow.keras.model.Model.fit'. default: 1.
* `use_multiprocessing` Same parameter in 'tensorflow.keras.model.Model.fit'. Note that: 'multiprocessing' is not available when used GPU numbers < 2 or tensorflow version < 2.5.0. default: True. 
* `shuffle` Same parameter in 'tensorflow.keras.model.Model.fit'. default: True.

***
Callable Arguments:
* `train_data` $<pandas.DataFrame>$ The dataframe used for training.
* `fcst_data` $<pandas.DataFrame>$ The dataframe used for forecasting.
* `embed_data` $<pandas.DataFrame>$ The dataframe used for embedding.

***
Return:
* `train_dataset` $<tensorflow.data.Dataset>$ Dataset for training.
* `val_dataset` $<tensorflow.data.Dataset>$ Dataset for validation.
* `fcst_dataset` $<tensorflow.data.Dataset>$ Dataset for forecasting.
* `input_settings` $<dictionary>$ Updated input_settings.
* `prediction` $<pandas.DataFrame>$ The primary key frame of forecast, including seq_key and seq_label.


```python
from tensorflow.keras.callbacks import EarlyStopping
from mammoth.losses import PearsonCoef
tf_ignore_warnings() ## Ignore warnings

## define input_settings
input_settings = {'seq_key':['series_key'],
                  'seq_label':['time_idx'],
                  'seq_target':['y'],
                  'perc_horizon':128,'fcst_horizon':1,
                  'enc_feat':['sin_wx', 'cos_wx'], 
                  'dec_feat':['sin_wx', 'cos_wx'], 
                  'embed_feat':['key_embed'],
                  'val_rate':0.25, 'window_freq':2}

## define train_settings
train_settings = {'learning_rate':0.01, 'loss_func':'mae', 'batch_size':3,
                  'weighted_metrics':['mae',PearsonCoef],
                  'epochs':200, 'callbacks':[EarlyStopping(monitor='val_loss', patience=10)],
                  'use_multiprocessing':False}

train_data = data[data['time_idx']<=900]
fcst_data = data[data['time_idx']>900]
embed_data = data[input_settings['seq_key']+input_settings['embed_feat']].drop_duplicates()

## Use 'dsbuilder' to create datasets for training, validation and forecasting
train_dataset,\
val_dataset,\
fcst_dataset, \
input_settings, \
prediction = dsbuilder(input_settings)(train_data, fcst_data, embed_data)
```

## Build time series models
***

<font size=4>**signle_tsm** is the function to generate single time series model.</font>

***
<font size=3>**Arguments**</font>

***input_settings*** \
See the tutourials above.

***train_settings*** \
See the tutourials above.

***embedding*** \
A dictionary whose key is the method of embedding, normally string type, and value is also a dictionary which contains the hyperparameter for embedding(e.g. {'SimpleEmbedding':{'n_embed_layers':2}}).
The builtin embedding is called 'SimpleEmbedding'. For customized embedding function, please refer to `mammoth.model.nn_blocks.ModelBlock`. default: {'SimpleEmbedding':{}}.
        
***encoder*** \
A dictionary whose key is the method of time series encoding, normally string type, and value is also a dictionary which contains the hyperparameter for encoding(e.g. {'WavenetEncoder':{'n_enc_filters':8, 'n_enc_layers':6}}).
There are 4 builtin encoding methods:
* `WavenetEncoder` -> https://arxiv.org/pdf/1609.03499v2.pdf
* `TransformerEncoder` -> https://arxiv.org/pdf/1706.03762v5.pdf
* `InformerEncoder` -> https://arxiv.org/pdf/2012.07436v3.pdf
* `SciEncoder` -> https://arxiv.org/pdf/2106.09305v3.pdf

For customized encoding function, please refer to `mammoth.model.nn_blocks.ModelBlock`. default: {'WavenetEncoder':{}}.
        
***decoder*** \
A dictionary whose key is the method of time series decoding, normally string type, and value is also a dictionary which contains the hyperparameter for decoding(e.g. {'DenseDecoder':{'dec_activation':'elu'}}).
There are 2 builtin decoding methods:
* `DenseDecoder` -> The most common decoding methods. Using fully connected layers after the encoder part to get decoder part.
* `CaCtDecoder` -> https://arxiv.org/pdf/1711.11053v2.pdf

For customized decoding function, please refer to mammoth.model.nn_blocks.ModelBlock. default: {'DenseDecoder':{}}.

***recoder*** \
A dictionary whose key is the method of time series reverse encoding, normally string type, and value is also a dictionary which contains the hyperparameter for reverse encoding(e.g. {'ConvRecoder':{'n_rec_layers':2}}). Reverse encoding is used in multi-horizon forecasting. When you have enough future information and use them as the latent features, you can reverse encode the future information to its relative past time points.
There are 1 builtin reverse encoding method:
* `ConvRecoder` -> Use convolution to encode future information to now reversely.

For customized reverse encoding function, please refer to mammoth.model.nn_blocks.ModelBlock. default: {'ConvRecoder':{}}.
        
***output*** \
A dictionary whose key is the method of output, normally string type, and value is also a dictionary which contains the hyperparameter for output(e.g. {'MlpOutput':{'mlp_dims':[16,8]}}).
There are 3 builtin output methods:
* `MlpOutput` -> The most common output methods. Just use several fully connected layers to get the final output.
* `AttentionOutput` -> The attention mechanism applied on the feature dimension. Usually used in attention-related models, e.g. Informer.
* `TabnetOutput` -> Using TabNet for the final regression or classification. For TabNet, please refer to https://arxiv.org/pdf/1908.07442v5.pdf.

***is_fork*** \
Whether use fork training. Fork training is only available to RNN-based and RNN-based models, e.g. 'WaveNet' and 'SCINet'. Fork training can save memory and speed up training process.
        
***norm_feat*** \
A dictionary whose key is the normalization methods, and value is a iterable object that contains features you want to normalize. e.g. {'standard':['f1', 'f2']}. There are 4 normalization methods:
* `standard`: $\frac{x-avg(x)}{std(x)}$
* `minmax`: $\frac{x-min(x)}{max(x)-min(x)}$
* `mean`: $\frac{x}{avg(x)}$
* `maxabs`: $\frac{x}{max(abs(x))}$

Note that: the target will be normalized using `RevIN` automatically. Thus you don't need to declare the target in 'norm_feat'. For `RevIN`, please refer to https://openreview.net/pdf?id=cGDAkQo1C0p. default: None.

***perc_horizon*** \
The length of looking back window. If None, it will be the same with that defined in 'input_settings'. If not None, the value will replace the original 'perc_horizon' defined in 'input_settings'. default: None.

***
<font size=3>**Return**</font>

The untrained model with object type `ModelBase`


```python
## Here take 'Transformer' as an example

transformer = single_tsm(input_settings, train_settings, 
                     embedding = {'SimpleEmbedding':{'embed_out_dim':2}}, 
                     encoder = {'TransformerEncoder':{'enc_num_heads':4, 'enc_k_dim':16, 'enc_v_dim':16,
                                                      'attn_dropout':0.4, 'enc_attn_layers':2}}, 
                     decoder = {'DenseDecoder':{}}, 
                     recoder = None, 
                     output = {'MlpOutput':{'mlp_dims':[16,8]}},
                     norm_feat = {},
                     perc_horizon = 128)
transformer.summary()
```

    Model: "ts_model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 676, 3)]     0           []                               
                                                                                                      
     input_2 (InputLayer)           [(None, 676, 1)]     0           []                               
                                                                                                      
     MovingWindowTransform1 (Moving  ((None, 338, 128, 3  0          ['input_1[0][0]',                
     WindowTransform)               ),                                'input_2[0][0]']                
                                     (None, 338, 1, 2),                                               
                                     (None, 338, 128, 1                                               
                                    ),                                                                
                                     (None, 338, 1, 1))                                               
                                                                                                      
     RevIN_1 (RevIN)                ((None, 338, 128, 3  2           ['MovingWindowTransform1[0][0]', 
                                    ),                                'MovingWindowTransform1[0][2]'] 
                                     (None, 338, 1, 1),                                               
                                     (None, 338, 1, 1))                                               
                                                                                                      
     encoder_input_multiply_masking  (None, 338, 128, 3)  0          ['RevIN_1[0][0]',                
     _1 (Multiply)                                                    'MovingWindowTransform1[0][2]'] 
                                                                                                      
     input_3 (InputLayer)           [(None, 1)]          0           []                               
                                                                                                      
     TransformerEncoder1 (Transform  (None, 338, 128, 16  1218       ['encoder_input_multiply_masking_
     erEncoder)                     )                                1[0][0]']                        
                                                                                                      
     SimpleEmbedding1 (SimpleEmbedd  (None, 2)           2           ['input_3[0][0]']                
     ing)                                                                                             
                                                                                                      
     slice_encoder_output_1 (Lambda  (None, 338, 128, 16  0          ['TransformerEncoder1[0][0]']    
     )                              )                                                                 
                                                                                                      
     expand_embed_input_dims_1 (Res  (None, 1, 1, 2)     0           ['SimpleEmbedding1[0][0]']       
     hape)                                                                                            
                                                                                                      
     DenseDecoder1 (DenseDecoder)   (None, 338, 1, 16)   128         ['slice_encoder_output_1[0][0]'] 
                                                                                                      
     decoder_input_multiply_masking  (None, 338, 1, 2)   0           ['MovingWindowTransform1[0][1]', 
     _1 (Multiply)                                                    'MovingWindowTransform1[0][3]'] 
                                                                                                      
     tf.tile (TFOpLambda)           (None, 338, 1, 2)    0           ['expand_embed_input_dims_1[0][0]
                                                                     ']                               
                                                                                                      
     concat_all_decoder_feats_1 (Co  (None, 338, 1, 20)  0           ['DenseDecoder1[0][0]',          
     ncatenate)                                                       'decoder_input_multiply_masking_
                                                                     1[0][0]',                        
                                                                      'tf.tile[0][0]']                
                                                                                                      
     outlayer_input_multiply_maskin  (None, 338, 1, 20)  0           ['concat_all_decoder_feats_1[0][0
     g_1 (Multiply)                                                  ]',                              
                                                                      'MovingWindowTransform1[0][3]'] 
                                                                                                      
     MlpOutput1 (MlpOutput)         (None, 338, 1, 1)    480         ['outlayer_input_multiply_masking
                                                                     _1[0][0]']                       
                                                                                                      
     denormalize (Denormalize)      (None, 338, 1, 1)    0           ['MlpOutput1[0][0]',             
                                                                      'RevIN_1[0][1]',                
                                                                      'RevIN_1[0][2]']                
                                                                                                      
     output (Lambda)                (None, 338, 1, 1)    0           ['denormalize[0][0]']            
                                                                                                      
     ts_model_1 (TSModel)           (None, 113, 1, 1)    1830        []                               
                                                                                                      
    ==================================================================================================
    Total params: 3,660
    Trainable params: 1,830
    Non-trainable params: 1,830
    __________________________________________________________________________________________________


## Train the time series model


```python
transformer.fit(train_dataset, val_dataset)
```

    Epoch 1/200
    1/1 [==============================] - 8s 8s/step - loss: 7.6503 - mae: 8.4885 - PearsonCoef: 0.3103 - val_loss: 7.4571 - val_mae: 7.4571 - val_PearsonCoef: -0.0188
    Epoch 2/200
    1/1 [==============================] - 6s 6s/step - loss: 7.3762 - mae: 8.1843 - PearsonCoef: 0.3144 - val_loss: 7.3808 - val_mae: 7.3808 - val_PearsonCoef: -0.0052
    Epoch 3/200
    1/1 [==============================] - 6s 6s/step - loss: 7.2025 - mae: 7.9916 - PearsonCoef: 0.3260 - val_loss: 7.3652 - val_mae: 7.3652 - val_PearsonCoef: 0.0159
    Epoch 4/200
    1/1 [==============================] - 6s 6s/step - loss: 7.0810 - mae: 7.8568 - PearsonCoef: 0.3140 - val_loss: 7.2269 - val_mae: 7.2269 - val_PearsonCoef: 0.0322
    Epoch 5/200
    1/1 [==============================] - 6s 6s/step - loss: 6.9312 - mae: 7.6906 - PearsonCoef: 0.3355 - val_loss: 7.1002 - val_mae: 7.1002 - val_PearsonCoef: 0.0623
    Epoch 6/200
    1/1 [==============================] - 6s 6s/step - loss: 6.7382 - mae: 7.4764 - PearsonCoef: 0.3310 - val_loss: 6.8832 - val_mae: 6.8832 - val_PearsonCoef: 0.0938
    Epoch 7/200
    1/1 [==============================] - 6s 6s/step - loss: 6.5011 - mae: 7.2133 - PearsonCoef: 0.3426 - val_loss: 6.6474 - val_mae: 6.6474 - val_PearsonCoef: 0.1506
    Epoch 8/200
    1/1 [==============================] - 6s 6s/step - loss: 6.2829 - mae: 6.9713 - PearsonCoef: 0.3316 - val_loss: 6.2200 - val_mae: 6.2200 - val_PearsonCoef: 0.2468
    Epoch 9/200
    1/1 [==============================] - 6s 6s/step - loss: 5.8373 - mae: 6.4769 - PearsonCoef: 0.3726 - val_loss: 5.8899 - val_mae: 5.8899 - val_PearsonCoef: 0.3499
    Epoch 10/200
    1/1 [==============================] - 6s 6s/step - loss: 5.5480 - mae: 6.1559 - PearsonCoef: 0.3868 - val_loss: 5.7393 - val_mae: 5.7393 - val_PearsonCoef: 0.3719
    Epoch 11/200
    1/1 [==============================] - 6s 6s/step - loss: 5.5395 - mae: 6.1464 - PearsonCoef: 0.3928 - val_loss: 5.7725 - val_mae: 5.7725 - val_PearsonCoef: 0.3868
    Epoch 12/200
    1/1 [==============================] - 6s 6s/step - loss: 5.3829 - mae: 5.9727 - PearsonCoef: 0.4126 - val_loss: 5.7906 - val_mae: 5.7906 - val_PearsonCoef: 0.4140
    Epoch 13/200
    1/1 [==============================] - 6s 6s/step - loss: 5.2073 - mae: 5.7778 - PearsonCoef: 0.4219 - val_loss: 5.8605 - val_mae: 5.8605 - val_PearsonCoef: 0.4428
    Epoch 14/200
    1/1 [==============================] - 6s 6s/step - loss: 5.1557 - mae: 5.7206 - PearsonCoef: 0.4186 - val_loss: 5.8531 - val_mae: 5.8531 - val_PearsonCoef: 0.4794
    Epoch 15/200
    1/1 [==============================] - 6s 6s/step - loss: 5.0451 - mae: 5.5978 - PearsonCoef: 0.4124 - val_loss: 5.6959 - val_mae: 5.6959 - val_PearsonCoef: 0.5207
    Epoch 16/200
    1/1 [==============================] - 6s 6s/step - loss: 4.8981 - mae: 5.4347 - PearsonCoef: 0.4149 - val_loss: 5.5282 - val_mae: 5.5282 - val_PearsonCoef: 0.5409
    Epoch 17/200
    1/1 [==============================] - 6s 6s/step - loss: 4.7836 - mae: 5.3077 - PearsonCoef: 0.4258 - val_loss: 5.4546 - val_mae: 5.4546 - val_PearsonCoef: 0.5419
    Epoch 18/200
    1/1 [==============================] - 6s 6s/step - loss: 4.7509 - mae: 5.2714 - PearsonCoef: 0.4351 - val_loss: 5.4582 - val_mae: 5.4582 - val_PearsonCoef: 0.5431
    Epoch 19/200
    1/1 [==============================] - 6s 6s/step - loss: 4.6582 - mae: 5.1686 - PearsonCoef: 0.4384 - val_loss: 5.4625 - val_mae: 5.4625 - val_PearsonCoef: 0.5507
    Epoch 20/200
    1/1 [==============================] - 6s 6s/step - loss: 4.5439 - mae: 5.0417 - PearsonCoef: 0.4392 - val_loss: 5.4634 - val_mae: 5.4634 - val_PearsonCoef: 0.5608
    Epoch 21/200
    1/1 [==============================] - 6s 6s/step - loss: 4.4682 - mae: 4.9577 - PearsonCoef: 0.4386 - val_loss: 5.4552 - val_mae: 5.4552 - val_PearsonCoef: 0.5710
    Epoch 22/200
    1/1 [==============================] - 6s 6s/step - loss: 4.3918 - mae: 4.8730 - PearsonCoef: 0.4376 - val_loss: 5.3941 - val_mae: 5.3941 - val_PearsonCoef: 0.5807
    Epoch 23/200
    1/1 [==============================] - 6s 6s/step - loss: 4.3065 - mae: 4.7783 - PearsonCoef: 0.4406 - val_loss: 5.2094 - val_mae: 5.2094 - val_PearsonCoef: 0.5914
    Epoch 24/200
    1/1 [==============================] - 6s 6s/step - loss: 4.1661 - mae: 4.6225 - PearsonCoef: 0.4533 - val_loss: 5.0308 - val_mae: 5.0308 - val_PearsonCoef: 0.5975
    Epoch 25/200
    1/1 [==============================] - 6s 6s/step - loss: 4.0893 - mae: 4.5374 - PearsonCoef: 0.4632 - val_loss: 4.9424 - val_mae: 4.9424 - val_PearsonCoef: 0.5984
    Epoch 26/200
    1/1 [==============================] - 6s 6s/step - loss: 4.0177 - mae: 4.4579 - PearsonCoef: 0.4647 - val_loss: 4.8623 - val_mae: 4.8623 - val_PearsonCoef: 0.6073
    Epoch 27/200
    1/1 [==============================] - 6s 6s/step - loss: 3.9245 - mae: 4.3544 - PearsonCoef: 0.4667 - val_loss: 4.7830 - val_mae: 4.7830 - val_PearsonCoef: 0.6246
    Epoch 28/200
    1/1 [==============================] - 6s 6s/step - loss: 3.8442 - mae: 4.2654 - PearsonCoef: 0.4722 - val_loss: 4.7621 - val_mae: 4.7621 - val_PearsonCoef: 0.6362
    Epoch 29/200
    1/1 [==============================] - 6s 6s/step - loss: 3.7406 - mae: 4.1505 - PearsonCoef: 0.4739 - val_loss: 4.7001 - val_mae: 4.7001 - val_PearsonCoef: 0.6462
    Epoch 30/200
    1/1 [==============================] - 6s 6s/step - loss: 3.6302 - mae: 4.0279 - PearsonCoef: 0.4806 - val_loss: 4.6294 - val_mae: 4.6294 - val_PearsonCoef: 0.6587
    Epoch 31/200
    1/1 [==============================] - 6s 6s/step - loss: 3.5195 - mae: 3.9051 - PearsonCoef: 0.4933 - val_loss: 4.5911 - val_mae: 4.5911 - val_PearsonCoef: 0.6710
    Epoch 32/200
    1/1 [==============================] - 6s 6s/step - loss: 3.4220 - mae: 3.7969 - PearsonCoef: 0.5000 - val_loss: 4.5377 - val_mae: 4.5377 - val_PearsonCoef: 0.6837
    Epoch 33/200
    1/1 [==============================] - 6s 6s/step - loss: 3.3389 - mae: 3.7047 - PearsonCoef: 0.5008 - val_loss: 4.4590 - val_mae: 4.4590 - val_PearsonCoef: 0.6968
    Epoch 34/200
    1/1 [==============================] - 6s 6s/step - loss: 3.2653 - mae: 3.6230 - PearsonCoef: 0.5024 - val_loss: 4.3717 - val_mae: 4.3717 - val_PearsonCoef: 0.7041
    Epoch 35/200
    1/1 [==============================] - 6s 6s/step - loss: 3.1890 - mae: 3.5384 - PearsonCoef: 0.5076 - val_loss: 4.2694 - val_mae: 4.2694 - val_PearsonCoef: 0.7134
    Epoch 36/200
    1/1 [==============================] - 6s 6s/step - loss: 3.1267 - mae: 3.4692 - PearsonCoef: 0.5160 - val_loss: 4.1976 - val_mae: 4.1976 - val_PearsonCoef: 0.7235
    Epoch 37/200
    1/1 [==============================] - 6s 6s/step - loss: 3.0422 - mae: 3.3755 - PearsonCoef: 0.5228 - val_loss: 4.1199 - val_mae: 4.1199 - val_PearsonCoef: 0.7390
    Epoch 38/200
    1/1 [==============================] - 6s 6s/step - loss: 2.9052 - mae: 3.2235 - PearsonCoef: 0.5258 - val_loss: 4.0592 - val_mae: 4.0592 - val_PearsonCoef: 0.7468
    Epoch 39/200
    1/1 [==============================] - 6s 6s/step - loss: 2.8176 - mae: 3.1263 - PearsonCoef: 0.5325 - val_loss: 3.9663 - val_mae: 3.9663 - val_PearsonCoef: 0.7543
    Epoch 40/200
    1/1 [==============================] - 6s 6s/step - loss: 2.6949 - mae: 2.9901 - PearsonCoef: 0.5339 - val_loss: 3.8949 - val_mae: 3.8949 - val_PearsonCoef: 0.7529
    Epoch 41/200
    1/1 [==============================] - 6s 6s/step - loss: 2.6718 - mae: 2.9646 - PearsonCoef: 0.5420 - val_loss: 3.8288 - val_mae: 3.8288 - val_PearsonCoef: 0.7659
    Epoch 42/200
    1/1 [==============================] - 6s 6s/step - loss: 2.6677 - mae: 2.9600 - PearsonCoef: 0.5400 - val_loss: 3.7801 - val_mae: 3.7801 - val_PearsonCoef: 0.7658
    Epoch 43/200
    1/1 [==============================] - 6s 6s/step - loss: 2.5505 - mae: 2.8299 - PearsonCoef: 0.5479 - val_loss: 3.8168 - val_mae: 3.8168 - val_PearsonCoef: 0.7655
    Epoch 44/200
    1/1 [==============================] - 6s 6s/step - loss: 2.5856 - mae: 2.8689 - PearsonCoef: 0.5518 - val_loss: 3.9612 - val_mae: 3.9612 - val_PearsonCoef: 0.7693
    Epoch 45/200
    1/1 [==============================] - 6s 6s/step - loss: 2.5997 - mae: 2.8845 - PearsonCoef: 0.5385 - val_loss: 3.7777 - val_mae: 3.7777 - val_PearsonCoef: 0.7715
    Epoch 46/200
    1/1 [==============================] - 6s 6s/step - loss: 2.4198 - mae: 2.6849 - PearsonCoef: 0.5519 - val_loss: 3.6760 - val_mae: 3.6760 - val_PearsonCoef: 0.7749
    Epoch 47/200
    1/1 [==============================] - 6s 6s/step - loss: 2.4336 - mae: 2.7002 - PearsonCoef: 0.5651 - val_loss: 3.7188 - val_mae: 3.7188 - val_PearsonCoef: 0.7796
    Epoch 48/200
    1/1 [==============================] - 6s 6s/step - loss: 2.4168 - mae: 2.6816 - PearsonCoef: 0.5584 - val_loss: 3.6865 - val_mae: 3.6865 - val_PearsonCoef: 0.7834
    Epoch 49/200
    1/1 [==============================] - 6s 6s/step - loss: 2.3271 - mae: 2.5821 - PearsonCoef: 0.5628 - val_loss: 3.6322 - val_mae: 3.6322 - val_PearsonCoef: 0.7831
    Epoch 50/200
    1/1 [==============================] - 6s 6s/step - loss: 2.4231 - mae: 2.6886 - PearsonCoef: 0.5629 - val_loss: 3.7668 - val_mae: 3.7668 - val_PearsonCoef: 0.7857
    Epoch 51/200
    1/1 [==============================] - 6s 6s/step - loss: 2.3961 - mae: 2.6586 - PearsonCoef: 0.5499 - val_loss: 3.5736 - val_mae: 3.5736 - val_PearsonCoef: 0.7910
    Epoch 52/200
    1/1 [==============================] - 6s 6s/step - loss: 2.2218 - mae: 2.4652 - PearsonCoef: 0.5641 - val_loss: 3.4622 - val_mae: 3.4622 - val_PearsonCoef: 0.7912
    Epoch 53/200
    1/1 [==============================] - 6s 6s/step - loss: 2.4140 - mae: 2.6785 - PearsonCoef: 0.5792 - val_loss: 3.6629 - val_mae: 3.6629 - val_PearsonCoef: 0.7862
    Epoch 54/200
    1/1 [==============================] - 6s 6s/step - loss: 2.3168 - mae: 2.5706 - PearsonCoef: 0.5527 - val_loss: 3.6152 - val_mae: 3.6152 - val_PearsonCoef: 0.7897
    Epoch 55/200
    1/1 [==============================] - 6s 6s/step - loss: 2.2476 - mae: 2.4939 - PearsonCoef: 0.5545 - val_loss: 3.4259 - val_mae: 3.4259 - val_PearsonCoef: 0.7988
    Epoch 56/200
    1/1 [==============================] - 6s 6s/step - loss: 2.2477 - mae: 2.4940 - PearsonCoef: 0.5722 - val_loss: 3.4248 - val_mae: 3.4248 - val_PearsonCoef: 0.8043
    Epoch 57/200
    1/1 [==============================] - 7s 7s/step - loss: 2.1348 - mae: 2.3687 - PearsonCoef: 0.5726 - val_loss: 3.5179 - val_mae: 3.5179 - val_PearsonCoef: 0.8059
    Epoch 58/200
    1/1 [==============================] - 6s 6s/step - loss: 2.1836 - mae: 2.4228 - PearsonCoef: 0.5634 - val_loss: 3.4178 - val_mae: 3.4178 - val_PearsonCoef: 0.8094
    Epoch 59/200
    1/1 [==============================] - 6s 6s/step - loss: 2.1067 - mae: 2.3375 - PearsonCoef: 0.5664 - val_loss: 3.2635 - val_mae: 3.2635 - val_PearsonCoef: 0.8108
    Epoch 60/200
    1/1 [==============================] - 6s 6s/step - loss: 2.1376 - mae: 2.3718 - PearsonCoef: 0.5743 - val_loss: 3.2817 - val_mae: 3.2817 - val_PearsonCoef: 0.8123
    Epoch 61/200
    1/1 [==============================] - 6s 6s/step - loss: 2.0936 - mae: 2.3229 - PearsonCoef: 0.5728 - val_loss: 3.4326 - val_mae: 3.4326 - val_PearsonCoef: 0.8126
    Epoch 62/200
    1/1 [==============================] - 6s 6s/step - loss: 2.0918 - mae: 2.3210 - PearsonCoef: 0.5645 - val_loss: 3.4214 - val_mae: 3.4214 - val_PearsonCoef: 0.8134
    Epoch 63/200
    1/1 [==============================] - 6s 6s/step - loss: 2.0514 - mae: 2.2761 - PearsonCoef: 0.5682 - val_loss: 3.2762 - val_mae: 3.2762 - val_PearsonCoef: 0.8148
    Epoch 64/200
    1/1 [==============================] - 6s 6s/step - loss: 2.0468 - mae: 2.2710 - PearsonCoef: 0.5789 - val_loss: 3.2053 - val_mae: 3.2053 - val_PearsonCoef: 0.8151
    Epoch 65/200
    1/1 [==============================] - 6s 6s/step - loss: 2.0077 - mae: 2.2277 - PearsonCoef: 0.5806 - val_loss: 3.2725 - val_mae: 3.2725 - val_PearsonCoef: 0.8157
    Epoch 66/200
    1/1 [==============================] - 7s 7s/step - loss: 1.9567 - mae: 2.1711 - PearsonCoef: 0.5706 - val_loss: 3.3050 - val_mae: 3.3050 - val_PearsonCoef: 0.8180
    Epoch 67/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9884 - mae: 2.2062 - PearsonCoef: 0.5675 - val_loss: 3.2188 - val_mae: 3.2188 - val_PearsonCoef: 0.8229
    Epoch 68/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9254 - mae: 2.1364 - PearsonCoef: 0.5746 - val_loss: 3.1779 - val_mae: 3.1779 - val_PearsonCoef: 0.8258
    Epoch 69/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9599 - mae: 2.1747 - PearsonCoef: 0.5820 - val_loss: 3.2646 - val_mae: 3.2646 - val_PearsonCoef: 0.8263
    Epoch 70/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9223 - mae: 2.1330 - PearsonCoef: 0.5780 - val_loss: 3.3640 - val_mae: 3.3640 - val_PearsonCoef: 0.8234
    Epoch 71/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9368 - mae: 2.1490 - PearsonCoef: 0.5717 - val_loss: 3.2709 - val_mae: 3.2709 - val_PearsonCoef: 0.8242
    Epoch 72/200
    1/1 [==============================] - 6s 6s/step - loss: 1.8783 - mae: 2.0841 - PearsonCoef: 0.5760 - val_loss: 3.1472 - val_mae: 3.1472 - val_PearsonCoef: 0.8253
    Epoch 73/200
    1/1 [==============================] - 6s 6s/step - loss: 1.9155 - mae: 2.1254 - PearsonCoef: 0.5829 - val_loss: 3.2086 - val_mae: 3.2086 - val_PearsonCoef: 0.8249
    Epoch 74/200
    1/1 [==============================] - 6s 6s/step - loss: 1.8184 - mae: 2.0176 - PearsonCoef: 0.5787 - val_loss: 3.2964 - val_mae: 3.2964 - val_PearsonCoef: 0.8253
    Epoch 75/200
    1/1 [==============================] - 7s 7s/step - loss: 1.8644 - mae: 2.0686 - PearsonCoef: 0.5741 - val_loss: 3.2305 - val_mae: 3.2305 - val_PearsonCoef: 0.8297
    Epoch 76/200
    1/1 [==============================] - 6s 6s/step - loss: 1.8167 - mae: 2.0157 - PearsonCoef: 0.5789 - val_loss: 3.1238 - val_mae: 3.1238 - val_PearsonCoef: 0.8337
    Epoch 77/200
    1/1 [==============================] - 6s 6s/step - loss: 1.8159 - mae: 2.0149 - PearsonCoef: 0.5842 - val_loss: 3.1131 - val_mae: 3.1131 - val_PearsonCoef: 0.8358
    Epoch 78/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7806 - mae: 1.9756 - PearsonCoef: 0.5830 - val_loss: 3.2032 - val_mae: 3.2032 - val_PearsonCoef: 0.8365
    Epoch 79/200
    1/1 [==============================] - 7s 7s/step - loss: 1.7785 - mae: 1.9734 - PearsonCoef: 0.5765 - val_loss: 3.1948 - val_mae: 3.1948 - val_PearsonCoef: 0.8395
    Epoch 80/200
    1/1 [==============================] - 7s 7s/step - loss: 1.7761 - mae: 1.9707 - PearsonCoef: 0.5773 - val_loss: 3.0881 - val_mae: 3.0881 - val_PearsonCoef: 0.8446
    Epoch 81/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7429 - mae: 1.9339 - PearsonCoef: 0.5863 - val_loss: 3.0843 - val_mae: 3.0843 - val_PearsonCoef: 0.8472
    Epoch 82/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7643 - mae: 1.9576 - PearsonCoef: 0.5892 - val_loss: 3.1155 - val_mae: 3.1155 - val_PearsonCoef: 0.8474
    Epoch 83/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7065 - mae: 1.8935 - PearsonCoef: 0.5849 - val_loss: 3.1296 - val_mae: 3.1296 - val_PearsonCoef: 0.8468
    Epoch 84/200
    1/1 [==============================] - 7s 7s/step - loss: 1.7217 - mae: 1.9103 - PearsonCoef: 0.5820 - val_loss: 3.0480 - val_mae: 3.0480 - val_PearsonCoef: 0.8491
    Epoch 85/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7219 - mae: 1.9106 - PearsonCoef: 0.5871 - val_loss: 3.0473 - val_mae: 3.0473 - val_PearsonCoef: 0.8507
    Epoch 86/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6917 - mae: 1.8770 - PearsonCoef: 0.5898 - val_loss: 3.1475 - val_mae: 3.1475 - val_PearsonCoef: 0.8503
    Epoch 87/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7083 - mae: 1.8955 - PearsonCoef: 0.5856 - val_loss: 3.1411 - val_mae: 3.1411 - val_PearsonCoef: 0.8531
    Epoch 88/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6978 - mae: 1.8839 - PearsonCoef: 0.5849 - val_loss: 3.0388 - val_mae: 3.0388 - val_PearsonCoef: 0.8566
    Epoch 89/200
    1/1 [==============================] - 6s 6s/step - loss: 1.7008 - mae: 1.8872 - PearsonCoef: 0.5878 - val_loss: 3.0199 - val_mae: 3.0199 - val_PearsonCoef: 0.8571
    Epoch 90/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6522 - mae: 1.8332 - PearsonCoef: 0.5890 - val_loss: 3.0719 - val_mae: 3.0719 - val_PearsonCoef: 0.8562
    Epoch 91/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6771 - mae: 1.8608 - PearsonCoef: 0.5874 - val_loss: 3.0453 - val_mae: 3.0453 - val_PearsonCoef: 0.8593
    Epoch 92/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6636 - mae: 1.8458 - PearsonCoef: 0.5878 - val_loss: 3.0848 - val_mae: 3.0848 - val_PearsonCoef: 0.8590
    Epoch 93/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6619 - mae: 1.8440 - PearsonCoef: 0.5856 - val_loss: 3.0664 - val_mae: 3.0664 - val_PearsonCoef: 0.8584
    Epoch 94/200
    1/1 [==============================] - 7s 7s/step - loss: 1.6283 - mae: 1.8067 - PearsonCoef: 0.5889 - val_loss: 3.0437 - val_mae: 3.0437 - val_PearsonCoef: 0.8585
    Epoch 95/200
    1/1 [==============================] - 7s 7s/step - loss: 1.6100 - mae: 1.7864 - PearsonCoef: 0.5899 - val_loss: 3.0195 - val_mae: 3.0195 - val_PearsonCoef: 0.8611
    Epoch 96/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5977 - mae: 1.7728 - PearsonCoef: 0.5897 - val_loss: 3.0029 - val_mae: 3.0029 - val_PearsonCoef: 0.8644
    Epoch 97/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5997 - mae: 1.7750 - PearsonCoef: 0.5871 - val_loss: 2.9980 - val_mae: 2.9980 - val_PearsonCoef: 0.8665
    Epoch 98/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6228 - mae: 1.8006 - PearsonCoef: 0.5873 - val_loss: 2.9803 - val_mae: 2.9803 - val_PearsonCoef: 0.8674
    Epoch 99/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5964 - mae: 1.7714 - PearsonCoef: 0.5890 - val_loss: 2.9674 - val_mae: 2.9674 - val_PearsonCoef: 0.8675
    Epoch 100/200
    1/1 [==============================] - 6s 6s/step - loss: 1.6067 - mae: 1.7827 - PearsonCoef: 0.5924 - val_loss: 3.0131 - val_mae: 3.0131 - val_PearsonCoef: 0.8666
    Epoch 101/200
    1/1 [==============================] - 7s 7s/step - loss: 1.5763 - mae: 1.7490 - PearsonCoef: 0.5904 - val_loss: 2.9823 - val_mae: 2.9823 - val_PearsonCoef: 0.8687
    Epoch 102/200
    1/1 [==============================] - 7s 7s/step - loss: 1.5966 - mae: 1.7715 - PearsonCoef: 0.5922 - val_loss: 2.9918 - val_mae: 2.9918 - val_PearsonCoef: 0.8686
    Epoch 103/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5613 - mae: 1.7324 - PearsonCoef: 0.5899 - val_loss: 2.9737 - val_mae: 2.9737 - val_PearsonCoef: 0.8685
    Epoch 104/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5658 - mae: 1.7373 - PearsonCoef: 0.5898 - val_loss: 2.9112 - val_mae: 2.9112 - val_PearsonCoef: 0.8699
    Epoch 105/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5705 - mae: 1.7426 - PearsonCoef: 0.5947 - val_loss: 2.9879 - val_mae: 2.9879 - val_PearsonCoef: 0.8680
    Epoch 106/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5374 - mae: 1.7058 - PearsonCoef: 0.5893 - val_loss: 3.0581 - val_mae: 3.0581 - val_PearsonCoef: 0.8652
    Epoch 107/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5781 - mae: 1.7510 - PearsonCoef: 0.5870 - val_loss: 2.9883 - val_mae: 2.9883 - val_PearsonCoef: 0.8664
    Epoch 108/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5532 - mae: 1.7234 - PearsonCoef: 0.5954 - val_loss: 2.9713 - val_mae: 2.9713 - val_PearsonCoef: 0.8660
    Epoch 109/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5431 - mae: 1.7121 - PearsonCoef: 0.5967 - val_loss: 3.0286 - val_mae: 3.0286 - val_PearsonCoef: 0.8645
    Epoch 110/200
    1/1 [==============================] - 7s 7s/step - loss: 1.5399 - mae: 1.7086 - PearsonCoef: 0.5889 - val_loss: 2.9633 - val_mae: 2.9633 - val_PearsonCoef: 0.8679
    Epoch 111/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5421 - mae: 1.7111 - PearsonCoef: 0.5921 - val_loss: 2.9719 - val_mae: 2.9719 - val_PearsonCoef: 0.8692
    Epoch 112/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5357 - mae: 1.7040 - PearsonCoef: 0.5932 - val_loss: 3.0356 - val_mae: 3.0356 - val_PearsonCoef: 0.8680
    Epoch 113/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5329 - mae: 1.7008 - PearsonCoef: 0.5927 - val_loss: 2.9543 - val_mae: 2.9543 - val_PearsonCoef: 0.8703
    Epoch 114/200
    1/1 [==============================] - 6s 6s/step - loss: 1.5230 - mae: 1.6899 - PearsonCoef: 0.5971 - val_loss: 2.9165 - val_mae: 2.9165 - val_PearsonCoef: 0.8711


## Forecast based on fitted model
***

<font size=3>There are two forecasting modes: </font>

1. `multi-horizon` is used when 'fcst_horizon' defined in 'input_settings' is equal to the real future length you want to give prediction. You can call `transformer.predict(fcst_dataset)` to get the prediction.

2. `rolling` is used when 'fcst_horizon' defined in 'input_settings' is smaller to the real future length you want to give prediction. You can call `transformer.predict(fcst_dataset, rolling=True)` to get the prediction.


```python
prediction['pred'] = transformer.predict(fcst_dataset, rolling=True).flatten()
```

    WARNING: AutoGraph could not transform <function Model.make_predict_function.<locals>.predict_function at 0x7f9a32624ef0> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: 'arguments' object has no attribute 'posonlyargs'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING: AutoGraph could not transform <bound method RevIN.call of <mammoth.networks.normalization.RevIN object at 0x7f99068a4710>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module 'gast' has no attribute 'Constant'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    WARNING: AutoGraph could not transform <bound method Denormalize.call of <mammoth.networks.normalization.Denormalize object at 0x7f9a323f5550>> and will run it as-is.
    Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
    Cause: module 'gast' has no attribute 'Constant'
    To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
    1/1 [==============================] - 2s 2s/step
    1/1 [==============================] - 0s 106ms/step
    1/1 [==============================] - 0s 154ms/step
    1/1 [==============================] - 0s 167ms/step
    1/1 [==============================] - 0s 152ms/step
    1/1 [==============================] - 0s 159ms/step
    1/1 [==============================] - 0s 130ms/step
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 108ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 103ms/step
    1/1 [==============================] - 0s 99ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 99ms/step
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 99ms/step
    1/1 [==============================] - 0s 108ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 97ms/step
    1/1 [==============================] - 0s 112ms/step
    1/1 [==============================] - 0s 97ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 108ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 110ms/step
    1/1 [==============================] - 0s 99ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 108ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 108ms/step
    1/1 [==============================] - 0s 113ms/step
    1/1 [==============================] - 0s 103ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 115ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 111ms/step
    1/1 [==============================] - 0s 139ms/step
    1/1 [==============================] - 0s 147ms/step
    1/1 [==============================] - 0s 165ms/step
    1/1 [==============================] - 0s 159ms/step
    1/1 [==============================] - 0s 138ms/step
    1/1 [==============================] - 0s 117ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 105ms/step
    1/1 [==============================] - 0s 97ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 136ms/step
    1/1 [==============================] - 0s 103ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 110ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 106ms/step
    1/1 [==============================] - 0s 106ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 103ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 107ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 116ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 106ms/step
    1/1 [==============================] - 0s 98ms/step
    1/1 [==============================] - 0s 102ms/step
    1/1 [==============================] - 0s 104ms/step
    1/1 [==============================] - 0s 103ms/step
    1/1 [==============================] - 0s 99ms/step
    1/1 [==============================] - 0s 113ms/step
    1/1 [==============================] - 0s 100ms/step
    1/1 [==============================] - 0s 101ms/step
    1/1 [==============================] - 0s 105ms/step



```python
## Check the prediction accuracy
for key in data['series_key'].unique():
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(data[data['series_key']==key]['time_idx'],
             data[data['series_key']==key]['y'], label='Actual')
    plt.plot(prediction[prediction['series_key']==key]['time_idx'],
             prediction[prediction['series_key']==key]['pred'], label='Prediction')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(key)
    plt.show()
```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)



```python

```
