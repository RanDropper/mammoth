import pandas as pd
import numpy as np
import tensorflow as tf
from mammoth.ts_cluster import ts_kmeans
from mammoth.networks.normalization import MinMaxNorm, StandardNorm, MeanNorm, MaxAbsNorm

class DataProcessing:
    def __init__(self, input_settings):
        self.input_settings = input_settings
        self._valid_input_setting_type = {'seq_label':(str, int, list),
                                          'seq_key':(str, int, list),
                                          'seq_target':(str, int, list),
                                          'norm_feat':dict,
                                          'enc_feat':list,
                                          'dec_feat':list,
                                          'embed_feat':list}
        self._default_input_settings()
        self._init_input_check()
        self._string_to_list()
        self.his_data = None
        self.embed_encoder_info = None
        
    
    def _default_input_settings(self):
        _no_default_keys = ['seq_key', 'seq_label', 'seq_target', 'perc_horizon', 'fcst_horizon']
        _default_dict = {'enc_feat':[], 'dec_feat':[], 'embed_feat':[], 'norm_feat':{},
                         'min_avail_perc_rate':0.25, 'min_avail_seq_rate':0.5, 'window_freq':1}
        
        for key in _no_default_keys:
            if self.input_settings.get(key) is None:
                raise ValueError("'{}' must be defined in 'input_settings' dictionary before it is feeded to dataset building functions.".format( key ))
                
        for key, value in _default_dict.items():
            if self.input_settings.get(key) is None:
                self.input_settings[key] = value
        
    
    def _init_input_check(self):
        enc_feat = self.input_settings.get('enc_feat')
        dec_feat = self.input_settings.get('dec_feat')
        embed_feat = self.input_settings.get('embed_feat')
        seq_key = self.input_settings.get('seq_key')
        
        for key, value in self._valid_input_setting_type.items():
            if not isinstance(self.input_settings.get(key), value):
                raise TypeError("The type of key value '{}' should be {}".format( self.input_settings.get(key), value ))
        
        for col in enc_feat:
            if col in seq_key:
                raise ValueError("The col '{}' in 'enc_feat' cannot exist in 'seq_key'.".format(col))
        for col in dec_feat:
            if col in seq_key:
                raise ValueError("The col '{}' in 'dec_feat' cannot exist in 'seq_key'.".format(col))
        for col in embed_feat:
            if col in seq_key:
                raise ValueError("The col '{}' in 'embed_feat' cannot exist in 'seq_key'.".format(col))
                
                
    def _string_to_list(self):
        for col in ['seq_key','seq_label','seq_target']:
            value = self.input_settings.get(col)
            if not isinstance(value, list):
                self.input_settings[col] = [value]
                
                
    def _embed_col_type_init(self, embed_data):
        for col in self.input_settings.get('embed_feat'):
            embed_data[col] = embed_data[col].astype('object')
        return embed_data
            
    
    def _col_type_record(self, data, col):
        self.col_type_ = data[col].dtypes
        self.included_cols = col
        
        
    def _col_type_transform(self, data):
        for col in self.included_cols:
            data[col] = data[col].astype( self.col_type_[col] )
        return data
        
    
    def embed_label_encoder(self, seq_key_frame, embed_data, embed_feat):
        if self.embed_encoder_info is None:
            self.embed_encoder_info = {}
            
            for col in embed_feat:
                if embed_data[col].dtypes in [float, int]:
                    embed_data[col] /= np.max(np.abs(embed_data[col]))
                else:
                    tmp = embed_data[[col]].drop_duplicates()
                    tmp['LabeL'] = np.random.normal(size=tmp.shape[0])
                    self.embed_encoder_info[col] = tmp
        
        embed_data = seq_key_frame.merge(embed_data, how='left')
        for col in embed_feat:
            if embed_data[col].dtypes not in [float, int]:
                embed_data = pd.merge(embed_data, self.embed_encoder_info[col], how='left', on=[col]).fillna({'LabeL':0})
                embed_data[col] = embed_data['LabeL'].copy()
                embed_data.drop(columns=['LabeL'], inplace=True)
        return embed_data

    
    def neighbor_padding(self, data, seq_key, seq_label):
        try:
            data[seq_key]
        except:
            raise KeyError("Input paramerter:'seq_key' should exist in columns of input data.")
        if 'seq_idx' in data.columns:
            data.drop(columns=['seq_idx'], inplace=True)
            
        seq_len = data[seq_label].drop_duplicates().shape[0]
        label_frame = pd.DataFrame({'seq_idx':np.arange(seq_len)})
        label_frame['tmp'] = 1

        seq_key_frame = data[seq_key].drop_duplicates()
        seq_key_frame['tmp'] = 1
        
        data['seq_idx'] = data.groupby(seq_key)[seq_label].transform(lambda x:np.arange(seq_len-x.count(), seq_len))

        full_data = pd.merge(seq_key_frame, label_frame).drop(columns=['tmp']).merge(data, how='left')
        full_data['padding'] = 1.
        full_data.loc[full_data.isna().any(axis=1), 'padding'] = 0.

        return full_data.fillna(0), seq_key_frame.shape[0], seq_len
    
    
    def sequential_padding(self, data, seq_key, seq_label):
        try:
            data[seq_key]
        except:
            raise KeyError("Input paramerter:'seq_key' should exist in columns of input data.")
        if 'seq_idx' in data.columns:
            data.drop(columns=['seq_idx'], inplace=True)
            
        label_frame = data[seq_label].drop_duplicates().reset_index(drop=True)
        seq_len = label_frame.shape[0]
        label_frame['seq_idx'] = np.arange(seq_len)
        label_frame['tmp'] = 1
        
        seq_key_frame = data[seq_key].drop_duplicates()
        seq_key_frame['tmp'] = 1
        
        full_data = pd.merge(seq_key_frame, label_frame).drop(columns=['tmp']).merge(data, how='left', on = seq_key+seq_label)
        full_data['padding'] = 1.
        full_data.loc[full_data.isna().any(axis=1), 'padding'] = 0.

        return full_data.fillna(0), seq_key_frame.shape[0], seq_len
    
    
    def normalization(self, data, seq_key, norm_feat):
        valid_method_list = ['standard', 'minmax', 'mean', 'maxabs']
        for method, cols in norm_feat.items():
            method = method.lower()
            
            if method not in valid_method_list:
                raise ValueError("The 'method' should be within {}, but receive {}.".format(valid_method_list, method))
                
            if method == 'standard':
                Norm = StandardNorm
            elif method == 'minmax':
                Norm = MinMaxNorm
            elif method == 'mean':
                Norm = MeanNorm
            elif method == 'maxabs':
                Norm = MaxAbsNorm
            
            for col in cols:
                data = Norm(data, seq_key, col, indices = (data['padding']==1))
        return data
    
    
    def weight_settings(self, data, seq_key, perc_horizon, min_avail_perc_rate):
        data['avail_rate'] = data.groupby(seq_key)['padding'].transform(lambda x:x.rolling(perc_horizon, 1).sum())/perc_horizon
        data.loc[data['avail_rate']<min_avail_perc_rate, 'padding'] = 0
        
        data['weight'] = data['avail_rate'] * data['padding']
        
        return data
    
    
    def sample_selection(self, data, seq_key, min_avail_seq_rate):
        train_samples = data.groupby(seq_key)['padding'].mean().reset_index()
        train_samples = train_samples[train_samples['padding'] >= min_avail_seq_rate].drop(columns=['padding'])
        data = pd.merge(data, train_samples)
        
        self.input_settings['n_train_samples'] = data[seq_key].drop_duplicates().shape[0]
        
        return data
    
    
    
class DatasetBuilder(DataProcessing):
    """ Transfer table data (DataFrame) to time_series_training_data in tensorflow.dataset format, which is memory-saved.
    
    Class Arguments:
        input_settings: A dictionary containing all settings for the data preprocess. 
        Required keys for 'input_settings' are as follows:
            * seq_key: <List/string/integer> The primary key of time series.
            * seq_label: <List/string/integer> The columns to identity the time sequence, such as calendar date or cumulative number.
            * seq_target: <List/string/integer> The target to predict.
            * perc_horizon: <integer> The size of looking back window, which means how long of historical data you will refer to.
            * fcst_horizon: <integer> The prediction length.
        
         Alternative keys for 'input_settings' are as follows:
            * enc_feat: <List> The features used in the encoding part. default: [].
            * dec_feat: <List> The features used in the decoding part. default: [].
            * embed_feat: <List> The embedding features. default: [].
            * norm_feat: <dictionary> The features you want to normalize. It is recommended to define 'norm_feat' within 'single_tsm'
                        function. When you define 'norm_feat' here, there won't be trainable weights on mean and standard deviation.
                        default: {}.
            * min_avail_perc_rate: <float> The minimum rate of padding time points verus perception horizon. When exceeded, the 
                                            data at that time point will be padded. default: 0.25.
            * min_avail_seq_rate: <float> The minimum rate of padding time points verus each time series. Exceeded time series won't 
                                            be trained. default: 0.5.
            * window_freq: <integer> The strides of sliding window to get time series pieces. default: 1.
            * val_rate: <float> The rate of validation part in train data. If None, no validation data will be built. default: None
    
    Callable Arguments:
        train_data: <pandas.DataFrame> The dataframe used for training.
        fcst_data: <pandas.DataFrame> The dataframe used for forecasting.
        embed_data: <pandas.DataFrame> The dataframe used for embedding.

    Return:
        train_dataset: <tensorflow.data.Dataset> Dataset for training.
        val_dataset: <tensorflow.data.Dataset> Dataset for validation.
        fcst_dataset: <tensorflow.data.Dataset> Dataset for forecasting.
        input_settings: <dictionary> Updated input_settings.
        prediction: <pandas.DataFrame> The primary key frame of forecast, including seq_key and seq_label.
    """
    def __init__(self, input_settings, method='memory', nsplits=None, n_clusters=1):
        super(DatasetBuilder, self).__init__(input_settings)
        self.method = method
        self.nsplits = nsplits
        self.n_clusters = n_clusters
        
        
    def __call__(self, train_data, fcst_data, embed_data = None):
        self._duplicate_checking(train_data, 'input training data')
        self._duplicate_checking(fcst_data, 'input forecasting data')

        embed_init = self._embed_col_type_init(embed_data)
        
        train_data, val_data, embed_data, his_data = self.train_data_process(train_data, embed_init)
        
        fcst_data, fcst_embed_data, prediction = self.fcst_data_process(fcst_data, his_data, embed_init)
        
        if self.method == 'memory':
            if self.n_clusters > 1:
                prediction_ = prediction.copy()
                train_dataset = []
                val_dataset = []
                fcst_dataset = []
                prediction = []
                default = self.cluster['cluster'].value_counts().index[0]
                self.cluster = fcst_data[self.input_settings['seq_key']].drop_duplicates().merge(self.cluster, how='left')
                self.cluster.fillna(default, inplace=True)
                self.input_settings['cluster_info'] = self.cluster
                for clabel in self.cluster['cluster'].unique():
                    key_frame = self.cluster[self.cluster['cluster']==clabel].drop(columns=['cluster'])
                    tmp_train_dataset, \
                    tmp_val_dataset, \
                    tmp_fcst_dataset = self.build_dataset_in_memory(train_data.merge(key_frame),
                                                                    embed_data if embed_data is None else embed_data.merge(key_frame),
                                                                    val_data if val_data is None else val_data.merge(key_frame),
                                                                    fcst_data.merge(key_frame),
                                                                    fcst_embed_data.merge(key_frame),
                                                                    key_frame.shape[0])
                    train_dataset.append(tmp_train_dataset)
                    val_dataset.append(tmp_val_dataset)
                    fcst_dataset.append(tmp_fcst_dataset)
                    prediction.append(prediction_.merge(key_frame))
            else:
                train_dataset, \
                val_dataset, \
                fcst_dataset = self.build_dataset_in_memory(train_data, embed_data, val_data, fcst_data, fcst_embed_data,
                                                            self.input_settings['n_train_samples'])

        self.input_settings['is_dsbuilt'] = True
        return train_dataset, val_dataset, fcst_dataset, self.input_settings, prediction


    def _duplicate_checking(self, data, name):
        seq_key = self.input_settings['seq_key']
        seq_label = self.input_settings['seq_label']

        data['_count_'] = 1
        count = data.groupby(seq_key+seq_label)['_count_'].count().reset_index()
        count = count[count['_count_']>1].drop(columns=['_count_'])

        if count.shape[0] > 0:
            raise NotImplementedError("The {} has duplicated rows: \n {}".format(name, count))


    def build_dataset_in_memory(self, train_data, embed_data, val_data, fcst_data, fcst_embed_data, n_samples):
        train_dataset = self.convert_to_dataset_in_memory(train_data,
                                                          embed_data,
                                                          n_samples,
                                                          self.input_settings['train_seq_len'])

        if self.input_settings.get('val_rate') is not None:
            val_dataset = self.convert_to_dataset_in_memory(val_data,
                                                            embed_data,
                                                            n_samples,
                                                            self.input_settings['val_seq_len'],
                                                            remainder=self.input_settings['remainder'])
        else:
            val_dataset = None

        fcst_dataset = self.convert_to_fcst_dataset_in_memory(fcst_data,
                                                              fcst_embed_data,
                                                              n_samples,
                                                              self.input_settings['fcst_seq_len'])
        return train_dataset, val_dataset, fcst_dataset

        
    def train_data_process(self, data, embed_data):
        seq_key = self.input_settings['seq_key']
        seq_label = self.input_settings['seq_label']
        seq_target = self.input_settings['seq_target']
        enc_feat = self.input_settings['enc_feat']
        dec_feat = self.input_settings['dec_feat']
        embed_feat = self.input_settings['embed_feat']
        norm_feat = self.input_settings['norm_feat']
        perc_horizon = self.input_settings['perc_horizon']
        min_avail_perc_rate = self.input_settings['min_avail_perc_rate']
        min_avail_seq_rate = self.input_settings['min_avail_seq_rate']
        window_freq = self.input_settings['window_freq']
        
        included_cols = np.unique(
            seq_key + seq_label + seq_target + enc_feat + dec_feat
        ).tolist()
        
        self._col_type_record(data, included_cols)
        
        data = data[included_cols].sort_values(seq_key+seq_label)
        
        data, n_samples, seq_len = self.sequential_padding(data, seq_key, seq_label)
        
        if perc_horizon > seq_len:
            raise ValueError("The 'perc_horizon' cannot be larger than the whole length of sequence."+
                             "Now the 'perc_horizon' is {} and the whole length of sequence is {}.".format(perc_horizon, seq_len))
        
        # save historical data for forecasting
        his_data = data.groupby(seq_key).tail(perc_horizon).reset_index(drop=True)
        
        data = self.weight_settings(data, seq_key, perc_horizon, min_avail_perc_rate)
        data = self.sample_selection(data, seq_key, min_avail_seq_rate)

        if self.n_clusters > 1:
            X = data[seq_target].values.reshape((n_samples, seq_len*len(seq_target)))
            self.cluster = data[seq_key].drop_duplicates()
            self.cluster['cluster'] = ts_kmeans(X, self.n_clusters)
        
        if len(embed_feat) > 0:
            embed_data = self.embed_label_encoder(data[seq_key].drop_duplicates(), embed_data.drop_duplicates(seq_key), embed_feat)
        else:
            embed_data = None

        val_rate = self.input_settings.get('val_rate')

        if val_rate is not None:
            remainder = int( seq_len * val_rate)
            self.input_settings['remainder'] = remainder
            self.input_settings['train_seq_len'] = seq_len - remainder
            self.input_settings['val_seq_len'] = remainder + perc_horizon
            self.input_settings['val_len'] = int(np.ceil(remainder/window_freq))
            
            train_data = data[data['seq_idx'] < self.input_settings['train_seq_len']]
            val_data = data[data['seq_idx'] > data['seq_idx'].max()-self.input_settings['val_seq_len']]

            if len(norm_feat) > 0:
                train_data = self.normalization(train_data, seq_key, norm_feat)
                val_data = self.normalization(val_data, seq_key, norm_feat)
                
        else:
            self.input_settings['train_seq_len'] = seq_len
            train_data = data.copy()

            if len(norm_feat) > 0:
                train_data = self.normalization(train_data, seq_key, norm_feat)
                
            val_data = None
        
        return train_data, val_data, embed_data, his_data

    
    
    def fcst_data_process(self, fcst_data, his_data, embed_data):
        seq_key = self.input_settings['seq_key']
        seq_label = self.input_settings['seq_label']
        seq_target = self.input_settings['seq_target']
        enc_feat = self.input_settings['enc_feat']
        dec_feat = self.input_settings['dec_feat']
        embed_feat = self.input_settings['embed_feat']
        norm_feat = self.input_settings['norm_feat']
        
        included_cols = np.unique(
            seq_key + seq_label + seq_target + enc_feat + dec_feat
        ).tolist()
        
        for col in seq_target + enc_feat:
            if col not in fcst_data.columns:
                fcst_data[col] = 0
        
        self._col_type_record(fcst_data, included_cols)
        
        fcst_data = fcst_data[included_cols].sort_values(seq_key+seq_label)
        fcst_data, n_samples, seq_len = self.sequential_padding(fcst_data, seq_key, seq_label)
        self.input_settings['n_fcst_samples'] = n_samples
        self.input_settings['whole_fcst_horizon'] = seq_len
        
        fcst_samples = fcst_data[seq_key].drop_duplicates()
        data = pd.concat([his_data.merge(fcst_samples), fcst_data])
        data, n_samples, seq_len = self.sequential_padding(data, seq_key, seq_label)
        self.input_settings['fcst_seq_len'] = seq_len
        
        if len(embed_feat) > 0:
            embed_data = self.embed_label_encoder(data[seq_key].drop_duplicates(), embed_data.drop_duplicates(seq_key), embed_feat)
        else:
            embed_data = None
        
        if len(norm_feat) > 0:
            data = self.normalization(data, seq_key, norm_feat)
        
        return data, embed_data, fcst_data[seq_key + seq_label]
    
    
    @tf.autograph.experimental.do_not_convert 
    def convert_to_dataset_in_memory(self, data, embed_data, n_samples, seq_len, remainder = None):
        seq_target = self.input_settings['seq_target']
        enc_feat = self.input_settings['enc_feat']
        dec_feat = self.input_settings['dec_feat']
        embed_feat = self.input_settings['embed_feat']
        fcst_horizon = self.input_settings['fcst_horizon']
        window_freq = self.input_settings['window_freq']
        
        dynamic_feat = seq_target + np.unique(enc_feat + dec_feat).tolist()
        self.input_settings['dynamic_feat'] = dynamic_feat
        
        seq_dataset = tf.data.Dataset.from_tensor_slices(
            data[dynamic_feat].values.reshape(n_samples, seq_len, len(dynamic_feat))
        )
        masking_dataset = tf.data.Dataset.from_tensor_slices(
            data['padding'].values.reshape(n_samples, seq_len, 1)
        )
        if len(embed_feat) > 0:
            embed_dataset = tf.data.Dataset.from_tensor_slices(
                embed_data[embed_feat].values.reshape(n_samples, len(embed_feat))
            )
            input_dataset = tf.data.Dataset.zip((seq_dataset, embed_dataset, masking_dataset))
        else:
            input_dataset = tf.data.Dataset.zip((seq_dataset, masking_dataset))
        
        if remainder is not None:
            data = data[data['seq_idx'] >= data['seq_idx'].max()-remainder]
            seq_len = remainder + 1
            
        padding_len = compute_padding_len(seq_len-1, window_freq, fcst_horizon)
        target_dataset = tf.data.Dataset.from_tensor_slices(
            np.concatenate([data[seq_target].values.reshape(n_samples, seq_len, len(seq_target))[:, 1:, :],
                            np.zeros((n_samples, padding_len, len(seq_target)), dtype=float)], axis=1)
        ).map(lambda x:tf.signal.frame(x, fcst_horizon, window_freq, axis=0))
        
        weight_dataset = tf.data.Dataset.from_tensor_slices(
            np.concatenate([data['weight'].values.reshape(n_samples, seq_len, 1)[:, 1:, :],
                            np.zeros((n_samples, padding_len, 1), dtype=float)], axis=1)
        ).map(lambda x:tf.signal.frame(x, fcst_horizon, window_freq, axis=0))

        return tf.data.Dataset.zip((input_dataset, target_dataset, weight_dataset))
    
    
    @tf.autograph.experimental.do_not_convert 
    def convert_to_fcst_dataset_in_memory(self, data, embed_data, n_samples, seq_len):
        seq_target = self.input_settings['seq_target']
        enc_feat = self.input_settings['enc_feat']
        dec_feat = self.input_settings['dec_feat']
        embed_feat = self.input_settings['embed_feat']
        fcst_horizon = self.input_settings['fcst_horizon']
        perc_horizon = self.input_settings['perc_horizon']
        whole_fcst_horizon = self.input_settings['whole_fcst_horizon']
        
        dynamic_feat = seq_target + np.unique(enc_feat + dec_feat).tolist()
        data_numpy = data[dynamic_feat].values.reshape((n_samples, seq_len, len(dynamic_feat)))
        masking_numpy = data['padding'].values.reshape((n_samples, seq_len, 1))
        
        padding_len = int(np.ceil((whole_fcst_horizon/fcst_horizon)))*fcst_horizon - whole_fcst_horizon
        if padding_len > 0:
            data_numpy = np.concatenate([data_numpy, np.zeros((n_samples, padding_len, len(dynamic_feat)), dtype=float)], axis=1)
            masking_numpy = np.concatenate([masking_numpy, np.zeros((n_samples, padding_len, 1), dtype=float)], axis=1)
        
        fcst_dataset = []
        for i in range(perc_horizon, seq_len+padding_len, fcst_horizon):
            tmp_seq_dataset = tf.data.Dataset.from_tensor_slices(data_numpy[:, i-perc_horizon:i+fcst_horizon, :])
            tmp_masking_dataset = tf.data.Dataset.from_tensor_slices(masking_numpy[:, i-perc_horizon:i+fcst_horizon, :])
            if len(embed_feat) > 0:
                tmp_embed_dataset = tf.data.Dataset.from_tensor_slices(
                    embed_data[embed_feat].values.reshape(n_samples, len(embed_feat))
                )
                tmp_dataset = tf.data.Dataset.zip((tmp_seq_dataset, tmp_embed_dataset, tmp_masking_dataset))
            else:
                tmp_dataset = tf.data.Dataset.zip((tmp_seq_dataset, tmp_masking_dataset))
            fcst_dataset.append(tf.data.Dataset.zip((tmp_dataset,)))
        return fcst_dataset
    

def compute_padding_len(seq_len, window_freq, fcst_horizon):
    valid_count = int(np.ceil(seq_len/window_freq))
    remain_count = seq_len - (valid_count-1) * window_freq
    if remain_count < fcst_horizon:
        return fcst_horizon - remain_count
    else:
        return 0