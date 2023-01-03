import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as k
from mammoth.losses import PearsonCoef
import abc
from mammoth.model.tsmodel import TSModel
import numpy as np
from functools import wraps


def counter(func):
    @wraps(func)
    def temp(*args, **kwargs):
        temp.count += 1
        return func(*args, **kwargs)
    temp.count = 0
    return temp


class ModelPipeline(metaclass = abc.ABCMeta):
    def __init__(self):
        self.is_initial = False
        self.init_models = []
        self.model = None
        self.fcst_model = None
        pass
    
    
    @abc.abstractmethod
    def NN(self, inputs, is_fcst = False):
        '''
        The main structure of Neural Network
        '''
        pass
    
    
    @counter
    def build(self, remainder = None):
        remainder = self.input_settings.get('val_len')
        
        optimizer = get_optimizer_func(self.train_settings['optimizer'])
        learning_rate = self.train_settings['learning_rate']
        loss_func = self.train_settings['loss_func']
        loss_weights = self.train_settings['loss_weights']
        weighted_metrics = self.train_settings['weighted_metrics']
        
        self.built_times = self.build.count

        if remainder is None:
            with self.strategy.scope():
                seq_len = self.input_settings.get('train_seq_len')
                inputs = self.model_input(seq_len)

                output = self.NN(**inputs)
                
                model = TSModel(inputs = [i for i in inputs.values() if i is not None], outputs = output)
                model.compile(loss = loss_func, 
                              loss_weights = loss_weights,
                              optimizer = optimizer(lr=learning_rate), 
                              weighted_metrics = weighted_metrics,
                              sample_weight_mode = "temporal",
                              run_eagerly = True)
        else:
            with self.strategy.scope():
                seq_len = self.input_settings.get('train_seq_len')
                train_inputs = self.model_input(seq_len)
                train_output = self.NN(**train_inputs)
                
                seq_len = self.input_settings.get('val_seq_len')
                eval_inputs = self.model_input(seq_len)
                eval_output = self.NN(**eval_inputs, remainder = remainder)
            
                model = TSModel(inputs = [i for i in train_inputs.values() if i is not None], 
                                outputs = train_output,
                                evaluates = ([i for i in eval_inputs.values() if i is not None], eval_output))
                model.compile(loss = loss_func, 
                              loss_weights = loss_weights,
                              optimizer = optimizer(lr=learning_rate), 
                              weighted_metrics = weighted_metrics,
                              sample_weight_mode = "temporal",
                              run_eagerly = True)
                
        if self.added_loss != 0:
            model.add_loss( self.added_loss )
            
        return model
    
    
    def model_input(self, seq_len):
        fdim = len(self.input_settings.get('dynamic_feat'))
        edim = len(self.input_settings.get('embed_feat'))
        batch_size = self.train_settings.get('batch_size')
        
        if self.input_settings.get('n_train_samples') == 1:
            bs = 1
        else:
            bs = None
            
        inputs = {'seq_input':None, 'embed_input':None, 'masking':None}
            
        inputs['seq_input'] = Input((seq_len, fdim), batch_size = bs, dtype=tf.float32)
        inputs['masking'] = Input((seq_len, 1), batch_size = bs, dtype=tf.float32)
        
        if edim > 0:
            inputs['embed_input'] = Input((edim, ), batch_size = bs, dtype=tf.float32)
        
        return inputs
    
    
    def fit(self, train_dataset, validation_dataset=None):
        batch_size = self.train_settings.get('batch_size')
        epochs = self.train_settings.get('epochs')
        callbacks = self.train_settings.get('callbacks')
        verbose = self.train_settings.get('verbose')
        validation_steps = self.train_settings.get('validation_steps')
        validation_batch_size = self.train_settings.get('validation_batch_size')
        validation_freq = self.train_settings.get('validation_freq')
        use_multiprocessing = self.train_settings.get('use_multiprocessing')
        shuffle = self.train_settings.get('shuffle')
        n_train_samples = self.input_settings.get('n_train_samples')
        
        if self.is_pretrain:
            self.pretrain(train_dataset, self.pretrain_epoch)
        if self.is_stack:
            for init_model in self.model_lists:
                self.model = self.copy_weights_from_initial(self.model, init_model.model)
        
        if self.input_settings.get('val_len') is not None:
            train_dataset = train_dataset.shuffle(n_train_samples).batch(batch_size)
            validation_dataset = validation_dataset.batch(batch_size)
           
            self.model.fit(train_dataset,
                           epochs = epochs,
                           callbacks = callbacks,
                           validation_data = validation_dataset,
                           batch_size = batch_size,
                           validation_steps = validation_steps,
                           validation_batch_size = validation_batch_size,
                           validation_freq = validation_freq,
                           shuffle = shuffle,
                           use_multiprocessing = use_multiprocessing,
                           verbose = verbose)
        else:
            train_dataset = train_dataset.shuffle(n_train_samples).batch(batch_size)
                
            self.model.fit(train_dataset,
                           epochs = epochs,
                           callbacks = callbacks,
                           batch_size = batch_size,
                           shuffle = shuffle,
                           use_multiprocessing = use_multiprocessing,
                           verbose = verbose)


    def predict_rolling(self, fcst_dataset, perc_horizon, fcst_horizon, batch_size=None):
        pred = []
        for i, tmp_dataset in enumerate(fcst_dataset):
            if i > 0:
                if len(pred) > 1:
                    former_pred = np.concatenate(pred, axis=1)
                else:
                    former_pred = pred[0]

                seq_input = np.array(list(tmp_dataset.map(lambda x: x[0]).as_numpy_iterator()))
                embed_input = tmp_dataset.map(lambda x: x[1])
                masking_input = tmp_dataset.map(lambda x: x[2])
                seq_input[:, -fcst_horizon * (i + 1):-fcst_horizon, 0:1] = former_pred[:, -min(fcst_horizon * i, perc_horizon):, :]
                seq_input = tf.data.Dataset.from_tensor_slices(seq_input)

                tmp_dataset = tf.data.Dataset.zip((seq_input, embed_input, masking_input))
                tmp_dataset = tf.data.Dataset.zip((tmp_dataset,))

            tmp_pred = self.fcst_model.predict(tmp_dataset.batch(batch_size))
            tmp_pred = tmp_pred.reshape((tmp_pred.shape[0], tmp_pred.shape[2], tmp_pred.shape[3]))
            pred.append(tmp_pred)

        if len(pred) > 1:
            pred = np.concatenate(pred, axis=1)
        else:
            pred = pred[0]
        return pred


    def predict(self, fcst_dataset, rolling = None, batch_size = None):
        perc_horizon = self.input_settings['perc_horizon']
        fcst_horizon = self.input_settings['fcst_horizon']
        whole_fcst_horizon = self.input_settings['whole_fcst_horizon']
        
        if batch_size is None:
            batch_size = self.input_settings.get('n_fcst_samples')
        
        if self.fcst_model is None:
            with self.strategy.scope():
                seq_len = perc_horizon + fcst_horizon
                inputs = self.model_input(seq_len)

                output = self.NN(**inputs, is_fcst = True)

                self.fcst_model = TSModel(inputs = [i for i in inputs.values() if i is not None], outputs = output)
                self.copy_weights_in_fcsting()

        if rolling is True:
            pred = self.predict_rolling(fcst_dataset, perc_horizon, fcst_horizon, batch_size)
        else:
            pred = []
            for tmp_dataset in fcst_dataset:
                tmp_pred = self.fcst_model.predict(tmp_dataset.batch(batch_size))
                tmp_pred = tmp_pred.reshape((tmp_pred.shape[0], tmp_pred.shape[2], tmp_pred.shape[3]))
                pred.append(tmp_pred)
            if len(pred) > 1:
                pred = np.concatenate(pred, axis=1)
            else:
                pred = pred[0]
        
        return pred[:, :whole_fcst_horizon]
    
    
    def distribute_gpu(self):
        n_gpu = len(tf.config.list_physical_devices('GPU'))
        if n_gpu == 0:
            devices = None
        else:
            if self.input_settings.get('n_train_samples') == 1:
                devices = ['/gpu:0']
            else:
                devices = None
        return devices
    
    
    def pretrain(self, train_dataset, pretrain_epoch):
        for i in range(len(self.model_lists)):
            self.model_lists[i].model.fit(train_dataset.batch(self.train_settings['batch_size']),
                                          epochs = pretrain_epoch,
                                          use_multiprocessing = True,
                                          verbose = False)
            print('******PreTraining of ts model "{}" finish.******'.format(self.model_lists[i].name))
            
    
    
    def copy_weights_in_fcsting(self):
        fcst_weight_layers = [layer.name for layer in self.fcst_model.layers if len(layer.weights)>0]
        for layer in self.model.layers:
            if len(layer.weights) > 0:
                try:
                    self.fcst_model.get_layer(layer.name).set_weights( self.model.get_layer(layer.name).get_weights() )
                    fcst_weight_layers.remove(layer.name)
                except ValueError:
                    if 'ts_model' not in layer.name:
                        print('warnings: The weight of Layer {} is not copied successfully.'.format(layer.name))
        if len(fcst_weight_layers) > 0:
            print('warnings: The fcst_model has layers {}, that is not included in trained model.'.format(fcst_weight_layers))

                
    def copy_weights_from_initial(self, m1, m2):
        for layer in m2.layers:
            if len(layer.weights) > 0:
                try:
                    m1.get_layer(layer.name).set_weights( m2.get_layer(layer.name).get_weights() )
                except ValueError:
                    if 'ts_model' not in layer.name:
                        print('warnings: The weight of Layer {} is not copied successfully.'.format(layer.name))
        return m1
    
    
    
class LowerUpperBound(Constraint):
    def __init__(self, init, vol):
        super(LowerUpperBound, self).__init__()
        self.init = init
        self.vol = vol
        
    def __call__(self, w):
        lower = (1 - self.vol)*self.init
        upper = (1 + self.vol)*self.init
        return tf.clip_by_value(w, lower, upper)

    
def get_optimizer_func(optimizer):
    if not isinstance(optimizer, str):
        return optimizer
    
    if optimizer.lower() == 'adadeta':
        optimizer = Adadelta
    elif optimizer.lower() == 'adagrad':
        optimizer = Adagrad
    elif optimizer.lower() == 'adam':
        optimizer = Adam
    elif optimizer.lower() == 'adamax':
        optimizer = Adamax
    elif optimizer.lower() == 'ftrl':
        optimizer = Ftrl
    elif optimizer.lower() == 'nadam':
        optimizer = Nadam
    elif optimizer.lower() == 'rmsprop':
        optimizer = RMSprop
    elif optimizer.lower() == 'sgd':
        optimizer = SGD
    return optimizer
