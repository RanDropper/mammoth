import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Multiply, Add, TimeDistributed, ZeroPadding1D, ZeroPadding2D, Concatenate, Dropout, LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras import backend as k
from tensorflow.keras.regularizers import L1
import numpy as np


class Interactor(Layer):
    def __init__(self, n_splits, hidden_size, kernel_size, groups, regular, dropout, is_fork, **kwargs):
        super(Interactor, self).__init__(**kwargs)
        self.n_splits = n_splits
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.groups = groups
        self.regular = regular
        self.dropout = dropout
        self.is_fork = is_fork
        
        self._built_from_signature = False
        
        
    def _build_from_signature(self):
        self._built_from_signature = True
        
        self.mul_conv_layers = self._build_inner_conv_sequence()
        self.add_conv_layers = self._build_inner_conv_sequence()
        
        
    def _build_inner_conv_sequence(self):
        if self.is_fork:
            seq = [Sequential([Conv1D(filters=int(self.input_dim * self.hidden_size),
                                      kernel_size=self.kernel_size,
                                      padding='causal',
                                      groups=self.groups,
                                      activation=LeakyReLU(0.01),
                                      kernel_regularizer=L1(self.regular),
                                      bias_regularizer=L1(self.regular)),
                               Dropout(self.dropout),
                               Conv1D(filters=self.input_dim,
                                      kernel_size=3,
                                      padding='causal',
                                      groups=self.groups,
                                      activation='tanh',
                                      kernel_regularizer=L1(self.regular),
                                      bias_regularizer=L1(self.regular))]) for n in range(self.n_splits)]
        else:
            seq = [Sequential([TimeDistributed(Conv1D(filters = int(self.input_dim * self.hidden_size),
                                                      kernel_size = self.kernel_size,
                                                      padding = 'causal',
                                                      groups = self.groups,
                                                      activation = LeakyReLU(0.01),
                                                      kernel_regularizer = L1(self.regular),
                                                      bias_regularizer = L1(self.regular))),
                               Dropout(self.dropout),
                               TimeDistributed(Conv1D(filters = self.input_dim,
                                                      kernel_size = 3,
                                                      padding = 'causal',
                                                      groups = self.groups,
                                                      activation = 'tanh',
                                                      kernel_regularizer = L1(self.regular),
                                                      bias_regularizer = L1(self.regular)))]) for n in range(self.n_splits)]
        return seq
    
    
    def call(self, tensor):
        self.input_dim = tensor.shape[-1]
        if not self._built_from_signature:
            self._build_from_signature()

        if self.is_fork:
            tensor_splits = [tensor[:, i::self.n_splits, :] for i in range(self.n_splits)]
        else:
            tensor_splits = [tensor[:, :, i::self.n_splits, :] for i in range(self.n_splits)]
        idx = [i for i in range(self.n_splits)]
        scale = self.n_splits - 1
            
        mul_tensor_list = []
        for i, tensor in enumerate(tensor_splits):
            multiplier = 1
            for j in idx:
                if j != i: multiplier *= k.exp( self.mul_conv_layers[i](tensor_splits[j]) )
            multiplier = multiplier**(1/scale)
            mul_tensor_list.append( tensor * multiplier )
        
        add_tensor_list = []
        for i, tensor in enumerate(mul_tensor_list):
            adder = 0
            for j in idx:
                if j != i: adder += self.add_conv_layers[i](mul_tensor_list[j])
            adder = adder*(1/scale)
            if i%2 == 0:
                add_tensor_list.append( tensor + adder )
            else:
                add_tensor_list.append( tensor - adder )
        return add_tensor_list


class SCINet(Layer):
    def __init__(self, n_splits, n_levels, hidden_size, kernel_size, groups, regular, dropout, is_fork, **kwargs):
        super(SCINet, self).__init__(**kwargs)
        
        self.n_splits = n_splits
        self.n_levels = n_levels
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.groups = groups
        self.regular = regular
        self.dropout = dropout
        self.is_fork = is_fork
        
        self._built_from_signature = False
    
    
    def get_config(self):
        config = {
            "n_splits": self.n_splits,
            "n_levels": self.n_levels,
            "hidden_size": self.hidden_size,
            "kernel_size": self.kernel_size,
            "groups": self.groups,
            "regular": self.regular,
            "dropout": self.dropout
        }
        base_config = super(SCINet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    
    def _build_from_signature(self):
        self._built_from_signature = True
        
        self.inner_conv_layers = []
        for i in range(self.n_levels):
            self.inner_conv_layers.append( [Interactor(self.n_splits, 
                                                       self.hidden_size, 
                                                       self.kernel_size, 
                                                       self.groups, 
                                                       self.regular, 
                                                       self.dropout,
                                                       self.is_fork,
                                                       name = 'Interactor_{}{}'.format(i,n)) for n in range(self.n_splits**i)] )
            
            
    
    def call(self, tensor):
        if not self._built_from_signature:
            self._build_from_signature()
        
        min_unit_len = self.n_splits ** self.n_levels

        if self.is_fork:
            padding_len = int(np.ceil(tensor.shape[1]/min_unit_len))*min_unit_len - tensor.shape[1]
            tensor_padded = ZeroPadding1D((padding_len, 0))(tensor)
        else:
            padding_len = int(np.ceil(tensor.shape[2]/min_unit_len))*min_unit_len - tensor.shape[2]
            tensor_padded = ZeroPadding2D(((0,0), (padding_len, 0)))(tensor)
        
        sci_input = [tensor_padded]
        for i in range(self.n_levels):
            ret = []
            for j in range(self.n_splits**i):
                ret += self.inner_conv_layers[i][j](sci_input[j])
            sci_input = ret
        
        sci_output = self.rev_split(sci_input)
        if self.is_fork:
            return sci_output[:, -tensor.shape[1]:, :]
        else:
            return sci_output[:, :, -tensor.shape[2]:, :]
    
    
    def rev_split(self, sci_output):
        while len(sci_output) > 1:
            steps = int(len(sci_output)/self.n_splits)
            tmp_sci_reversed = []
            for i in range(steps):
                tmp_sci_reversed.append(Concatenate(-2)(sci_output[i::steps]))
            sci_output = tmp_sci_reversed
        return sci_output[0]