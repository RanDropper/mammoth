import tensorflow as tf
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.initializers import Ones, Zeros
from tensorflow.keras import backend as k
from mammoth.utils import compute_normal_mean, compute_normal_std, compute_normal_max, compute_normal_min
import numpy as np
import pandas as pd


class RevIN(Layer):
    """
    Please refer to the thesis called: 
    "REVERSIBLE INSTANCE NORMALIZATION FOR ACCURATE TIME-SERIES FORECASTING AGAINST DISTRIBUTION SHIFT"
    However, here RevIN is only applied to target.
    """ 
    def __init__(self, **kwargs):
        super(RevIN, self).__init__(**kwargs)
    
    
    def build(self, input_shape):
        self.affine_weight = self.add_weight(
            name = 'affine_weight',
            shape = (1,),
            initializer = Ones(),
            trainable = True,
            dtype=self.dtype
        )
        self.affine_bias = self.add_weight(
            name = 'affine_bias',
            shape = (1,),
            initializer = Zeros(),
            trainable = True,
            dtype=self.dtype
        )
        
    
    def call(self, tensor, masking=None):
        y_mean = compute_normal_mean(tensor[:,:,:,0:1], masking, axis=2, keepdims=True)
        y_std = compute_normal_std(tensor[:,:,:,0:1], masking, axis=2, keepdims=True)
        
        y_scaled = tf.math.divide_no_nan(tensor[:,:,:,0:1]-y_mean, y_std)
        y_scaled = self.affine_weight * y_scaled + self.affine_bias
        scaled_tensor = Concatenate()([y_scaled, tensor[:,:,:,1:]])

        return scaled_tensor, y_mean, y_std
    
    
    def denormalize(self, tensor, y_mean, y_std):
        remainder = tensor.shape[1]
        y_mean = y_mean[:, -remainder:, :, :]
        y_std = y_std[:, -remainder:, :]

        return tf.math.divide_no_nan(tensor-self.affine_bias, self.affine_weight) * y_std + y_mean
    

class InstanceNormalization(Layer):
    """
    The normalization along the second dimension (time dimension). Also I add trainable affine weights in this layer.
    """
    def __init__(self, axis=2, method="standard", affine=True, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.method = method
        self.affine = affine
        
        self._method_check()
        
    def _method_check(self):
        self.valid_method_list = ['standard', 'minmax', 'mean', 'maxabs']
        
        if not isinstance(self.method, str):
            raise TypeError("The input parameter 'method' should be string type.")
            
        self.method = self.method.lower()
        if self.method not in self.valid_method_list:
            raise ValueError("The 'method' should be within {}, but receive {}.".format(self.valid_method_list, self.method))
            
    
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        num_feat = input_shape[-1]
        
        self.affine_weight = self.add_weight(
            name = 'affine_weight',
            shape = (num_feat,),
            initializer = Ones(),
            trainable = True,
            dtype=self.dtype
        )
        self.affine_bias = self.add_weight(
            name = 'affine_bias',
            shape = (num_feat,),
            initializer = Zeros(),
            trainable = True,
            dtype=self.dtype
        )
        
    
    def call(self, tensor, ref_tensor = None, masking = None):
        """
        tensor: The tensor you would apply normalization to.
        
        ref_tensor: The tensor used to compute the norm scaler. If None, ref_tensor is equal to tensor.
        """
        if ref_tensor is None:
            ref_tensor = tensor
        
        rank = len(tensor.shape)
        
        if self.method == 'standard':
            mean = compute_normal_mean(ref_tensor, masking, axis=self.axis, keepdims=True)
            std = compute_normal_std(ref_tensor, masking, axis=self.axis, keepdims=True)

            while len(mean.shape) < rank:
                mean = tf.expand_dims(mean, axis=1)
                std = tf.expand_dims(std, axis=1)
            tensor = tf.math.divide_no_nan(tensor-mean, std)
        
        elif self.method == 'minmax':
            Min = compute_normal_min(ref_tensor, masking, axis=self.axis, keepdims=True)
            Max = compute_normal_max(ref_tensor, masking, axis=self.axis, keepdims=True)
            
            while len(Min.shape) < rank:
                Min = tf.expand_dims(Min, axis=1)
                Max = tf.expand_dims(Max, axis=1)
            tensor = tf.math.divide_no_nan(tensor - Min, Max - Min)
        
        elif self.method == 'mean':
            mean = compute_normal_mean(ref_tensor, masking, axis=self.axis, keepdims=True)
                
            while len(mean.shape) < rank:
                mean = tf.expand_dims(mean, axis=1)
            tensor = tf.math.divide_no_nan(tensor, mean)
            
        elif self.method == 'maxabs':
            Max = compute_normal_max(tf.abs(ref_tensor), masking, axis=self.axis, keepdims=True)
            
            while len(Max.shape) < rank:
                Max = tf.expand_dims(Max, axis=1)
            tensor = tf.math.divide_no_nan(tensor, Max)
            
        if self.affine:
            tensor = tensor*self.affine_weight + self.affine_bias
            
        return tensor
    

def MinMaxNorm(data, key, col, indices=None):
    if indices is None:
        scaler = data.groupby(key)[col].agg(['min','max']).reset_index()
    else:
        scaler = data[indices].groupby(key)[col].agg(['min','max']).reset_index()
    
    data = pd.merge(data, scaler, how='left').fillna({'min':0, 'max':1})
    data[col] = (data[col]-data['min'])/np.maximum(data['max']-data['col'], 1.0e-6)
    return data.drop(columns=['min','max'])


def StandardNorm(data, key, col, indices=None):
    if indices is None:
        scaler = data.groupby(key)[col].agg(['mean','std']).reset_index()
    else:
        scaler = data[indices].groupby(key)[col].agg(['mean','std']).reset_index()
    
    data = pd.merge(data, scaler, how='left').fillna({'mean':0, 'std':1})
    data[col] = (data[col]-data['mean'])/np.maximum(data['std'], 1.0e-6)
    return data.drop(columns=['mean','std'])


def MaxAbsNorm(data, key, col, indices=None):
    if indices is None:
        scaler = data.groupby(key)[col].apply(lambda x:np.abs(x).max()).reset_index().rename(columns={col:'max'})
    else:
        scaler = data[indices].groupby(key)[col].apply(lambda x:np.abs(x).max()).reset_index().rename(columns={col:'max'})
    
    data = pd.merge(data, scaler, how='left').fillna({'max':1})
    data[col] = data[col]/np.maximum(data['max'], 1.0e-6)
    return data.drop(columns=['max'])


def MeanNorm(data, key, col, indices=None):
    if indices is None:
        scaler = data.groupby(key)[col].agg(['mean']).reset_index()
    else:
        scaler = data[indices].groupby(key)[col].agg(['mean']).reset_index()
    
    data = pd.merge(data, scaler, how='left').fillna({'mean':1})
    data[col] = data[col]/np.maximum(data['mean'], 1.0e-6)
    return data.drop(columns=['mean'])