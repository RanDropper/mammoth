import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.engine import data_adapter


class TSModel(Model):
    def __init__(self, inputs, outputs, evaluates=None, **kwargs):
        '''
        evaluates: <tuple> (x_eval, y_eval)
        '''
        super(TSModel, self).__init__(inputs, outputs, **kwargs)
            
        if evaluates is None:
            self.eval_model = None
        else:
            self.eval_model = TSModel(inputs = evaluates[0], outputs = evaluates[1], trainable = False)

    
    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        if self.eval_model is None:
            y_pred = self(x, training=False)
        else:
            self.copy_weights_in_training()
            y_pred = self.eval_model(x, training=False)
        
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        
        return {m.name: m.result() for m in self.metrics}
    
    
    def copy_weights_in_training(self):
        for layer in self.layers:
            if len(layer.weights) > 0:
                try:
                    self.eval_model.get_layer(layer.name).set_weights( self.get_layer(layer.name).get_weights() )
                except ValueError:
                    pass