import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine import data_adapter


class TSModel(Model):
    def __init__(self, inputs, outputs, evaluates=None,
                 error_bound=False, ema_decay=0.99, epsilon=0.01, loss_threshold=1., **kwargs):
        '''
        evaluates: <tuple> (x_eval, y_eval)
        '''
        super(TSModel, self).__init__(inputs, outputs, **kwargs)
        if evaluates is None:
            self.eval_model = None
        else:
            self.eval_model = TSModel(inputs = evaluates[0], outputs = evaluates[1], trainable = False)

        self.error_bound = error_bound
        if error_bound:
            self.target_model = TSModel(inputs = inputs, outputs = outputs, trainable = True)
            self.ema_decay = ema_decay
            self.epsilon = epsilon
            self.loss_threshold = loss_threshold
        else:
            self.target_model = None

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        del x  # The default implementation does not use `x`.
        return self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )

    def compute_metrics(self, x, y, y_pred, sample_weight):
        del x  # The default implementation does not use `x`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
            Eloss = tf.reduce_mean(loss)
            if (self.error_bound) and (Eloss<=self.loss_threshold):
                y_pred_t = self.target_model(x, training=True)
                loss_t = self.compute_loss(x, y, y_pred_t, sample_weight)
                loss = tf.abs(loss - loss_t + self.epsilon)+loss_t

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        if (self.error_bound) and (Eloss<=self.loss_threshold):
            self.ema_weights_interactive()
        return self.compute_metrics(x, y, y_pred, sample_weight)
    
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
                    if 'ts_model' not in layer.name:
                        print('warnings: The weight of Layer {} is not copied to validation model successfully.'.format(layer.name))

    def ema_weights_interactive(self):
        for layer in self.layers:
            if len(layer.weights) > 0:
                if 'ts_model' not in layer.name:
                    target_weights = self.target_model.get_layer(layer.name).get_weights()
                    source_weights = self.get_layer(layer.name).get_weights()
                    target_weights_new = []
                    for i in range(len(target_weights)):
                        target_weights_new.append(self.ema_decay*target_weights[i] + (1-self.ema_decay)*source_weights[i])
                    self.target_model.get_layer(layer.name).set_weights(target_weights_new)


class ModelBlock(Layer):
    def __init__(self, **kwargs):
        super(ModelBlock, self).__init__(name=kwargs.get('name'))
        self.added_loss = None

    @tf.autograph.experimental.do_not_convert
    def call(self, tensor, **kwargs):
        if len(tensor.shape) == 2:
            tensor_shape = (None, tensor.shape[-1])
        elif len(tensor.shape) == 3:
            tensor_shape = (None, None, tensor.shape[-1])
        elif len(tensor.shape) == 4:
            tensor_shape = (None, None, tensor.shape[-2], tensor.shape[-1])
        else:
            raise ValueError(
                "The rank of {} input tensor should be <= 4, but recieve {}".format(self.name, tensor.shape))

        def inner_build(tensor):
            return self.forward(tensor, **kwargs)

        return inner_build(tensor)

    def forward(self, tensor, **kwargs):
        return tensor