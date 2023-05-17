import tensorflow as tf
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Layer, LayerNormalization, Reshape
from tensorflow.keras.regularizers import L1


class ResidualBlock(Layer):
    def __init__(self, unit, dropout, regular, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.unit = unit
        self.dropout = dropout
        self.regular = regular

    def build(self, input_shape):
        self.relu_dense = Dense(self.unit, activation='relu',
                                kernel_regularizer = L1(self.regular),
                                bias_regularizer = L1(self.regular))
        self.res_dense = Dense(self.unit,
                               kernel_regularizer=L1(self.regular),
                               bias_regularizer=L1(self.regular))
        self.lin_dense = Dense(self.unit,
                               kernel_regularizer=L1(self.regular),
                               bias_regularizer=L1(self.regular))
        self.LN = LayerNormalization()

    def call(self, tensor):
        residual = self.res_dense(tensor)

        tensor = self.relu_dense(tensor)
        tensor = self.lin_dense(tensor)
        tensor = Dropout(self.dropout)(tensor)

        return self.LN(tensor + residual)


class TiDE(Layer):
    def __init__(self, feat_proj_unit, n_enc_layers, enc_unit, n_dec_layers, dec_unit,
                 out_unit, dropout=0., regular=0., **kwargs):
        super(TiDE, self).__init__(**kwargs)
        self.feat_proj_unit = feat_proj_unit
        self.n_enc_layers = n_enc_layers
        self.enc_unit = enc_unit
        self.n_dec_layers = n_dec_layers
        self.dec_unit = dec_unit
        self.out_unit = out_unit
        self.dropout = dropout
        self.regular = regular

    def build(self, input_shape):
        B,T,L,E = input_shape
        self.feat_projection = ResidualBlock(self.feat_proj_unit, self.dropout, self.regular, name='Feat_Projection')
        self.encoder = Sequential([ResidualBlock(self.enc_unit, self.dropout, self.regular, name='Encoder_{}'.format(i))
                                   for i in range(self.n_enc_layers)])
        self.decoder = Sequential([ResidualBlock(L*self.dec_unit, self.dropout, self.regular, name='Decoder_{}'.format(i))
                                   for i in range(self.n_dec_layers)])
        self.temp_decoder = ResidualBlock(self.out_unit, self.dropout, self.regular, name='Temporal_Decoder')
        self.res_dense = Dense(self.out_unit,
                               kernel_regularizer=L1(self.regular),
                               bias_regularizer=L1(self.regular),
                               name = 'residual_y_dense')

    def call(self, y, attr, feat):
        feat = self.feat_projection(feat)
        enc_input = Concatenate()([Reshape((y.shape[1], y.shape[2]*y.shape[3]))(y),
                                   Reshape((attr.shape[1], attr.shape[2]*attr.shape[3]))(attr),
                                   Reshape((feat.shape[1], feat.shape[2]*feat.shape[3]))(feat)])
        e_i = self.encoder(enc_input)
        g_i = self.decoder(e_i)

        d_i = Reshape((feat.shape[1], feat.shape[2], self.dec_unit))(g_i)

        td = self.temp_decoder(Concatenate()([d_i, feat]))
        y_res = self.res_dense(y)

        return td+y_res


