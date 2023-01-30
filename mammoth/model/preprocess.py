import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding1D
from mammoth.model.tsmodel import ModelBlock
from mammoth.utils import scatter_update
from mammoth.networks.normalization import InstanceNormalization

class MovingWindowTransform(ModelBlock):
    def __init__(self, hp, name='MovingWindowTransform', **kwargs):
        super(MovingWindowTransform, self).__init__(name=name, **kwargs)
        self.hp = hp.copy()

    def forward(self, tensor, **kwargs):
        perc_horizon = self.hp.get('perc_horizon')
        fcst_horizon = self.hp.get('fcst_horizon')
        window_freq = self.hp.get('window_freq')
        masking = kwargs.get('masking')
        is_fcst = kwargs.get('is_fcst')

        if is_fcst:
            padding_len = perc_horizon + fcst_horizon - tensor.shape[1]
            if padding_len > 0:
                tensor = ZeroPadding1D((padding_len, 0))(tensor)
                masking = ZeroPadding1D((padding_len, 0))(masking)
            tensor = tensor[:, -(perc_horizon + fcst_horizon):, :]
            masking = masking[:, -(perc_horizon + fcst_horizon):, :]
            tensor = tf.expand_dims(tensor, axis=1)
            masking = tf.expand_dims(masking, axis=1)
        else:
            tensor = ZeroPadding1D((perc_horizon - 1, fcst_horizon - 1))(tensor)
            tensor = tf.signal.frame(tensor, perc_horizon + fcst_horizon, window_freq, axis=1)

            masking = ZeroPadding1D((perc_horizon - 1, fcst_horizon - 1))(masking)
            masking = tf.signal.frame(masking, perc_horizon + fcst_horizon, window_freq, axis=1)

        return (tensor, masking)


class TSInstanceNormalization(ModelBlock):
    def __init__(self, hp, name='TSInstanceNormalization', **kwargs):
        super(TSInstanceNormalization, self).__init__(name=name, **kwargs)
        self.hp = hp.copy()
        self.norm_method = kwargs.get('norm_method')
        self.axis = kwargs.get('axis', -2)
        self.norm_idx = hp['{}_norm_idx'.format(self.norm_method)]
        self.affine = hp.get('is_feat_affine', True)
        self._built_from_signature()

    def _built_from_signature(self):
        self.IN = InstanceNormalization(axis=self.axis, method=self.norm_method, affine=self.affine)

    def forward(self, tensor, **kwargs):
        masking = kwargs.get('masking')
        tensor_norm_update = self.IN(tf.gather(tensor, self.norm_idx, axis=-1), masking=masking)
        tensor = scatter_update(tensor, self.norm_idx, tensor_norm_update, axis=-1)
        return tensor


class SplitHisFut(ModelBlock):
    def __init__(self, hp, name='SplitHisFut', **kwargs):
        super(SplitHisFut, self).__init__(name=name, **kwargs)
        self.hp = hp

    def forward(self, tensor, **kwargs):
        masking = kwargs.get('masking')
        dynamic_feat = kwargs.get('dynamic_feat')
        seq_target = kwargs.get('seq_target')
        enc_feat = kwargs.get('enc_feat')
        dec_feat = kwargs.get('dec_feat')
        is_fcst = kwargs.get('is_fcst')
        remainder = kwargs.get('remainder')
        perc_horizon = self.hp.get('perc_horizon')
        fcst_horizon = self.hp.get('fcst_horizon')
        window_freq = self.hp.get('window_freq')

        enc_idx = [dynamic_feat.index(i) for i in seq_target + enc_feat]
        dec_idx = [dynamic_feat.index(i) for i in dec_feat]

        if len(tensor.shape) == 4:
            enc_tensor = tf.gather(tensor[:, :, :perc_horizon, :], enc_idx, axis=-1)
            his_masking = masking[:, :, :perc_horizon, :]
            if remainder is None:
                fut_masking = masking[:, :, -fcst_horizon:, :]
            else:
                fut_masking = masking[:, -remainder:, -fcst_horizon:, :]

            if len(dec_idx) > 0:
                if remainder is None:
                    dec_tensor = tf.gather(tensor[:, :, -fcst_horizon:, :], dec_idx, axis=-1)
                else:
                    dec_tensor = tf.gather(tensor[:, -remainder:, -fcst_horizon:, :], dec_idx, axis=-1)
            else:
                dec_tensor = None
        elif len(tensor.shape) == 3:
            if is_fcst:
                enc_tensor = tf.gather(tensor[:, :perc_horizon, :], enc_idx, axis=-1)
                if len(dec_idx) > 0:
                    dec_tensor = tf.gather(tf.expand_dims(tensor[:, -fcst_horizon:, :], axis=1), dec_idx, axis=-1)
                else:
                    dec_tensor = None
                his_masking = masking[:, :perc_horizon, :]
                fut_masking = tf.expand_dims(masking[:, -fcst_horizon:, :], axis=1)
            else:
                enc_tensor = tf.gather(tensor[:, :-1, :], enc_idx, axis=-1)
                if len(dec_idx) > 0:
                    dec_tensor = tf.gather(
                        tf.signal.frame(tensor[:, 1:, :], fcst_horizon, window_freq, axis=1),
                        dec_idx, axis=-1
                    )
                    if remainder is not None:
                        dec_tensor = dec_tensor[:, -remainder:, :, :]
                else:
                    dec_tensor = None
                his_masking = masking[:, :-1, :]
                fut_masking = tf.signal.frame(masking[:, 1:, :], fcst_horizon, window_freq, axis=1)
                if remainder is not None:
                    fut_masking = fut_masking[:, -remainder:, :, :]

        return (enc_tensor, dec_tensor, his_masking, fut_masking)