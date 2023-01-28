import numpy as np
from scipy import signal
from scipy import special as ss
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.keras.initializers import GlorotNormal, Zeros
try:
    from tensorflow.keras.layers import EinsumDense
except ImportError:
    from tensorflow.keras.layers.experimental import EinsumDense


def tf_complex_einsum(equation, var1, var2):
    var1_real = tf.math.real(var1)
    var1_imag = tf.math.imag(var1)
    out_real = tf.einsum(equation, var1_real, var2)
    out_imag = tf.einsum(equation, var1_imag, var2)
    return tf.complex(out_real, out_imag)


def transition(measure, N, **measure_args):
    """ A, B transition matrices for different measures.
    measure: the type of measure
      legt - Legendre (translated)
      legs - Legendre (scaled)
      glagt - generalized Laguerre (translated)
      lagt, tlagt - previous versions of (tilted) Laguerre with slightly different normalization
    """
    # Laguerre (translated)
    if measure == 'lagt':
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    if measure == 'tlagt':
        # beta = 1 corresponds to no tilt
        b = measure_args.get('beta', 1.0)
        A = (1. - b) / 2 * np.eye(N) - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    # Generalized Laguerre
    # alpha 0, beta small is most stable (limits to the 'lagt' measure)
    # alpha 0, beta 1 has transition matrix A = [lower triangular 1]
    if measure == 'glagt':
        alpha = measure_args.get('alpha', 0.0)
        beta = measure_args.get('beta', 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]

        L = np.exp(.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1)))
        A = (1. / L[:, None]) * A * L[None, :]
        B = (1. / L[:, None]) * B * np.exp(-.5 * ss.gammaln(1 - alpha)) * beta ** ((1 - alpha) / 2)
    # Legendre (translated)
    elif measure == 'legt':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    # LMU: equivalent to LegT up to normalization
    elif measure == 'lmu':
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1)[:, None]  # / theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
        B = (-1.) ** Q[:, None] * R
    # Legendre (scaled)
    elif measure == 'legs':
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]

    return A, B


class Freq_enhanced_layer(Layer):
    def __init__(self, modes, compression=0):
        super(Freq_enhanced_layer, self).__init__()
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.compression = compression

    def build(self, input_shape):
        B, H, N, T, C = input_shape
        self.modes = min(self.modes, C // 2 + 1)
        self.scale = 1/(T**2)

        if self.compression == 0:
            self.W = self.add_weight("W",
                                     shape=(T, T, self.modes),
                                     initializer=GlorotNormal(),
                                     trainable=True)*self.scale
        elif self.compression > 0:  ## Low-rank approximation
            self.w0 = self.add_weight("w0",
                                      shape=(T, self.compression),
                                      initializer=GlorotNormal(),
                                      trainable=True)*self.scale
            self.w1 = self.add_weight("w1",
                                      shape=(self.compression, self.compression, self.modes),
                                      initializer=GlorotNormal(),
                                      trainable=True)*self.scale
            self.w2 = self.add_weight("w2",
                                      shape=(self.compression, T),
                                      initializer=GlorotNormal(),
                                      trainable=True)*self.scale

    def call(self, tensor):
        B, H, N, T, C = tensor.shape
        # Compute Fourier coefficients up to factor of e^(- something constant)
        tensor_ft = tf.signal.rfft(tensor)
        # Multiply relevant Fourier modes
        padding_size = C // 2 + 1 - self.modes
        if padding_size > 0:
            padding_tensor = tf.tile(tf.cast(tensor[:, :, :, :, 0:1] * 0, dtype=tf.complex64),
                                     (1, 1, 1, 1, padding_size))
        if self.compression == 0:
            a = tensor_ft[:, :, :, :, :self.modes]
            out_ft = tf_complex_einsum('bnjix,iox->bnjox', a, self.W)
            if padding_size > 0:
                out_ft = Concatenate(-1)([out_ft, padding_tensor])
        else:
            a = tensor_ft[:, :, :, :, :self.modes]
            a = tf_complex_einsum('bnjix,ih->bnjhx', a, self.w0)
            a = tf_complex_einsum('bnjhx,hkx->bnjkx', a, self.w1)
            out_ft = tf_complex_einsum('bnjkx,ko->bnjox', a, self.w2)
            if padding_size > 0:
                out_ft = Concatenate(-1)([out_ft, padding_tensor])

        # Return to physical space
        x = tf.signal.irfft(out_ft)
        return x


class LPU(Layer):
    def __init__(self, N=256, dt=1.0, discretization='bilinear'):
        # N: the order of the Legendre projection
        # dt: step size - can be roughly inverse to the length of t
        super(LPU, self).__init__()
        self.N = N
        A, B = transition('lmu', N)  ### LMU projection matrix
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)
        B = tf.squeeze(B, axis=-1)

        vals = np.arange(0.0, 1.0, dt)
        self.eval_matrix = tf.constant(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T, dtype=tf.float32)
        self.A = tf.cast(A, tf.float32)
        self.B = tf.cast(B, tf.float32)

    def call(self, inputs):
        # inputs: (length, ...)
        # output: (length, ..., N) where N is the order of the Lege
        cs = []
        c = tf.tile(inputs[:,:,:,0:1]*0, (1,1,1,self.N))
        for i in range(inputs.shape[-1]):
            f = inputs[:, :, :, i:i+1]
            new = tf.matmul(f, tf.expand_dims(self.B, axis=0))
            c = tf.matmul(c, self.A) + new
            cs.append(c)
        return tf.stack(cs, axis=0)


class FiLM(Layer):
    def __init__(self, pred_len, seq_len, modes, compression, multiscale, **kwargs):
        super(FiLM, self).__init__(**kwargs)
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.modes = modes
        self.compression = compression
        self.window_sizes = [i*pred_len for i in multiscale]

    def build(self, input_shape):
        self.legts = [LPU(N=self.seq_len, dt=1 / ws) for ws in self.window_sizes]
        self.FELs = [Freq_enhanced_layer(self.modes, self.compression) for ws in self.window_sizes]

    def call(self, tensor):
        out_list = []
        for i, ws in enumerate(self.window_sizes):
            tmp_tensor = tensor[:, :, -ws:, :]
            legt = self.legts[i]
            tmp_tensor = tf.transpose(
                legt(tf.transpose(tmp_tensor, (0, 1, 3, 2))), (1, 2, 3, 4, 0)
            )
            out = self.FELs[i](tmp_tensor)
            out = tf.transpose(out, (0, 1, 2, 4, 3))[:, :, :, -1, :]
            out = tf.matmul(out, tf.transpose(legt.eval_matrix))
            out_list.append(out)
        output = Concatenate()(out_list)
        return tf.transpose(output, (0, 1, 3, 2))