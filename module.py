import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=2,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)


def ConvDiscriminator(input_shape=(256, 256, 3),
                      dim=64,
                      n_downsamplings=3,
                      norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 8)
        h = keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # 3
    h = keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return keras.Model(inputs=inputs, outputs=h)

""" import torch
import torch.nn as nn

class ConvDiscriminator(nn.Module):
    def __init__(self, input_shape=(256, 256, 3), dim=64, n_downsamplings=3, norm='instance_norm'):
        super(ConvDiscriminator, self).__init__()
        self.dim_ = dim
        self.norm = _get_norm_layer(norm)

        # 0
        self.h = nn.Identity()
        self.inputs = nn.Identity()

        # 1
        self.h = nn.Sequential(
            nn.Conv2d(input_shape[2], dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        for _ in range(n_downsamplings - 1):
            dim = min(dim * 2, self.dim_ * 8)
            self.h = nn.Sequential(
                nn.Conv2d(dim // 2, dim, kernel_size=4, stride=2, padding=1, bias=False),
                self.norm(dim),
                nn.LeakyReLU(0.2)
            )

        # 2
        dim = min(dim * 2, self.dim_ * 8)
        self.h = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=4, stride=1, padding=1, bias=False),
            self.norm(dim),
            nn.LeakyReLU(0.2)
        )

        # 3
        self.h = nn.Conv2d(dim, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.inputs(x)
        x = self.h(x)
        return x """

# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):

        self._steps = tf.cast(self._steps, tf.float32)
        step = tf.cast(step, tf.float32)
        self._step_decay = tf.cast(self._step_decay, tf.float32)

        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate


""" import torch.optim.lr_scheduler as lr_scheduler

class LinearDecay(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, step_decay, last_epoch=-1):
        self.initial_learning_rate = optimizer.defaults['lr']
        self.total_steps = total_steps
        self.step_decay = step_decay
        super(LinearDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.step_decay:
            decay_steps = self.total_steps - self.step_decay
            current_step = self.last_epoch - self.step_decay + 1
            decay_factor = (1 - current_step / decay_steps)
            return [base_lr * decay_factor for base_lr in self.base_lrs]
        else:
            return self.base_lrs """