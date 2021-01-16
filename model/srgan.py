import time

import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.keras.losses import MeanAbsoluteError

from model.common import normalize, denormalize, pixel_shuffle, resolve, psnr, evaluate


class SrGan(tf.keras.Model):
    def __init__(self, scale, checkpoint_dir, valid_ds, steps=100000, num_filters=32, num_res_blocks=8,
                 res_block_expansion=6, res_block_scaling=None):

        super(SrGan, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.valid_ds = valid_ds
        self.steps = steps
        self.evaluate_every = tf.Variable(1000)

        self.learning_rate = PiecewiseConstantDecay(boundaries=[200000], values=[1e-3, 5e-4])

        self.loss_mean = Mean()

        # the model
        self.model = self.create_model(num_filters, num_res_blocks, res_block_expansion, res_block_scaling, scale)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(self.learning_rate),
                                              model=self.model)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=3)

        self.restore()

    def create_model(self, num_filters, num_res_blocks, res_block_expansion, res_block_scaling, scale):
        x_in = Input(shape=(None, None, 3))
        x = Lambda(normalize)(x_in)

        # main branch
        m = Conv2D(num_filters, 3, padding='same')(x)

        for i in range(num_res_blocks):
            m = self.res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)

        m = Conv2D(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
        m = Lambda(pixel_shuffle(scale))(m)

        # skip branch
        s = Conv2D(3 * scale ** 2, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(x)
        s = Lambda(pixel_shuffle(scale))(s)
        x = Add()([m, s])
        x = Lambda(denormalize)(x)

        return Model(x_in, x, name="SrGan")

    @staticmethod
    def res_block(x_in, num_filters, expansion, kernel_size, scaling):
        linear = 0.8
        x = Conv2D(num_filters * expansion, 1, padding='same', activation='relu')(x_in)
        x = Conv2D(int(num_filters * linear), 1, padding='same')(x)
        x = Conv2D(num_filters, kernel_size, padding='same')(x)
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        x = Add()([x_in, x])
        return x

    def compile(self):
        super(SrGan, self).compile()

    def restore(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step}.')

    def evaluate(self, dataset):
        return evaluate(self.checkpoint.model, dataset)

    @tf.function
    def train_step(self, images):
        lr, hr = images

        self.checkpoint.step.assign_add(1)

        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = MeanAbsoluteError()(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        self.loss_mean(loss_value)

        return {"c_loss": loss_value}


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss_value = self.model.loss_mean.result()
        self.model.loss_mean.reset_states()

        psnr_val = evaluate(self.model.checkpoint.model, self.model.valid_ds)

        print(f", loss: {loss_value:.3f}, PSNR: {psnr_val.numpy():.3f}", end='')

        if psnr_val > self.model.checkpoint.psnr:
            self.model.checkpoint.psnr = psnr_val
            self.model.checkpoint_manager.save()
            print(', Saved!!!')

