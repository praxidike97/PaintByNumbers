import os
import numpy as np

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, Input, Add, Concatenate, BatchNormalization, LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from utils.data_utils import load_real_images, load_fake_images

data_path = "/media/fabian/Data/ML_Data/horse2zebra"


def _build_res_block(x, num_filters=256):
    conv01 = Conv2D(filters=num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")(x)
    conv02 = Conv2D(filters=num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")(conv01)

    return Add()([conv02, x])


def _build_generator(image_shape=(256, 256, 3)):
    input_generator = Input(image_shape)
    x = Conv2D(64, kernel_size=(7, 7), padding="same", strides=(1, 1), activation="relu")(input_generator)
    x = InstanceNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu")(x)
    x = InstanceNormalization()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu")(x)
    x = InstanceNormalization()(x)

    for _ in range(9):
        x = _build_res_block(x, num_filters=256)

    x = Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu")(x)
    x = InstanceNormalization()(x)
    x = Conv2DTranspose(64, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu")(x)
    x = InstanceNormalization()(x)

    x = Conv2D(3, kernel_size=(7, 7), padding="same", strides=(1, 1), activation="tanh")(x)

    return Model(input_generator, x)


def _build_discriminator(image_shape=(256, 256, 3)):
    input_discriminator = Input(image_shape)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same")(input_discriminator)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Conv2D(512, kernel_size=(4, 4), padding="same", activation="relu")(x)
    #x = InstanceNormalization()(x)

    patch_out = Conv2D(1, kernel_size=(4, 4), padding="same")(x)

    return Model(input_discriminator, patch_out)


def _build_combined_model(image_shape, generator01, generator02, discriminator):
    generator01.trainable = True
    generator02.trainable = False
    discriminator.trainable = False

    input_fake = Input(image_shape)
    input_id = Input(image_shape)

    # Discriminator error
    output_generator01 = generator01(input_fake)
    output_discriminator = discriminator(output_generator01)

    # Identity loss
    output_id_generator01 = generator01(input_id)

    # Forward loss
    output_forward_generator02 = generator02(output_generator01)

    # Backward loss
    output_backward_generator02 = generator02(input_id)
    output_backward_generator01 = generator01(output_backward_generator02)

    combined_model = Model([input_fake, input_id],
                           [output_discriminator, output_id_generator01, output_forward_generator02, output_backward_generator01])
    combined_model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(lr=0.0002))

    return combined_model


image_shape = (256, 256, 3)

generator_XtoY = _build_generator(image_shape=image_shape)
generator_YtoX = _build_generator(image_shape=image_shape)
generator_YtoX.summary()

discriminator_X = _build_discriminator(image_shape=image_shape)
discriminator_Y = _build_discriminator(image_shape=image_shape)
discriminator_Y.summary()

# X -> Y -> X [real/fake]
combined_model_XtoY = _build_combined_model(image_shape, generator_XtoY, generator_YtoX, discriminator_Y)

# Y -> X -> Y [real/fake]
combined_model_YtoX = _build_combined_model(image_shape, generator_YtoX, generator_XtoY, discriminator_X)


data = np.load(os.path.join(data_path, "trainX" + ".npy"))
X, y = load_fake_images(generator_XtoY, data, 128, 16)