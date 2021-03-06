import os
import numpy as np
import sys
import getopt
sys.path.append("..")

from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Conv2D, Conv2DTranspose, Input, Add, LeakyReLU, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal

from utils.data_utils import load_real_images, load_fake_images, update_image_pool
from models import CycleGAN


def _build_res_block(x, num_filters=256):
    conv01 = Conv2D(filters=num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1), activation="relu")(x)
    conv01 = InstanceNormalization(axis=-1)(conv01)
    conv02 = Conv2D(filters=num_filters, kernel_size=(3, 3), padding="same", strides=(1, 1))(conv01)
    conv02 = InstanceNormalization(axis=-1)(conv02)

    return Add()([conv02, x])


def _build_generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    input_generator = Input(image_shape)
    x = Conv2D(32, kernel_size=(7, 7), padding="same", strides=(1, 1), activation="relu", kernel_initializer=init)(input_generator)
    x = InstanceNormalization(axis=-1)(x)
    x = Conv2D(64, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)

    for _ in range(9):
        x = _build_res_block(x, num_filters=128)

    x = Conv2DTranspose(64, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = Conv2DTranspose(32, kernel_size=(3, 3), padding="same", strides=(2, 2), activation="relu", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)

    x = Conv2D(3, kernel_size=(7, 7), padding="same", strides=(1, 1), activation="tanh")(x)

    return Model(input_generator, x)


def _build_discriminator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)

    input_discriminator = Input(image_shape)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(input_discriminator)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding="same", kernel_initializer=init)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    #x = Conv2D(512, kernel_size=(4, 4), padding="same", activation="relu")(x)
    #x = InstanceNormalization()(x)

    patch_out = Conv2D(1, kernel_size=(4, 4), padding="same", kernel_initializer=init)(x)

    model = Model(input_discriminator, patch_out)
    model.compile(loss="mse", optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

    return model


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
    combined_model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=Adam(lr=0.0002, beta_1=0.5))

    return combined_model


def train(generator_AtoB, generator_BtoA, discriminator_A, discriminator_B,
          combined_model_AtoB, combined_model_BtoA, trainA, trainB,
          n_epochs=100, batch_size=1, save_path=""):

    n_patch = discriminator_A.output_shape[1]
    poolA, poolB = list(), list()
    batches_per_epoch = int(len(trainA) / batch_size)
    n_steps = batches_per_epoch * n_epochs

    for i in range(n_steps):
        X_realA, y_realA = load_real_images(trainA, batch_size, n_patch)
        X_realB, y_realB = load_real_images(trainB, batch_size, n_patch)

        X_fakeA, y_fakeA = load_fake_images(generator_BtoA, X_realB, batch_size, n_patch)
        X_fakeB, y_fakeB = load_fake_images(generator_AtoB, X_realA, batch_size, n_patch)

        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)

        g_loss2, _, _, _, _ = combined_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1 = discriminator_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = discriminator_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = combined_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1 = discriminator_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = discriminator_B.train_on_batch(X_fakeB, y_fakeB)

        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (
        i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

        if i%5000 == 0:
            generator_AtoB.save(os.path.join(save_path, "generator_AtoB-epoch_" + str(i+1) + ".h5"))
            generator_BtoA.save(os.path.join(save_path, "generator_BtoA-epoch_" + str(i+1) + ".h5"))

            discriminator_B.save(os.path.join(save_path, "discriminatorB-epoch_" + str(i+1) + ".h5"))
            discriminator_A.save(os.path.join(save_path, "discriminatorA-epoch_" + str(i+1) + ".h5"))


if __name__ == "__main__":
    cycleGAN = CycleGAN(image_shape=(256, 256, 3))

    args = sys.argv[1:]

    data_path = ''

    try:
        opts, args = getopt.getopt(args, "p:", ["data_path="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p", "--data_path"):
            data_path = arg

    image_shape = (256, 256, 3)

    #generator_AtoB = cycleGAN.G_A2B
    generator_AtoB = _build_generator(image_shape=image_shape)

    #plot_model(generator_AtoB, to_file="generator_own.png")
    #generator_BtoA = cycleGAN.G_B2A
    generator_BtoA = _build_generator(image_shape=image_shape)
    generator_AtoB.summary()

    #discriminator_A = cycleGAN.D_A
    discriminator_A = _build_discriminator(image_shape=image_shape)

    #plot_model(discriminator_A, to_file="discriminator_own.png")
    #discriminator_B = cycleGAN.D_B
    discriminator_B = _build_discriminator(image_shape=image_shape)
    discriminator_A.summary()

    # A -> B -> A [real/fake]
    combined_model_AtoB = _build_combined_model(image_shape, generator_AtoB, generator_BtoA, discriminator_B)

    # B -> A -> B [real/fake]
    combined_model_BtoA = _build_combined_model(image_shape, generator_BtoA, generator_AtoB, discriminator_A)

    trainA = np.load(os.path.join(data_path, "trainA" + ".npy"))
    trainB = np.load(os.path.join(data_path, "trainB" + ".npy"))

    train(generator_AtoB, generator_BtoA, discriminator_A, discriminator_B,
          combined_model_AtoB, combined_model_BtoA, trainA, trainB, save_path=data_path)
