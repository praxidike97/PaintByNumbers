import numpy as np
import matplotlib.pyplot as plt
import json

from keras.models import load_model, model_from_json
from keras.layers import Layer
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec

from utils.data_utils import load_real_images
import tensorflow as tf


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


data_path = "/media/fabian/Data/ML_Data/CycleGAN/ukiyoe2photo/trainB.npy"

generator_BtoA = load_model("/home/fabian/GitHubProjects/PaintByNumbers/data/generator_BtoA-epoch_45001.h5", custom_objects={'InstanceNormalization': InstanceNormalization, "ReflectionPadding2D": ReflectionPadding2D})

#a = json.loads("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_model_epoch_120.json")

#generator_BtoA = model_from_json(json.loads("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_model_epoch_120.json"), custom_objects={'InstanceNormalization': InstanceNormalization})
#generator_BtoA.load_weights("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_weights_epoch_120.hdf5")

trainB = np.load(data_path)

print(trainB.shape)
raw_predictions = generator_BtoA.predict(trainB[100:120])
predictions = ((raw_predictions+1.)*127.5).astype(int)

for i in range(20):
    plt.imshow(trainB[100+i])
    plt.show()
    plt.imshow(predictions[i])
    plt.show()
