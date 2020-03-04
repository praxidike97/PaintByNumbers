import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from tqdm import tqdm

from keras.models import load_model, model_from_json
from keras.layers import Layer
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec

from PIL import Image
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

generator_BtoA = load_model("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_model_all_epoch_120.h5", custom_objects={'InstanceNormalization': InstanceNormalization, "ReflectionPadding2D": ReflectionPadding2D})

#a = json.loads("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_model_epoch_120.json")

#generator_BtoA = model_from_json(json.loads("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_model_epoch_120.json"), custom_objects={'InstanceNormalization': InstanceNormalization})
#generator_BtoA.load_weights("/home/fabian/GitHubProjects/PaintByNumbers/data/G_B2A_model_weights_epoch_120.hdf5")

#trainB = np.load(data_path)

filelist = sorted(glob.glob('/home/fabian/Tmp/cycleGAN/*.jpg'))
#landscape_images = np.array([np.array(Image.open(fname)) for fname in filelist])

images = list()
for fname in tqdm(filelist):
    im = Image.open(fname)
    im = im.resize((256, 256), Image.ANTIALIAS)
    images.append(np.array(im))

#print(landscape_images.shape)
images = np.asarray(images)
np.save("/home/fabian/Tmp/lanscape-images.npy", images)

raw_predictions = generator_BtoA.predict(images)
predictions = ((raw_predictions+1.)*127.5).astype(int)

for i, prediction in enumerate(predictions):
    image = Image.fromarray(prediction.astype('uint8'), 'RGB')
    image.save('/home/fabian/Tmp/cycleGAN-transformed/image-%s.jpg' % str(i).zfill(4))

for i in range(100):
    #plt.imshow(trainB[i])
    #plt.show()
    plt.imshow(predictions[i])
    plt.show()
