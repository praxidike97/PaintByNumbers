import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from utils.data_utils import load_real_images

data_path = "/media/fabian/Data/ML_Data/CycleGAN/ukiyoe2photo/trainB.npy"

generator_AtoB = load_model("/home/fabian/GitHubProjects/PaintByNumbers/data/generator_BtoA-epoch_40001.h5", custom_objects={'InstanceNormalization': InstanceNormalization})
trainA = np.load(data_path)

print(trainA.shape)
raw_predictions = generator_AtoB.predict(trainA[:20])
predictions = ((raw_predictions+1.)*127.5).astype(int)

for i in range(20):
    plt.imshow(trainA[i])
    plt.show()
    plt.imshow(predictions[i])
    plt.show()
