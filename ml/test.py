import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from utils.data_utils import load_real_images

data_path = "/media/fabian/Data/ML_Data/horse2zebra/trainA.npy"

trainA = np.load(data_path)
generator_AtoB = load_model("../data/generator_AtoB-epoch_20001.h5", custom_objects={'InstanceNormalization': InstanceNormalization})

print(trainA.shape)
raw_predictions = generator_AtoB.predict(trainA[:10])
predictions = ((raw_predictions+1.)*127.).astype(int)

for i in range(10):
    plt.imshow(trainA[i])
    plt.show()
    plt.imshow(predictions[i])
    plt.show()
