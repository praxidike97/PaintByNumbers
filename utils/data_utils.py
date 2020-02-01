import glob
import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

data_path = "/media/fabian/Data/ML_Data/horse2zebra"


def images_to_npy():
    for dataset in ["trainX", "trainY", "testX", "testY"]:
        list_images = list()
        for image_path in glob.glob(os.path.join(data_path, dataset, "*")):
            im = Image.open(image_path)
            if np.array(im).shape == (256, 256, 3):
                list_images.append(np.array(im))

        array_images = np.array(list_images)
        np.save(os.path.join(data_path, dataset + ".npy"), array_images)


def load_real_images(dataset, number_images, n_patch):
    #data = np.load(os.path.join(data_path, dataset + ".npy"))
    indices = np.random.randint(low=0, high=len(dataset), size=number_images)
    X = dataset[indices]
    y = np.ones((number_images, n_patch, n_patch, 1))

    return X, y


def load_fake_images(generator, dataset, number_images, n_patch):
    #data = np.load(os.path.join(data_path, dataset + ".npy"))
    indices = np.random.randint(low=0, high=len(dataset), size=number_images)
    X = generator.predict(dataset[indices])
    y = np.zeros((number_images, n_patch, n_patch, 1))

    return X, y


if ["trainX.npy", "trainY.npy", "testX.npy", "testY.npy"] not in glob.glob(os.path.join(data_path)):
    images_to_npy()


if __name__ == "__main__":
    data = np.load(os.path.join(data_path, "trainX" + ".npy"))
    X, y = load_real_images(data, 128, 16)
    X, y = load_fake_images(data, 128, 16)
    print(X.shape)
    plt.imshow(X[100])
    plt.show()