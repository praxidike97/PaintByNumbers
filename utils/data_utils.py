import glob
import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

data_path = "/media/fabian/Data/ML_Data/horse2zebra"


def images_to_npy():
    for dataset in ["trainA", "trainB", "testA", "testB"]:
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


def update_image_pool(image_pool, images, size=50):
    selected_images = list()
    for image in images:
        if len(image_pool) < size:
            image_pool.append(image)
            selected_images.append(image)
        elif np.random.random() < 0.5:
            selected_images.append(image)
        else:
            indices = np.random.randint(0, len(image_pool))
            selected_images.append(image_pool[indices])
            image_pool[indices] = image
    return np.asarray(selected_images)


if ["trainA.npy", "trainB.npy", "testA.npy", "testB.npy"] not in glob.glob(os.path.join(data_path)):
    images_to_npy()


if __name__ == "__main__":
    data = np.load(os.path.join(data_path, "trainA" + ".npy"))
    X, y = load_real_images(data, 128, 16)
    X, y = load_fake_images(data, 128, 16)
    print(X.shape)
    plt.imshow(X[100])
    plt.show()