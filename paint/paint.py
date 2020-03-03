import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color
from skimage.filters import sobel, roberts, scharr_h, farid_v
from skimage.data import astronaut, camera
from skimage import util

from paint_algorithms import flood_fill_image

plt.axis('off')

im = np.array(Image.open("../data/bear.jpg"))
rgb = io.imread("../data/mona.jpg")
lab = color.rgb2lab(rgb)

#image = camera()
#edge_roberts = roberts(image)
#edge_sobel = sobel(image)

edge_sobel = sobel(color.rgb2gray(rgb))
plt.imshow(edge_sobel, cmap="gray")
plt.show()

image = flood_fill_image(lab)

edge_sobel = sobel(color.rgb2gray(image))

edge_emphasizer = lambda x: x if x <= 0.2 else 16.
vfunc = np.vectorize(edge_emphasizer)
edge_sobel = vfunc(edge_sobel)
edge_sobel /= 512.
edge_sobel[0, 0] = 1.

plt.axis('off')
plt.imshow(util.invert(edge_sobel), cmap="gray")
plt.show()

plt.axis('off')
edge_roberts = roberts(color.rgb2gray(image))
plt.imshow(util.invert(edge_roberts), cmap="gray")
plt.show()