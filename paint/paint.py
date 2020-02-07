import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, color

from paint_algorithms import get_neighbours, flood_fill_image

im = np.array(Image.open("../data/bear.jpg"))
rgb = io.imread("../data/monkey.png")
lab = color.rgb2lab(rgb)

# Fill a square near the middle with value 127, starting at index (76, 76)
print(rgb.shape)
#get_neighbours(image=rgb, position=(0, 0))
flood_fill_image(lab, (0, 0))