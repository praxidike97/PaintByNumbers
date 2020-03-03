import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from tqdm import tqdm
import pprint
import scipy.ndimage as ndimage

pp = pprint.PrettyPrinter(indent=4)

# Threshold for flood fill
threshold = 15.

# Merge areas with a size of 3 pixels or less with their surrounding
threshold_cell_size = 3

# The patch size where to look for the most frequent color
image_patch_size = 3


def generate_neighbours(image):
    """
    Generate all neighbours of all pixels in the image
    :param image:
    :return:
    """
    neighbour_dict = dict()
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            neighbour_dict[str((x,y))] = list(filter(lambda pos: 0 <= pos[0] < image.shape[0] and 0 <= pos[1] < image.shape[1],
                   [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]))

    return neighbour_dict


def get_most_frequent_color(image_patch):
    """
    Find the most frequent color in image_patch
    :param image_patch:
    :return:
    """
    color_dict = dict()
    for x in range(image_patch.shape[0]):
        for y in range(image_patch.shape[1]):
            if str(image_patch[x, y]) in color_dict.keys():
                count = color_dict[str(image_patch[x, y])][0] + 1
                color_dict[str(image_patch[x, y])] = (count, image_patch[x, y])
            else:
                color_dict[str(image_patch[x, y])] = (0, image_patch[x, y])

    return color_dict[max(color_dict.keys(), key=lambda k: color_dict[k][0])][1]


def flood_fill(image):
    """
    Flood fill: Merge neighbouring pixels if their color difference is below a
    certain threshold
    :param image:
    :return: flood filled image
    """
    all_pixels = set()

    # Set of all pixels
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            all_pixels.add((x, y))

    is_filled = set()
    index = 0
    cells = dict()
    pixel_to_cell = dict()

    neighbour_dict = generate_neighbours(image)

    # Exucute the algorithm until all the pixels in the image
    # have been considered
    while len(all_pixels) != 0:
        first_pixel = all_pixels.pop()
        points_to_fill = set([first_pixel])
        color_reference = image[first_pixel[0], first_pixel[1]]

        # One cell contains an area of the same color
        cells[index] = list()

        # Start with one point and fill its neighbours, its neighbour's neighbours etc
        # if their color is very similar to the first starting pixel
        while len(points_to_fill) != 0:

            for point in points_to_fill:
                image[point[0], point[1]] = color_reference
                is_filled.add(point)

                cells[index].append(point)
                pixel_to_cell[point] = index

            new_points_to_fill = set()
            for point in points_to_fill:
                new_points_to_fill.update(neighbour_dict[str((point[0], point[1]))])

            if len(new_points_to_fill) == 0:
                break

            # Check if the difference in color between all the neighbours and the starting
            # point is below a certain threshold --> these neighbours will also be filled
            new_points_to_fill = filter(
                lambda point: np.linalg.norm(image[point[0], point[1]] - color_reference) < threshold,
                new_points_to_fill)

            # Just take these points that we didn't look at before
            new_points_to_fill = set(new_points_to_fill).difference(is_filled)
            points_to_fill = new_points_to_fill

        index += 1

        all_pixels = all_pixels.difference(is_filled)
        print(len(all_pixels))

    return image, cells


def delete_small_cells(cells, image_patch_size, threshold_cell_size, image):
    small_cells = 0
    for key in tqdm(cells):
        if len(cells[key]) <= threshold_cell_size:
            small_cells += 1

            min_x = max(cells[key][0][0] - image_patch_size, 0)
            max_x = min(cells[key][0][0] + image_patch_size, image.shape[0] - 1)
            min_y = max(cells[key][0][1] - image_patch_size, 0)
            max_y = min(cells[key][0][1] + image_patch_size, image.shape[1] - 1)

            most_frequent_color_in_cell = get_most_frequent_color(image[min_x:max_x, min_y:max_y])

            for pixel in cells[key]:
                image[pixel[0], pixel[
                    1]] = most_frequent_color_in_cell  # color.rgb2lab([[most_frequent_color_in_cell[1]]])

    print("Number small cells: " + str(small_cells))
    return image


def flood_fill_image(image):
    """
    We do the following:
    1. Flood fill the image
    2. Merge small areas (below 3 pixels or so) with their neighbours
    3. Do a Gaussian blur
    4. Flood fill the image again

    :param image:
    :return:
    """
    image, cells = flood_fill(image)
    plt.imshow(color.lab2rgb(image))
    plt.show()

    # Delete small cells
    print("Number cells: " + str(len(cells)))
    image = delete_small_cells(cells, image_patch_size, threshold_cell_size, image)

    plt.imshow(color.lab2rgb(image))
    plt.show()

    # Gaussian blur
    rgb_image = color.lab2rgb(image)
    blurred_image = ndimage.gaussian_filter(rgb_image, sigma=(3, 3, 0), order=0)
    plt.imshow(blurred_image, interpolation='nearest')
    plt.show()

    # Flood fill (again)
    image, cells = flood_fill(color.rgb2lab(blurred_image))
    plt.imshow(color.lab2rgb(image))
    plt.show()

    # Delete small cells (again)
    plt.axis('off')
    image = delete_small_cells(cells, 8, 30, image)
    plt.imshow(color.lab2rgb(image))
    plt.show()

    return image
