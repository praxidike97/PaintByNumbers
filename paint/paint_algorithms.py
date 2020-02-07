from PIL import Image
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from tqdm import tqdm
import pprint
import scipy.ndimage as ndimage

pp = pprint.PrettyPrinter(indent=4)

threshold = 15.
threshold_cell_size = 3
image_patch_size = 3


def get_neighbours(position):
    x, y = position
    inner = list(filter(lambda pos: 0 <= pos[0] < image_x and 0 <= pos[1] < image_y,
                   [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]))
    return inner


def generate_neighbours(image):
    neighbour_dict = dict()
    for x in tqdm(range(image.shape[0])):
        for y in range(image.shape[1]):
            neighbour_dict[str((x,y))] = list(filter(lambda pos: 0 <= pos[0] < image.shape[0] and 0 <= pos[1] < image.shape[1],
                   [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]))

    return neighbour_dict


def get_most_frequent_color(image_patch):
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
    all_pixels = set()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            all_pixels.add((x, y))

    is_filled = set()
    index = 0
    cells = dict()
    pixel_to_cell = dict()

    # print(get_most_frequent_color(image[380:400, 700:720]))

    neighbour_dict = generate_neighbours(image)

    while len(all_pixels) != 0:
        first_pixel = all_pixels.pop()
        points_to_fill = set([first_pixel])
        color_reference = image[first_pixel[0], first_pixel[1]]

        cells[index] = list()
        # print("Vor points to fill...")
        while len(points_to_fill) != 0:
            # print("Vor for")
            for point in points_to_fill:
                image[point[0], point[1]] = color_reference
                is_filled.add(point)

                cells[index].append(point)
                pixel_to_cell[point] = index
            # is_filled.update(points_to_fill)

            # new_points_to_fill = [get_neighbours(point) for point in points_to_fill]
            # new_points_to_fill = [neighbour_dict[str((point[0],point[1]))] for point in points_to_fill]
            new_points_to_fill = set()
            for point in points_to_fill:
                new_points_to_fill.update(neighbour_dict[str((point[0], point[1]))])

            if len(new_points_to_fill) == 0:
                break

            # new_points_to_fill = reduce(lambda x, y: x + y, new_points_to_fill)
            new_points_to_fill = filter(
                lambda point: np.linalg.norm(image[point[0], point[1]] - color_reference) < threshold,
                new_points_to_fill)

            new_points_to_fill = set(new_points_to_fill).difference(is_filled)
            points_to_fill = new_points_to_fill  # set(points_to_fill).union(set(new_points_to_fill))

            # print(index)
        index += 1

        all_pixels = all_pixels.difference(is_filled)
        print(len(all_pixels))

    return image, cells


def flood_fill_image(image, position):
    x_ref, y_ref = position

    """
    color_reference = image[x_ref, y_ref]
    threshold = 15.
    threshold_cell_size = 3
    image_patch_size = 3

    all_pixels = set()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            all_pixels.add((x, y))

    is_filled = set()
    index = 0
    cells = dict()
    pixel_to_cell = dict()

    #print(get_most_frequent_color(image[380:400, 700:720]))

    neighbour_dict = generate_neighbours(image)

    while len(all_pixels) != 0:
        first_pixel = all_pixels.pop()
        points_to_fill = set([first_pixel])
        color_reference = image[first_pixel[0], first_pixel[1]]

        cells[index] = list()
        #print("Vor points to fill...")
        while len(points_to_fill) != 0:
            #print("Vor for")
            for point in points_to_fill:
                image[point[0], point[1]] = color_reference
                is_filled.add(point)

                cells[index].append(point)
                pixel_to_cell[point] = index
            #is_filled.update(points_to_fill)

            #new_points_to_fill = [get_neighbours(point) for point in points_to_fill]
            #new_points_to_fill = [neighbour_dict[str((point[0],point[1]))] for point in points_to_fill]
            new_points_to_fill = set()
            for point in points_to_fill:
                new_points_to_fill.update(neighbour_dict[str((point[0],point[1]))])

            if len(new_points_to_fill) == 0:
                break

            #new_points_to_fill = reduce(lambda x, y: x + y, new_points_to_fill)
            new_points_to_fill = filter(lambda point: np.linalg.norm(image[point[0], point[1]] - color_reference) < threshold,
                                        new_points_to_fill)

            new_points_to_fill = set(new_points_to_fill).difference(is_filled)
            points_to_fill = new_points_to_fill#set(points_to_fill).union(set(new_points_to_fill))

            #print(index)
        index += 1

        all_pixels = all_pixels.difference(is_filled)
        print(len(all_pixels))
    """

    image, cells = flood_fill(image)
    plt.imshow(color.lab2rgb(image))
    plt.show()

    # Delete small cells
    print("Number cells: " + str(len(cells)))
    small_cells = 0
    for key in tqdm(cells):
        if len(cells[key]) <= threshold_cell_size:
            small_cells +=1

            min_x = max(cells[key][0][0]-image_patch_size, 0)
            max_x = min(cells[key][0][0]+image_patch_size, image.shape[0]-1)
            min_y = max(cells[key][0][1]-image_patch_size, 0)
            max_y = min(cells[key][0][1]+image_patch_size, image.shape[1]-1)

            #most_frequent_color_in_cell = max(Image.fromarray(np.uint8(image[min_x:max_x, min_y:max_y])).getcolors(),
            #                                  key=lambda t: t[0])

            most_frequent_color_in_cell = get_most_frequent_color(image[min_x:max_x, min_y:max_y])

            for pixel in cells[key]:
                image[pixel[0], pixel[1]] = most_frequent_color_in_cell#color.rgb2lab([[most_frequent_color_in_cell[1]]])

    print("Number small cells: " + str(small_cells))

    #print(color.lab2rgb(get_most_frequent_color(image[380:400, 700:720])))
    #print(Image.fromarray(np.uint8(image[380:400, 700:720])).getcolors())
    #print(max(Image.fromarray(np.uint8(image[380:400, 700:720])).getcolors(), key=lambda t: t[0]))

    plt.imshow(color.lab2rgb(image))
    plt.show()

    rgb_image = color.lab2rgb(image)
    blurred_image = ndimage.gaussian_filter(rgb_image, sigma=(2, 2, 0), order=0)
    plt.imshow(blurred_image, interpolation='nearest')
    plt.show()

    image, cells = flood_fill(color.rgb2lab(blurred_image))
    plt.imshow(color.lab2rgb(image))
    plt.show()