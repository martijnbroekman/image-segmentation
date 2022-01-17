import numpy as np
from PIL import Image
import os
import math


def get_image_array(image_name):
    image = Image.open(image_name)
    pix = image.load()
    width = image.size[0]
    height = image.size[1]

    return pix, width, height


def compare_image(original_image, compared_image):

    original_image_array, width, height = get_image_array(original_image)
    compared_image_array, compare_width, compare_height = get_image_array(compared_image)

    total_pixel = width * height
    pixel_difference = 0

    orginal_color = original_image_array[0, 0]
    compare_color = compared_image_array[0, 0]

    for x in range(width):
        for y in range(height):
            # andere manier gebruiken
            if (original_image_array[x, y] != orginal_color and compared_image_array[x, y] == compare_color) or (original_image_array[x, y] == orginal_color and compared_image_array[x, y] != compare_color):
                pixel_difference += 1

    return pixel_difference * 100 / total_pixel


def rgb2hsv(image_name):
    image = Image.open(image_name)
    # image = image.convert('HSV')
    pix = image.load()
    width = image.size[0]
    height = image.size[1]

    return pix, width, height


compare_colors = {
    'a1': {
        1024: {0: (254, 242, 190), 1: (207, 159, 61)},
        512: {0: (254, 242, 190), 1: (204, 158, 64)},
        256: {0: (254, 242, 190), 1: (185, 139, 53)},
        128: {0: (255, 243, 193), 1: (196, 37, 16)},
        64: {0: (254, 244, 193), 1: (135, 106, 28)},
    },
    'a10': {
        1024: {0: (254, 242, 190), 1: (207, 159, 61)},
        512: {0: (204, 158, 64), 1: (254, 242, 190)},
        256: {0: (185, 139, 53), 1: (254, 242, 190)},
        128: {0: (176, 135, 55), 1: (255, 243, 193)},
        64: {0: (177, 147, 75), 1: (254, 244, 193)},
    },
    'a100': {
        1024: {0: (254, 242, 190), 1: (207, 159, 61)},
        512: {0: (254, 242, 190), 1: (204, 158, 64)},
        256: {0: (185, 139, 53), 1: (254, 242, 190)},
        128: {0: (255, 243, 193), 1: (176, 135, 55)},
        64: {0: (254, 244, 193), 1: (135, 106, 28)},
    }
}

kmeans_colors = {
    1024: {0: (254, 242, 190), 1: (217, 177, 81)},
    512: {0: (254, 242, 190), 1: (213, 173, 78)},
    256: {0: (212, 174, 91), 1: (254, 242, 190)},
    128: {0: (255, 243, 193), 1: (176, 135, 55)},
    64: {0: (177, 147, 75), 1: (254, 244, 193)},
}

def calculate_distance(color1, color2):
    return math.sqrt((color2[0] - color1[0]) ** 2 + (color2[1] - color1[1]) ** 2 + (color2[2] - color1[2]) ** 2)

for key in compare_colors.keys():
    range_dict = compare_colors[key]
    for range in range_dict.keys():
        color1 = kmeans_colors[range][0]
        color2 = kmeans_colors[range][1]

        compare_color1 = range_dict[range][0]
        compare_color2 = range_dict[range][1]

        color_distance = calculate_distance(color1, compare_color1)
        color_distance2 = calculate_distance(color1, compare_color2)
        color_distance3 = calculate_distance(color2, compare_color1)
        color_distance4 = calculate_distance(color2, compare_color2)

        print(key, range)
        if color_distance <= color_distance2 and color_distance <= color_distance3 and color_distance <= color_distance4:
            print(color_distance)
            print(color_distance4)
        elif color_distance2 <= color_distance and color_distance2 <= color_distance3 and color_distance2 <= color_distance4:
            print(color_distance2)
            print(color_distance3)
        elif color_distance3 <= color_distance and color_distance3 <= color_distance2 and color_distance3 <= color_distance4:
            print(color_distance2)
            print(color_distance3)
        else:
            print(color_distance)
            print(color_distance4)

#
# directory = os.fsencode('C:\\Users\\20202958\\source\\repos\\kmeansplus\\City\\kmeans')
# original_folder = 'C:\\Users\\20202958\\source\\repos\\kmeansplus\\City\\kmeans'
# compare_root_folder = 'C:\\Users\\20202958\\source\\repos\\kmeansplus\\City\\coreset'
# maxhue = 0
# for subdir, dirs, files in os.walk(compare_root_folder):
#     for dir in dirs:
#         compare_folder = os.path.join(compare_root_folder, dir)
#         print(compare_folder)
#
#         for file in os.listdir(directory):
#             colors = {}
#             distances = {}
#             filename = os.fsdecode(file)
#             if filename.endswith(".jpg"):
#                 result, width, height = rgb2hsv(os.path.join(original_folder, filename))
#                 result_compare, width2, height2 = rgb2hsv(os.path.join(compare_folder, filename))
#                 # if result < 50 and not ('a10' in filename and 'a100' in filename):
#                 for x in range(width):
#                     for y in range(height):
#                         h0 = result[x, y]
#                         colors[h0] = 0
#                         # h1 = result_compare[x, y]
#                         # hueDistance = math.sqrt((h1[0] - h0[0])**2 + (h1[1] - h0[1])**2 + (h1[2] - h0[2])**2)
#                         # distances[hueDistance] = 0
#
#             print(compare_folder, filename, len(colors) ,colors)