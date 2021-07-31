import os
from typing import Dict
import math
import pandas as pd
import numpy as np
import constants
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def combine_images(generated_images):
    """
    Combines the provided (multiple) images into a single image array.

    :param generated_images: numpy array containing multiple images

    :return: images: single numpy array containing the combined image.
    """
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    image_shape = generated_images.shape[1:4]
    image = np.zeros((height*image_shape[0], width*image_shape[1], image_shape[2]), dtype=generated_images.dtype)

    for idx, img in enumerate(generated_images):
        i = int(idx/width)
        j = idx%width

        image[i*image_shape[0]:(i+1)*image_shape[0], j*image_shape[1]:(j+1)*image_shape[1], :] = img[:, :, :]

    # handle black and white image.
    if image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    image *= 255
    return image.astype(np.uint8)


def plot_log(file_args: Dict, show=False):
    """
    Plots the log file using the provided file args.

    :param file_args: dictonary contating details for log file path (obatained from the experiment args)
    :param show: boolean on whether or not to show the plot after saving


    """
    filename = os.path.join(
        file_args[constants.SAVE_DIR],
        file_args[constants.LOG_FILENAME]
    )
    data = pd.read_csv(filename)
    fig = plt.figure(figsize=(12, 6), dpi=144)
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(1, 2, 1)
    plt.plot(data[constants.LOG_EPOCH], data[constants.LOG_TRAIN_LOSS],
             label=constants.LOG_TRAIN_LOSS)
    plt.plot(data[constants.LOG_EPOCH], data[constants.LOG_VAL_LOSS],
             label=constants.LOG_VAL_LOSS)

    plt.xlabel(f"# {constants.LOG_EPOCH}")
    plt.legend()
    plt.title('LOSS')

    fig.add_subplot(1, 2, 2)
    plt.plot(data[constants.LOG_EPOCH], data[constants.LOG_TRAIN_ACC],
             label=constants.LOG_TRAIN_ACC)
    plt.plot(data[constants.LOG_EPOCH], data[constants.LOG_VAL_ACC],
             label=constants.LOG_VAL_ACC)

    plt.xlabel(f"# {constants.LOG_EPOCH}")
    plt.legend()
    plt.title('ACCURACY')

    save_path = os.path.join(
        file_args[constants.SAVE_DIR],
        file_args[constants.LOG_PLOT_FILENAME]
    )
    fig.savefig(save_path)

    if show:
        plt.show()
