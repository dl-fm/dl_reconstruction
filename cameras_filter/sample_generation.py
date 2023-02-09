from typing import Union, Tuple
from pathlib import Path
import re
import os

from utils.read_write_model import (
    read_images_text,
    read_images_binary,
    write_images_text,
    write_images_binary,
)
from utils.read_write_model import Image
from utils.quaternion_transform import world_coordinates, colmap_coordinates, add_vector

import numpy as np


PathLikeObject = Union[str, Path]
pattern = re.compile("ios_([0-9]*).jpg")


def add_noise(
    path_to_images: PathLikeObject,
    path_to_output: PathLikeObject = "./images_sampled.bin",
    probability: float = 0.15,
    noise_scale: float = 1,
) -> Tuple[str]:
    """Add noise to each of the images
    from COLMAP reconstruction with a
    probability of parameter.

    Parameters
        --------------
        path_to_images : PathLikeObject
            Path to the images file from the COLMAP reconstruction.
            It's not recommended to set the original images file
            because of the risk of information loss. Make a copy
            of the sparse reconstruction directory.
        path_to_output : PathLikeObject = ./images_sampled.bin
            The path to the output images file with noise on
            some images. If the path_to_output is equal to the
            path_to_images, the source images file will be removed.
        probability : float = 0.15
            The approximate proportion of noised data.
        noise_scale: float = 1
            Standard deviation of a random noise variable.
    """
    path_to_images = Path(path_to_images)
    path_to_output = Path(path_to_output)

    read_method = (
        read_images_text if str(path_to_images).endswith(".txt") else read_images_binary
    )

    images = read_method(path_to_images)

    threshold = 1 - probability
    noised = []

    print("Noising image poses...")

    for key, image in images.items():
        if np.random.rand() > threshold:
            noise = np.random.standard_normal((3, 1)) * noise_scale
            images[key] = Image(
                id=image[0],
                qvec=image[1],
                tvec=add_vector(image[1], image[2], noise)[:, 0],
                camera_id=image[3],
                name=image[4],
                xys=image[5],
                point3D_ids=image[6],
            )
            noised.append(re.search(pattern, image[4])[1])

    if path_to_images == path_to_output:
        os.remove(path_to_images)
        path_to_output = Path(
            str(path_to_output)[:-4] + "_sampled" + str(path_to_output)[-4:]
        )
    write_method = (
        write_images_text
        if str(path_to_output).endswith(".txt")
        else write_images_binary
    )
    write_method(images, path_to_output)

    result = set(noised)

    print(
        f"{len(noised)} poses out of {len(images)} were noised ({round(len(noised)/len(images)*100, 1)}%):"
    )
    print(result)

    return result, path_to_output
