from pathlib import Path
from typing import Union
import os
import argparse

from utils.read_write_model import (
    read_images_binary,
    read_points3D_binary,
    write_points3D_binary,
)


PathLikeObject = Union[str, Path]


def delete_points(path_to_images: PathLikeObject, path_to_points: PathLikeObject):
    """Delete the dots that are associated with the deleted image.

    TODO: read/write points3D.txt and images.txt files.

    Parameters
        --------------
        path_to_images : PathLikeObject
            Path to the filtered file with image
            information of '.bin' extension.
        path_to_points: Optional[PathLikeObject]
            The regular points3D.bin file from
            COLMAP sparse reconstruction.
    """
    points = read_points3D_binary(path_to_points)
    images = read_images_binary(path_to_images)
    new_points = {}

    image_ids = images.keys()
    count = 0

    for key, value in points.items():
        if any([id not in image_ids for id in value[4]]):
            count += 1
            continue
        new_points[key] = value

    print(f"{round(count/len(points)*100, 2)}% points were removed.")

    os.remove(path_to_points)

    write_points3D_binary(new_points, path_to_points)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract passage")

    parser.add_argument("path_to_images", type=str, default="./")
    parser.add_argument("path_to_points", type=str, default="./")

    args = parser.parse_args()
    path_to_images, path_to_points = args.path_to_images, args.path_to_points

    delete_points(path_to_images=path_to_images, path_to_points=path_to_points)
