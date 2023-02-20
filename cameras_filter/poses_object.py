from pathlib import Path
from typing import Union, Tuple
import re

from utils.quaternion_transform import world_coordinates

import numpy as np


PathLikeObject = Union[str, Path]


class Poses:
    """Parent class for both COLMAP and Augmented City API poses.

    Initialize an object which consists poses directly from a
    file ('images.bin' or 'description.json').
    """

    pattern = re.compile(
        "[_]?([0-9]+).jpg", re.IGNORECASE
    )  # Extract id of every image from its name

    def __init__(self, file_with_poses: PathLikeObject, **kwargs) -> dict:
        """Construct poses object.

        Parameters
        --------------
            file_with_poses : PathLike object
                Reconstruction 'images.bin' file or 'description.json' file.

        Extra Parameters
        --------------
            selected_passage : int
            select_in_process : bool
        See 'passage_poses.Passage.extract_poses' docstring.
        """

        file_with_poses = Path(file_with_poses)

        print(f"Extracting poses of cameras from {str(file_with_poses)} file...")
        self.camera_poses = self.extract_poses(file_with_poses, **kwargs)

        self.images = tuple(self.camera_poses)
        self.num_of_objects = len(self.camera_poses)

        print(
            f"""Camera poses object was created. \nNumber of images: {self.num_of_objects}.\n"""
        )

    @property
    def camera_poses(self):
        return self._camera_poses

    @camera_poses.setter
    def camera_poses(self, value):
        self._camera_poses = value
        self.images = tuple(self.camera_poses)
        self.object_num = len(self.images)

    def extract_poses(self, file: PathLikeObject) -> dict:
        """Extract poses from file."""
        # Python virtual method
        raise NotImplementedError()

    def find_nearest(self, image: Tuple) -> Tuple[str, str]:
        """Auxiliary function for neighbour searching.

        Find two geometrically closest images for current image.
        """

        id, point = image
        passage = list(self.camera_poses.items())

        # Initial values
        first = second = None
        max_1 = max_2 = np.inf

        for item in passage:
            if item[0] == id:
                continue
            distance = Poses.distance(point, item[1])
            if distance <= max_1:
                max_2 = max_1
                max_1 = distance
                second = first
                first = item[0]
            elif distance <= max_2:
                max_2 = distance
                second = item[0]

        return (first, second)

    def find_neighbours(self, manual: bool = False):
        """Find two closest poses for every image.

        For every image in reconstruction/passage information
        algorithm searches two images, which are geometrically
        closest to current image. If the passage doesn't
        contain any geometric information, two closest
        images in image list will be neighbours.

        Parameters
        --------------
            manual : bool = False
                The variable was added specially for passages,
                which doesn't consist any geometric information
                (*_MANUAL).
        """

        self.neighbours = {}

        if manual:
            # The two closest to the first image are the second and the third
            self.neighbours[self.images[0]] = (self.images[1], self.images[2])
            # The two closest to the last image are the two previous images
            image_num = self.num_of_objects
            self.neighbours[self.images[image_num - 1]] = (
                self.images[image_num - 2],
                self.images[image_num - 3],
            )

            for ind, image in enumerate(self.images[1 : image_num - 1]):
                self.neighbours[image] = (self.images[ind - 1], self.images[ind + 1])
        else:
            for image in self.camera_poses.items():
                self.neighbours[image[0]] = tuple(self.find_nearest(image))

    @staticmethod
    def distance(point1: dict, point2: dict):
        point1 = np.array(list(point1.values()))
        point2 = np.array(list(point2.values()))
        return np.sqrt(sum((point2 - point1) ** 2))
