from pathlib import Path
from typing import Union, Tuple
import re

import numpy as np

from poses_object import Poses
from utils.read_write_model import read_images_binary, read_images_text
from utils.quaternion_transform import world_coordinates


PathLikeObject = Union[str, Path]


class Reconstruction(Poses):
    """COLMAP reconstruction camera positions.

    Extract positions in world coordinates, find neighbours
    and calculate average distances.
    """

    def extract_poses(self, path_to_images: PathLikeObject) -> dict:
        """Extract poses from file.

        File must contain information about images
        from COLMAP reconstruction (e.g. 'images.bin').

        Parameters
        --------------
            path_to_images : PathLikeObject
                Path to the file with image information
                of the '.txt' or '.bin' extension.
        """

        print("Load COLMAP reconstruction...")
        path_to_images = Path(path_to_images)
        images = (
            read_images_binary(path_to_images)
            if str(path_to_images).endswith(".bin")
            else read_images_text(images)
        )

        result = {}
        Oxyz = ("x", "y", "z")

        for image in images.values():

            qvec = image[1]
            tvec = image[2]

            result[re.search(self.pattern, image[4])[1]] = {
                key: value
                for key, value in zip(Oxyz, world_coordinates(qvec, tvec)[:, 0])
            }

        return result

    def find_neighbours(self):
        # COLMAP neighbours search can only be called with manual = False.
        Poses.find_neighbours(self, manual=False)

    def calculate_distances_stats(self, distances: np.ndarray) -> Tuple[float]:
        """Calculate statistics across the distances array.

        Count average distance to the neighbours and
        the standard deviation of distances.
        """

        mean_distances = distances.mean(axis=1)

        self.mean = np.mean(mean_distances)
        self.std = np.std(mean_distances)

        print(
            f"Average COLMAP representation distance: {self.mean}.",
            f"Standard deviation: {self.std}.\n",
            sep="\n",
        )

    def calculate_distances(self) -> np.ndarray:
        """Calculate distances to two closest neighbours
        for every image in COLMAP reconstruction.
        """

        self.distances = {}
        distances = []

        print("Calculating distances...")

        for image in self.neighbours.items():
            point = self.camera_poses[image[0]]  # Current image
            point1 = self.camera_poses[image[1][0]]  # The first neighbour.
            point2 = self.camera_poses[image[1][1]]  # The second one.

            distance_1 = Poses.distance(point, point1)
            distance_2 = Poses.distance(point, point2)

            self.distances[image[0]] = {
                image[1][0]: distance_1,
                image[1][1]: distance_2,
            }

            distances.append([distance_1, distance_2])

        self.calculate_distances_stats(np.array(distances))

    def delete_unnecessary_images(self, passage: Poses):
        """Delete images that are not considered in a particular passage."""

        print("Deleting extra images from COLMAP reconstruction representation...")

        self.camera_poses = {
            key: value
            for key, value in self.camera_poses.items()
            if key in passage.images
        }
