from pathlib import Path
from typing import Union, Tuple, Optional
import json
import copy

from reconstruction_poses import Reconstruction
from passage_poses import Passage

import numpy as np
from scipy.stats import norm


class CameraFilter:
    """Cameras filtering algorithm.

    Filter out incorrect camera positions from the
    COLMAP representation using a statistic sigma-rule
    with adjustable softness.

    Two configurations are available:
    1. Filter with both the COLMAP representation
    and the Augmented City API description file
    2. Filter with the COLMAP representation only.
    """

    def __init__(
        self, reconstruction: Reconstruction, passage: Optional[Passage] = None
    ):
        """Filter out wrong camera positions using the API
        information or the COLMAP representation only.
        """

        if not passage is None:
            self.passage = passage
            self.with_passage = True
        else:
            self.with_passage = False
        self.reconst = reconstruction

    def calculate_true_distances(self):
        """Get the distances to the true
        neighbours from the passage information
        from the Augmented City API.
        """

        if not self.with_passage:
            # Use the calculated COLMAP distances as the accurate distances.
            self.true_distances = copy.deepcopy(self.reconst.distances)
            return

        self.true_distances = {}

        for image in self.reconst.camera_poses.keys():

            true_neighbour_1, true_neighbour_2 = self.passage.neighbours[image]

            point = self.reconst.camera_poses[image]

            reconst_neighbour_1 = self.reconst.neighbours[image][0]
            reconst_neighbour_2 = self.reconst.neighbours[image][1]

            # It is possible that there won't be such images in the COLMAP representation
            # So extract the first two unique filenames, which exist in the COLMAP representation
            used = []
            ids = [
                true_neighbour_1,
                true_neighbour_2,
                reconst_neighbour_1,
                reconst_neighbour_2,
            ]
            neighbour_1, neighbour_2 = [
                x
                for x in ids
                if (x not in used)
                and (x in self.reconst.images)
                and (used.append(x) or True)
            ][:2]

            point_1 = self.reconst.camera_poses[neighbour_1]
            point_2 = self.reconst.camera_poses[neighbour_2]

            distance_1 = Reconstruction.distance(point, point_1)
            distance_2 = Reconstruction.distance(point, point_2)

            self.true_distances[image] = {
                neighbour_1: distance_1,
                neighbour_2: distance_2,
            }

    def is_anomaly(self, distances: dict, interval: Tuple[float, float] = 0.95):
        dist = np.mean(np.array(list(distances.values())))
        return not interval[0] < dist < interval[1]

    def filter(self, softness: float = 0.95) -> set:
        """Cameras filtering algorithm.

        Preprocess the camera positions information.
        Find neighbours, calculate distances
        and run the algorithm.

        Parameters
        --------------
            softness : float = 0.95
                A real parameter with a value between 0 and 1,
                which determines approximately how many images
                won't be deleted by the algorithm.
        """

        self.cameras_filter = {}

        if self.with_passage:
            # Extract only the necessary images.
            self.reconst.delete_unnecessary_images(self.passage)

        # The data preprocession
        self.reconst.find_neighbours()
        self.reconst.calculate_distances()
        if self.with_passage:
            self.passage.find_neighbours()

        self.calculate_true_distances()

        # Confidence interval
        interval = norm.interval(
            softness, loc=self.reconst.mean, scale=self.reconst.std
        )
        print(f"The confidence interval: {interval}")

        for image in self.reconst.camera_poses.keys():
            flag = 1 if self.is_anomaly(self.true_distances[image], interval) else 0
            result_distance = np.mean(
                np.array(list(self.reconst.distances[image].values()))
            )
            if flag:
                print(
                    f"Image {image}. Average distance to COLMAP neighbours: {result_distance}. \n",
                    end="",
                )
            self.cameras_filter[image] = flag

        filtered = sum(self.cameras_filter.values())

        print(
            f"{ filtered } cameras out of {self.reconst.object_num} were filtered (",
            f"{ round(filtered * 100 / self.reconst.object_num, 2) } %).",
        )

        result = set(key for key, value in self.cameras_filter.items() if value == 1)

        return set(key for key, value in self.cameras_filter.items() if value == 1)
