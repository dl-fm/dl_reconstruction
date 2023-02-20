from pathlib import Path
from typing import Union
import json
import re

from poses_object import Poses

PathLikeObject = Union[str, Path]


class Passage(Poses):
    """A passage object, which is used to select a subset
    of the reconstruction images. It is also believed
    that we could achieve better results from filtering
    algorithm using passage information.
    """

    # Passage styles
    linear_or_circular = {"circular": 0, "linear": 1, "calibration": 2}
    manual_or_auto = {"auto": 0, "manual": 1}

    def extract_poses(
        self,
        path_to_description: PathLikeObject,
        selected_passage: int = 0,
        select_in_process: bool = False,
    ) -> dict:
        """Extract true camera poses from the output
        description file from Augmented City API.

        Parameters
        --------------
            path_to_description : PathLikeObject
                Path to the description file of
                '.json' extension.

            selected_passage : int = 0
                The d of the selected passage from
                the list of passages.

            select_in_process : bool = False
                If True, function outputs the list
                of passages from the description file
                and asks user to choose passage.
        """
        path_to_description = Path(path_to_description)

        if select_in_process:
            selected_passage = self.select_passage(path_to_description)

        with open(path_to_description, "r") as read_file:
            passage = json.load(read_file)["passages"][selected_passage]

        style = str.split(passage["style"], "_")
        self.is_linear = bool(self.linear_or_circular[style[0]])
        self.is_manual = bool(self.manual_or_auto.get(style[-1], 0))
        self.passage_id = selected_passage

        result = [] if self.is_manual else {}

        for passage_iter in passage["points"]:
            for camera in passage_iter:
                key = re.search(self.pattern, camera["filename"])[1]
                if not self.is_manual:
                    result[key] = camera["camera"]["pose"]["position"]
                else:
                    result.append(key)

        return result

    def select_passage(self, path_to_description: PathLikeObject) -> int:
        """Output the list of passages and ask user to make choice."""

        print("Extracting information about available passages...\n")
        # Read the entire description file
        with open(path_to_description, "r") as read_file:
            passages = json.load(read_file)["passages"]

        # Output information about each passage
        for id, passage in enumerate(passages):

            number_of_images = 0

            for sub_passage in passage["points"]:
                number_of_images += len(sub_passage)

            print(
                f"Passage: {passage['style']}. Id: {id}. Number of images: {number_of_images}"
            )

        users_choice = input("Input selected passage id: ")

        return int(users_choice)

    def find_neighbours(self):
        """If the passage doesn't contain any geometric
        information, two closest images in image the list
        will be the neighbours.
        """
        Poses.find_neighbours(self, manual=self.is_manual)
