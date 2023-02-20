from pathlib import Path
from typing import Union, Tuple, Optional, Sequence
import json
import re
import os
import shutil
import argparse

import numpy as np

from passage_poses import Passage
from reconstruction_poses import Reconstruction
from data_manipulations import select_images
from utils.read_write_model import read_images_binary, read_images_text
from filter import CameraFilter
from utils.quaternion_transform import world_coordinates


PathLikeObject = Union[str, Path]


def save_filter(
    images_path: PathLikeObject,
    image_subset: Sequence,
    output_dir: PathLikeObject,
    sparse_dir: Optional[PathLikeObject] = None,
):
    """Split source data on 'filtered' and 'not filtered' images"""

    images_path = Path(images_path)
    output_dir = Path(output_dir)

    right_images_dir = output_dir / "right_positions"
    wrong_images_dir = output_dir / "wrong_positions"
    wrong_images_dir.mkdir(parents=True, exist_ok=True)
    right_images_dir.mkdir(parents=True, exist_ok=True)

    wrong_images_path = Path(wrong_images_dir) / "images.bin"
    right_images_path = Path(right_images_dir) / "images.bin"

    # Extracting and deleting filtered images
    select_images(
        reconst_images_path=images_path,
        image_subset=image_subset,
        delete=False,
        output_file=wrong_images_path,
    )
    select_images(
        reconst_images_path=images_path,
        image_subset=image_subset,
        delete=True,
        output_file=right_images_path,
    )

    # Copy other reconstruction files (cameras.bin, points.bin)
    sparse_dir = Path(images_path.parent) if sparse_dir is None else Path(sparse_dir)

    cameras_file_path = sparse_dir / "cameras.bin"
    points_file_path = sparse_dir / "points3D.bin"

    assert os.path.exists(
        cameras_file_path
    ), f"There is no 'cameras.bin' file in {sparse_dir}"
    assert os.path.exists(
        cameras_file_path
    ), f"There is no 'points3D.bin' file in {sparse_dir}"

    for dir in (right_images_dir, wrong_images_dir):
        shutil.copy(cameras_file_path, dir)
        shutil.copy(points_file_path, dir)


def main(
    images_path: PathLikeObject = Path("./sparse/images.bin"),
    description_file: Optional[PathLikeObject] = None,
    select_in_process: bool = False,
    selected_passage: int = 0,
    softness: float = 0.95,
    output_dir: Optional[PathLikeObject] = None,
    sparse_dir: Optional[PathLikeObject] = None,
) -> dict:
    """Run the full filtering algorithm from scratch.

    Set all necessary parameters.

    Parameters
        --------------
        images_path : PathLikeObject = ./sparse/images.bin
            Path to the file with image information
            of '.txt' or '.bin' extension.
        description_file: Optional[PathLikeObject] = None
            Path to the Augmented City output description.
            If it's None, runs algorithm with the
            COLMAP information only.
        select_in_process: bool = False
            If it's True, lets user choose passage_id
            before the alorithm starts.
        selected_passage: int = 0
            The id of the chosen passage.
        softness : float = 0.95
            A real parameter with value between 0 and 1,
            which determines approximately how many images
            won't be deleted by the algorithm.
        output_dir: Optional[PathLikeObject] = None
            Path to the directory, where
            right_images.bin and wrong_images.bin
            will be located. If it's None, the result
            won't be saved.
        sparse_dir : Optional[PathLikeObject] = None
            This parameter should be used, when the
            user's images.bin file is located not in
            its sparse reconstruction directory.
    """
    if not description_file is None:
        passage = Passage(
            file_with_poses=description_file,
            select_in_process=select_in_process,
            selected_passage=selected_passage,
        )
    else:
        passage = None

    images_path = Path(images_path)
    assert images_path.exists(), f"File {images_path} doesn't exist."

    reconst = Reconstruction(images_path)

    camera_filter = CameraFilter(reconst, passage)

    filtered = camera_filter.filter(softness=softness)

    if not output_dir is None:
        save_filter(
            images_path=images_path,
            image_subset=filtered,
            output_dir=output_dir,
            sparse_dir=sparse_dir,
        )

    return {"camera_filter": camera_filter, "filtered": filtered}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter reconstruction")

    parser.add_argument("input_dir", type=str, default="./")
    parser.add_argument("output_dir", type=str, default="./")
    parser.add_argument("path_to_images_dir", type=str, default=None)

    args = parser.parse_args()
    inp_dir, out_dir, path_to_images_dir = (
        args.input_dir,
        args.output_dir,
        args.path_to_images_dir,
    )

    result = main(images_path=inp_dir, output_dir=out_dir)

    path_to_images_dir = Path(path_to_images_dir)

    pattern = re.compile("[_]?([0-9]+).jpg", re.IGNORECASE)

    # Deleting images that are not in filtered file
    if path_to_images_dir is not None:
        for image in os.listdir(path_to_images_dir):
            ans = re.search(pattern, image)
            if ans:
                if (ans[1] in result["filtered"]) or (
                    ans[1] not in result["camera_filter"].reconst.images
                ):
                    print(path_to_images_dir / image)
                    os.remove(path_to_images_dir / image)
