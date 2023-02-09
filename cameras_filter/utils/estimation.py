"""
utils.estimate

This module provides a simple algorithm evaluation method.
"""
from sklearn.metrics import recall_score, precision_score, f1_score
import numpy as np

from typing import Tuple, Union, Optional
from pathlib import Path
import sys
import os

# sys.path.append("../")

from sample_generation import add_noise
from main import main as run_filter

PathLikeObject = Union[str, Path]


def score_points(cam_filter: dict, noised: set) -> Tuple:
    """Calculate 3 metrics:
        -recall
        -precision
        -f1_score
    See more information at
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

    Parameters
        --------------
        cam_filter : dict
            Camera filter result, which contains
            image_id's as keys and 0/1 as predict.

        noised : set
            This parameter contains id's of each
            image of noised data.
    """
    y_pred = np.array(list(cam_filter.values()))

    noised = {key: int(key in noised) for key in cam_filter.keys()}
    y_true = np.array(list(noised.values()))

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f_1_score = f1_score(y_true, y_pred)

    return recall, precision, f_1_score


def main(
    images_path: PathLikeObject,
    description_file: Optional[PathLikeObject] = None,
    selected_passage: Optional[int] = None,
    algorithm_softness: float = 0.85,
    number_of_tests: int = 5,
    number_of_passages: int = 10,
    noised_data_proportion=0.15,
    noise_scale=1,
):

    prec = []
    rec = []
    f_1 = []

    if selected_passage is None:
        select_in_process = True
    else:
        select_in_process = False

    images_subset = Path(
        str(Path(images_path).parent / Path(images_path).stem) + "_sampled.bin"
    )

    for i in range(number_of_tests):
        for passage_id in range(number_of_passages):

            noised, images_subset = add_noise(
                path_to_images=images_path,
                path_to_output=images_subset,
                probability=noised_data_proportion,
                noise_scale=noise_scale,
            )

            filtering_result = run_filter(
                images_path=images_subset,
                softness=algorithm_softness,
                description_file=description_file,
                selected_passage=selected_passage,
                select_in_process=select_in_process,
            )["camera_filter"]

            result = score_points(filtering_result.cameras_filter, noised)

            prec.append(result[0])
            rec.append(result[1])
            f_1.append(result[2])

            os.remove(images_subset)

    recall = np.mean(rec)
    precision = np.mean(prec)
    f_1_score = np.mean(f_1)

    print(
        f"Average recall-score: {recall}.",
        f"Average precision-score: {precision}",
        f"Average F1-score: {f_1_score}",
        sep="\n",
    )

    return recall, precision, f_1_score
