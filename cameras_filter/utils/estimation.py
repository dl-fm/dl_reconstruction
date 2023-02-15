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
<<<<<<< HEAD
=======
from data_manipulations import main as extract
>>>>>>> 3-add-filter-code

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
<<<<<<< HEAD
    description_file: Optional[PathLikeObject] = None,
    selected_passage: Optional[int] = None,
    algorithm_softness: float = 0.85,
    number_of_tests: int = 5,
    number_of_passages: int = 10,
    noised_data_proportion=0.15,
    noise_scale=1,
):
=======
    description_file: PathLikeObject,
    with_passage: bool = False,
    algorithm_softness: float = 0.85,
    number_of_tests: int = 5,
    number_of_passages: Optional[int] = 1,
    noised_data_proportion: float = 0.15,
    noise_scale: float = 1,
    uniform: bool = True,
):
    """Evaluate the algorithm using 3 popular ML quality metrics.

    Parameters
        --------------
        images_path : PathLikeObject
            Path to the file with image information
            of '.txt' or '.bin' extension.
        description_file: PathLikeObject
            Path to the description file of
            '.json' extension.
        with_passage: bool = False
            If it's true, tests every passage
            from description file.
        algorithm_softness : float = 0.85
            A real parameter with value between 0 and 1,
            which determines approximately how many images
            won't be deleted by the algorithm.
        number_of_tests: int = 5
            Every passage will be sampled, filtered and
            evaluated $number_of_tests times.
        number_of_passages: Optional[int] = 1
            How many passages does the description file
            contain. This parameter matters only if
            with_passage = True.
        noised_data_proportion: float = 0.15
            The fraction of data that will be
            noised.
        noise_scale: float = 1
            The noise variable will be scaled with
            the noise_scale parameter.
        uniform: bool = True
            Determines the type of noise. If it's
            True, the noise variable will have
            uniform distribution, else - normal.
    """
>>>>>>> 3-add-filter-code

    prec = []
    rec = []
    f_1 = []

<<<<<<< HEAD
    if selected_passage is None:
        select_in_process = True
    else:
        select_in_process = False

    images_subset = Path(
        str(Path(images_path).parent / Path(images_path).stem) + "_sampled.bin"
    )
=======
    passages = range(number_of_passages) if number_of_passages else [None]
>>>>>>> 3-add-filter-code

    for i in range(number_of_tests):
        for passage_id in range(number_of_passages):

<<<<<<< HEAD
            noised, images_subset = add_noise(
                path_to_images=images_path,
                path_to_output=images_subset,
                probability=noised_data_proportion,
                noise_scale=noise_scale,
=======
            images_subset = Path(
                str(Path(images_path).parent / Path(images_path).stem) + "_sampled.bin"
            )

            extract(
                reconst_images_path=images_path,
                description_file=description_file,
                output_file=images_subset,
                selected_passage=passage_id,
            )

            noised, images_subset = add_noise(
                path_to_images=images_subset,
                path_to_output=images_subset,
                probability=noised_data_proportion,
                noise_scale=noise_scale,
                uniform=uniform,
>>>>>>> 3-add-filter-code
            )

            filtering_result = run_filter(
                images_path=images_subset,
<<<<<<< HEAD
                softness=algorithm_softness,
                description_file=description_file,
                selected_passage=selected_passage,
                select_in_process=select_in_process,
=======
                description_file=description_file if with_passage else None,
                softness=algorithm_softness,
                selected_passage=passage_id,
>>>>>>> 3-add-filter-code
            )["camera_filter"]

            result = score_points(filtering_result.cameras_filter, noised)

<<<<<<< HEAD
            prec.append(result[0])
            rec.append(result[1])
=======
            rec.append(result[0])
            prec.append(result[1])
>>>>>>> 3-add-filter-code
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
