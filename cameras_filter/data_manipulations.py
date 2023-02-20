from typing import Union, Tuple, Sequence, Optional
from pathlib import Path
import re
import os

from utils.read_write_model import (
    read_images_binary,
    read_images_text,
    write_images_binary,
    write_images_text,
)
from poses_object import Poses
from reconstruction_poses import Reconstruction
from passage_poses import Passage

PathLikeObject = Union[str, Path]


def select_images(
    reconst_images_path: Union[str, Path],
    image_subset: Sequence,
    delete: bool = False,
    output_file: Optional[PathLikeObject] = None,
) -> Tuple:
    """Make manipulations with the
    reconstruction 'images.bin' file
    like deleting and extracting a
    subset of the images.

    Parameters
    --------------
        reconst_images_path : PathLikeObject
            Path to the file with image information
            of '.txt' or '.bin' extensions.

        image_subset : Sequence
            Select the images, which you want
            to delete or extract

        delete: bool = False
            Deleting if it is true, extracting if it's false.

        output_file: Optional[PathLikeObject] = None
            Path to the output file. If user sets it
            with the path to the source file, the
            file will be rewritten. If it's None,
            file will get the default name
            'images_subset.*' with the extension of
            the source images file.
    """

    path_to_images = Path(reconst_images_path)
    path_to_output = (
        Path(output_file)
        if output_file
        else Path(path_to_images.parent) / f"images_subset{Path(path_to_images).suffix}"
    )

    read_method = (
        read_images_text
        if str(reconst_images_path).endswith(".txt")
        else read_images_binary
    )

    images = read_method(path_to_images)

    mark = "delet" if delete else "extract"
    print(f"{mark.capitalize()}ing given subset from reconstruction...")
    subset_images = extract_delete_images(images, image_subset, delete)

    if path_to_images == path_to_output:
        os.remove(path_to_images)
        path_to_output = Path(
            str(path_to_output)[:-4] + f"_{mark}ed" + str(path_to_output)[-4:]
        )

    write_method = (
        write_images_text
        if str(path_to_output).endswith(".txt")
        else write_images_binary
    )
    write_method(subset_images, path_to_output)

    print(
        f"{len(subset_images)} images out of {len(images)} were {mark}ed ({round(len(subset_images)/len(images)*100, 1)}%).\n"
    )

    return subset_images, path_to_output


def extract_delete_images(images: dict, image_subset: Sequence, delete: bool = False):
    """Delete or extract necessary data."""

    pattern = Poses.pattern  # Extract image id from filename
    if delete:
        subset_images = {
            key: image
            for key, image in images.items()
            if not re.search(pattern, image[4])[1] in image_subset
        }
    else:
        subset_images = {
            key: image
            for key, image in images.items()
            if re.search(pattern, image[4])[1] in image_subset
        }
    return subset_images


def main(
    reconst_images_path: PathLikeObject,
    description_file: PathLikeObject,
    output_file: Optional[PathLikeObject] = None,
    selected_passage: Optional[int] = None,
) -> Tuple[PathLikeObject, int]:
    """Extract necessary passage subset directly
    from images.bin (.txt) file and write it to
    another file.

    Parameters
        --------------
        reconst_images_path : PathLikeObject
            Path to the file with image information
            of '.txt' or '.bin' extension.

        description_file: PathLikeObject
            Path to the description file of
            '.json' extension.

        output_file: Optional[PathLikeObject] = None
            Path to the output file.

        selected_passage: Optional[int] = None
            The id of the selected passage
            in the list of the passages. If
            it's None, function runs in the
            'select_in_process' mode.

        Read the 'select_images' docstring for more
        information.
    """

    select_in_process = True if selected_passage is None else False

    passage = Passage(
        description_file,
        select_in_process=select_in_process,
        selected_passage=selected_passage,
    )

    reconst_images_path = Path(reconst_images_path)
    if output_file is None:
        output_file = (
            Path(reconst_images_path.parent)
            / f"images_passage_id_{passage.passage_id}{Path(reconst_images_path).suffix}"
        )

    select_images(
        reconst_images_path=reconst_images_path,
        image_subset=passage.images,
        delete=False,
        output_file=output_file,
    )

    return output_file, passage.passage_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract passage")

    parser.add_argument("images_path", type=str, default="./")
    parser.add_argument("description_file", type=str, default="./")
    parser.add_argument("selected_passage", type=str, default="./")

    args = parser.parse_args()
    reconst_images_path, description_file, selected_passage = (
        args.images_path,
        args.description_file,
        args.selected_passage,
    )

    main(
        reconst_images_path=reconst_images_path,
        description_file=description_file,
        selected_passage=selected_passage,
    )
