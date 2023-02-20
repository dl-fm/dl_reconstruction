# Enter $path_to_set of depth maps
# and $output_file.

import os
import struct
from pathlib import Path
import argparse

import h5py
import numpy as np
from scipy import ndimage


def read_array(path: os.PathLike, window_size: int = 5) -> np.ndarray:
    """Converting from .bin to .h5 function.

    Extract ndarray from entered .bin file,
    preproccess it with smoothing aloritm
    and rewrite in .h5 format.
    """

    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)

        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)

        array = np.fromfile(fid, np.float32)

    array = array.reshape((width, height, channels), order="F")
    array = np.transpose(array, (1, 0, 2)).squeeze()

    return ndimage.median_filter(array, size=window_size)


def main():
    parser = argparse.ArgumentParser(
        description="Converting from .bin to .h5 function."
    )
    parser.add_argument("input_dir", type=str, default="./")
    parser.add_argument("output_dir", type=str, default="./")
    args = parser.parse_args()
    inp_dir, out_dir = args.input_dir, args.output_dir

    files = os.listdir(inp_dir)
    assert len(files) != 0, "There are no files in the input folder..."

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 1

    for image in os.listdir(inp_dir):

        depth_arr = read_array(inp_dir / image)

        h = h5py.File(out_dir / f"{image[:-4]}.h5", "w")
        h.create_dataset("/depth", data=depth_arr)
        h.close()

        if not count % 10:
            print(f"{count} depth maps have been already converted and preprocessed.")
        count += 1


if __name__ == "__main__":
    main()
