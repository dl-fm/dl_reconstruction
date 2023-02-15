import h5py
import numpy as np

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="Converting from .bin to .h5 function.")
parser.add_argument("input_dir", type=str, default="./")

args = parser.parse_args()
depth_path = Path(args.input_dir)

hdf5_file_read = h5py.File(depth_path, "r")
gt_depth = hdf5_file_read.get("/depth")
gt_depth = np.array(gt_depth)
print(f"shape of array: {gt_depth.shape}\n")
print(gt_depth)
hdf5_file_read.close()
