from pathlib import Path
from typing import Tuple, Union
import shutil
import os

import pycolmap
import plotly.graph_objects as go

from .viz_3d import init_figure, plot_reconstruction, plot_camera_colmap
from .read_write_model import read_images_binary, read_images_text

String_Path = Union[str, Path]


def visualize_without_cameras(path_to_sparse: String_Path) -> go.Figure:

    reconst = pycolmap.Reconstruction(path_to_sparse)

    fig = init_figure()

    plot_reconstruction(fig, reconst, color="rgba(0,0,255,0.5)", cameras=False)

    return fig


def visualize_cameras(
    fig: go.Figure,
    wrong_images_path: String_Path,
    wrong_images_sparse: String_Path,
    color: str = "rgba(0,0,255,0.5)",
) -> go.Figure:

    read_method = (
        read_images_binary
        if str(wrong_images_path).endswith(".bin")
        else read_images_text
    )

    images = read_method(wrong_images_path)

    model = pycolmap.Reconstruction(wrong_images_sparse)

    for image in images.values():
        pose = pycolmap.Image(tvec=image[2], qvec=image[1])
        if not image[0] in model.cameras.keys():
            print(image)
            print("=" * 20 + "\n", image[0], "\n" + "=" * 20)
        camera = model.cameras[image[0]]
        plot_camera_colmap(fig, pose, camera, color=color, name="Wrong camera!!!")

    return fig
