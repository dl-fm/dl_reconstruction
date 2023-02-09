from .read_write_model import qvec2rotmat

import numpy as np


def world_coordinates(q_vec: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Convert COLMAP coordinate representation into world coordinates.

    Etract rotation matrix R from quaternion q_vec and calculate world
    coordinates by the -R^t * T formula.
    """
    t_vec = t_vec.reshape((-1, 1))
    return np.dot(-qvec2rotmat(q_vec).T, t_vec)


def colmap_coordinates(q_vec: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.dot(-qvec2rotmat(q_vec), X)


def add_vector(
    q_vec: np.ndarray, t_vec: np.ndarray, addition: np.ndarray
) -> np.ndarray:
    """Add noise to vector in Oxyz coordinates."""
    X = world_coordinates(q_vec, t_vec)
    X += addition
    return colmap_coordinates(q_vec, X)
