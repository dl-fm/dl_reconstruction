from pathlib import Path
import os
from hloc import (
    extract_features,
    match_features,
    match_dense,
    reconstruction,
    visualization,
    pairs_from_retrieval,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

images = Path("datasets/South Building/")


outputs = Path("outputs/")
outputs_loftr = outputs / "LoFTR"
outputs_netvlad = outputs / "netVLAD"
print(os.path.abspath(outputs))

for fold in (outputs, outputs_loftr, outputs_netvlad):
    try:
        fold.mkdir(parents=True, exist_ok=True)
    except:
        raise Exception("Output folders creating problem.")

# Pairs
sfm_pairs = outputs_netvlad / "pairs-sfm.txt"

# Structure from Motion
sfm_dir_loftr = outputs_loftr / "sfm"


# Features
features_loftr = outputs_loftr / "features.h5"

# Matches
matches_loftr = outputs_loftr / "matches.h5"

# Retrieval
retrieval_conf = extract_features.confs["netvlad"]

# Configs
# LoFTR:
LoFTR_conf = match_dense.confs["loftr"]


# Images
references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

print(f"Mapping images{len(references)}")


retrieval_path = extract_features.main(retrieval_conf, images, outputs_netvlad)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


# Featuring and matching
features, matches = match_dense.main(
    LoFTR_conf, sfm_pairs, images, matches=matches_loftr, features=features_loftr
)

model = reconstruction.main(
    sfm_dir_loftr, images, sfm_pairs, features, matches, image_list=references
)
