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
outputs_netvlad = outputs / "netVLAD"
outputs_loftr_aachen = outputs / "LoFTR_Aachen"

for fold in (outputs, outputs_loftr_aachen, outputs_netvlad):
    try:
        fold.mkdir(parents=True, exist_ok=True)
    except:
        raise Exception("Output folders creating problem.")

print(os.path.abspath(outputs))

# Pairs
sfm_pairs = outputs_netvlad / "pairs-sfm.txt"

# Structure from Motion
sfm_dir_loftr_aachen = outputs_loftr_aachen / "sfm"


# Fetures
features_loftr_aachen = outputs_loftr_aachen / "features.h5"

# Matches
matches_loftr_aachen = outputs_loftr_aachen / "matches.h5"

# Retrieval
retrieval_conf = extract_features.confs["netvlad"]

# Configs
# LoFTR:
LoFTR_aachen_conf = match_dense.confs["loftr_aachen"]


references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

print(f"Mapping images{len(references)}")


retrieval_path = extract_features.main(retrieval_conf, images, outputs_netvlad)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


# Featuring and matching.
features, matches = match_dense.main(
    LoFTR_aachen_conf,
    sfm_pairs,
    images,
    matches=matches_loftr_aachen,
    features=features_loftr_aachen,
)
model = reconstruction.main(
    sfm_dir_loftr_aachen, images, sfm_pairs, features, matches, image_list=references
)
