# Необходимые модули
from pathlib import Path
import os
from hloc import extract_features, match_features, match_dense, reconstruction, visualization, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

# Снимки храма
images = Path('datasets/South Building/')


# Вывод
outputs = Path('outputs/')
outputs_loftr = outputs / 'LoFTR'
outputs_netvlad = outputs / 'netVLAD'

os.mkdir(outputs)
os.mkdir(outputs_loftr)
os.mkdir(outputs_netvlad)

# Пары изображений (файл .txt)
sfm_pairs = outputs_netvlad / 'pairs-sfm.txt'

# Structure from Motion вывод
sfm_dir_loftr = outputs_loftr / 'sfm'


# Признаки
features_loftr = outputs_loftr / 'features.h5'

# Мэчи
matches_loftr = outputs_loftr / 'matches.h5'

# Retrieval
retrieval_conf = extract_features.confs['netvlad']

# Параметры моделей 
# LoFTR:
LoFTR_conf = match_dense.confs['loftr']



# Изображения (10)
references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

# Вывод
print(len(references), "mapping images")


retrieval_path = extract_features.main(retrieval_conf, images, outputs_netvlad)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


# Извлекаем признаки из изображений в папке images. Результат в "features.h5".
features, matches = match_dense.main(LoFTR_conf, sfm_pairs, images, matches=matches_loftr, features=features_loftr)

model = reconstruction.main(sfm_dir_loftr, images, sfm_pairs, features, matches, image_list=references)