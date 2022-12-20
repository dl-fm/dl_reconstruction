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
outputs_netvlad = outputs / 'netVLAD'
outputs_loftr_aachen = outputs / 'LoFTR_Aachen'

os.mkdir(outputs)
os.mkdir(outputs_netvlad)
os.mkdir(outputs_loftr_aachen)

# Пары изображений (файл .txt)
sfm_pairs = outputs_netvlad / 'pairs-sfm.txt'

# Structure from Motion вывод
sfm_dir_loftr_aachen = outputs_loftr_aachen / 'sfm'


# Признаки
features_loftr_aachen = outputs_loftr_aachen / 'features.h5'

# Мэчи
matches_loftr_aachen = outputs_loftr_aachen / 'matches.h5'

# Retrieval
retrieval_conf = extract_features.confs['netvlad']

# Параметры моделей 
# LoFTR:
LoFTR_aachen_conf = match_dense.confs['loftr_aachen']


# Изображения (10)
references = [p.relative_to(images).as_posix() for p in (images).iterdir()]

# Вывод
print(len(references), "mapping images")


retrieval_path = extract_features.main(retrieval_conf, images, outputs_netvlad)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


# Извлекаем признаки из изображений в папке images. Результат в "features.h5".
features, matches = match_dense.main(LoFTR_aachen_conf, sfm_pairs, images, matches=matches_loftr_aachen, features=features_loftr_aachen)
model = reconstruction.main(sfm_dir_loftr_aachen, images, sfm_pairs, features, matches, image_list=references)
