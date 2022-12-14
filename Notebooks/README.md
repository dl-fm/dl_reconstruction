# LoFTR

Detector-Free Local Feature Matching with Transformers. Авторы были вдохновлены успехом SuperGlue и решили приспособить архитектуру Transformer для
целей мэтчинга. Также предложено использовать объединённую архитектуру для обнаружения ключевых точек, сопоставления векторов признаков и мэтчинга в отличие от типичных подходов.

<p align="center">
  <a href="https://arxiv.org/abs/2104.00680"><img src="images/GGIMpDo2Kok.jpg" width="100%"/></a>
  <br /><em>Архитектура LoFTR</em>
</p>

LoFTR имеет 4 основных компонента. В первом компоненте для двух изображений A и B свёрточная нейронная сеть (ResNet-18) конструирует по две карты признаков разных размеров. Меньшие карты признаков преобразуются в одномерные векторы и снабжаются позиционной информацией, после чего подаются на вход модулю LoFTR с механизмами самовнимания и кроссвнимания. Из полученных представлений далее получается доверительная матрица, на основе которой извлекаются примерные сопоставления. Для каждого такого сопоставления рассматривается локальное окно на карте признаков и на основе этого окна подбираются окончательные соответствия. 

Модель достигает SOTA результатов в задачах Visual Localization и Relative Pose Estimation. Для пары изображений разрешением 640х480 модель выдаёт результат за 116 мс на RTX 2080Ti.

## Демонстрация

Для запуска моделей локально необходимо иметь проект [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization).
Или попробовать colab ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fsE028H1pbqTwena6OZbLRChfHdMPYGk?usp=sharing)

## Установка [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)

Для `hloc` необходим Python >=3.7 и PyTorch >=1.1. Command line:
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

Все зависимости описаны в `requirements.txt`. 
