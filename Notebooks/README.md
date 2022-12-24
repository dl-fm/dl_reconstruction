# LoFTR

Detector-Free Local Feature Matching with Transformers. The authors were inspired by the success of Super Glue and decided to adapt the Transformer architecture for matching purposes. It is also proposed to use a unified architecture for detecting key points and matching, unlike typical approaches.

<p align="center">
  <a href="https://arxiv.org/abs/2104.00680"><img src="images/GGIMpDo2Kok.jpg" width="100%"/></a>
  <br /><em>Архитектура LoFTR</em>
</p>

LoFTR has 4 main components. In the first component, for two images A and B, a convolutional neural network (Resnet-18) constructs two feature maps of different sizes for each image. Smaller feature maps are converted into one-dimensional vectors and supplied with positional information, after which they are fed to the LoFTR module with self-attention and cross-attention mechanisms. From the received representations, a confidence matrix is further obtained, on the basis of which approximate comparisons are extracted. For each such comparison, a local window on the feature map is considered and final matches are selected based on this window.

Model reaches SOTA results in both Visual Localization и Relative Pose Estimation problems. For one image pair with 640х480 resolution model outputs result for 116 ms with RTX 2080Ti.

## Demo

To run models locally, you need to have the `hloc` project
[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization).
Or try Google Colab if you don't have a lot of resources. ➡️ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fsE028H1pbqTwena6OZbLRChfHdMPYGk?usp=sharing)

## Setup [Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)

For `hloc` Python >=3.7 and PyTorch >=1.1. are necessary. Command line:
```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

All dependicies are described in `requirements.txt`. 
