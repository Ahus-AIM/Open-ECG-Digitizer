# Open ECG Digitizer

[![npj Digital Medicine](https://img.shields.io/badge/npj_Digital_Medicine-paper-00cc66.svg)](https://doi.org/10.1038/s41746-025-02327-1) [![ECG-Image-Database on Kaggle](https://img.shields.io/badge/Kaggle-ECG--Image--Database-20BEFF.svg)](https://www.kaggle.com/datasets/physionet/ecg-image-database) ![Tests](https://github.com/Ahus-AIM/Electrocardiogram-Digitization/actions/workflows/test.yml/badge.svg?branch=main) ![](https://img.shields.io/badge/%20style-google-3666d6.svg) [![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-31211/)

This repository provides a highly configurable tool for digitizing 12-lead ECGs, extracting raw time series data from scanned images or photographs (e.g., taken with a phone). It supports any subset of the 12 standard leads and is robust to perspective distortions and image quality variations.

<div style="display: flex; justify-content: center; gap: 30px;">
  <img src="assets/visual_abstract-img0.png" alt="Mobile phone photo" width="31%">
  <img src="assets/visual_abstract-img1.png" alt="Segmented mobile phone photo" width="31%">
  <img src="assets/visual_abstract-img2.png" alt="Segmented and perspective corrected mobile phone photo" width="36.55%">
</div>

## Features

- Extracts raw time series data from 12-lead ECG images
- Supports both scanned and photographed ECGs (with perspective correction)
- Works with any subset of leads
- Easily configurable via yaml config files

## File structure and module overview

Each component of the ECG digitization pipeline is modularized under [`src/model`](src/model).  

<p align="center">
  <img src="assets/pipeline-overview.svg" alt="Pipeline Overview" width="100%">
</p>

Below is an overview of their purpose and debugging relevance, in approximate execution order:

| Module | Description |
|:--------|:-------------|
| [`src/model/unet.py`](src/model/unet.py) | **Semantic segmentation network** - a U-Net model trained to identify ECG traces, grids, and background. Retrain or fine-tune it using [`src/train.py`](src/train.py) if it underperforms. You can modify the on-the-fly transforms to mimic your own data [`src/transform/vision.py`](src/transform/vision.py). |
| ([`src/model/dewarper.py`](src/model/dewarper.py)) | **Experimental full dewarping** - for folded or curved ECG paper. Not formally evaluated. Not recommended for flat papers, as perspective correction is more robust. Not enabled in the provided configuration YAML files. |
| [`src/model/perspective_detector.py`](src/model/perspective_detector.py) | **Perspective correction** - estimates and corrects projective distortions. Handles up to ~45° rotation. |
| [`src/model/cropper.py`](src/model/cropper.py) | **Cropping and bounding box extraction** - used to crop the image based on the location of the ECG leads. |
| [`src/model/pixel_size_finder.py`](src/model/pixel_size_finder.py) | **Grid size estimation** - autocorrelation-based template matching. Configure grid parameters (minor/major ratio, expected line counts) in your inference YAML in case this underperforms. |
| [`src/model/lead_identifier.py`](src/model/lead_identifier.py) | **Layout identification** - matches cropped regions to known ECG lead layouts using predefined templates. Update or prune templates in `src/config/lead_layouts_*.yml`. |
| [`src/model/signal_extractor.py`](src/model/signal_extractor.py) | **Segmentation-to-trace conversion** - converts segmented images into digitized voltage–time signals. Might set parts of signals to NaN in case of overlapping signals. |
| [`src/model/inference_wrapper.py`](src/model/inference_wrapper.py) | **Main orchestration script** - connects all components. |

**Questions or in need of help?** Contact elias.stenhede at ahus.no

## Installation

**Requirements:** Python 3.12 or later.

> [!NOTE]
> This setup has been tested on Ubuntu 24.04.2 and Debian 12 with CUDA. You need to install git-lfs to download the weights.

1. Ensure you have installed python3.12, git and git-lfs.
2. Clone the repository: ```git clone git@github.com:Ahus-AIM/Electrocardiogram-Digitization.git```
3. Navigate to the project_source_code folder.
4. Create and activate a virtual environment: ```python3.12 -m venv venv && source venv/bin/activate```
5. Install dependencies ```python3 -m pip install -r requirements.txt```
6. Download the pre-trained weights: ```git lfs pull```

## Running inference on a folder with images
1. Modify a config file with your paths and settings, for example [src/config/inference_wrapper_ahus_testset.yml](src/config/inference_wrapper_ahus_testset.yml)
2. Ensure that your config file points to a layout file containing your expected layouts, for example [lead_layouts_reduced.yml](src/config/lead_layouts_reduced.yml) or [lead_layouts_george-moody-2024.yml](src/config/lead_layouts_george-moody-2024.yml)
3. Run: ```python3 -m src.digitize --config src/config/your_config_file.yml```
4. You can also override the config file, for example: ```python3 -m src.digitize --config src/config/your_config_file.yml DATA.output_path=my_output/folder```

Each ECG starts with seed 42 by default. On the same hardware and software, its sampled grid points, noise and signals are reproducible across runs and independent of folder order. Use `--seed N` to select another sequence. PNG export uses up to six CPU processes. Use `--plot-workers 0` for in-process export.

> [!NOTE]
> The output values are expressed in **microvolts (µV)**.

## Training dataset
The dataset is publicly available on Hugging Face:
[huggingface.co/datasets/Ahus-AIM/Open-ECG-Digitizer-Development-Dataset](https://huggingface.co/datasets/Ahus-AIM/Open-ECG-Digitizer-Development-Dataset)

### Dataset characteristics
- Multiple grid types, colors and sizes
- Perspective distortions, varying illumination, and noise
- Pixel-level annotations for ECG traces, grid, and background

The dataset is intended to support:
- Retraining or fine-tuning of the segmentation network
- Development of alternative ECG digitization approaches

## Train on custom dataset
1. Change `data_path` for TRAIN, VAL and TEST in [src/config/unet.yml](src/config/unet.yml) to the locations of the custom dataset.
2. Run: ```python3 -m src.train```

## Mandatory Citation

If you use this code or dataset in your research, **please cite the following paper**:
```bibtex
@article{stenhede_digitizing_2026,
  title        = {Digitizing Paper {ECGs} at Scale: An Open-Source Algorithm for Clinical Research},
  author       = {Stenhede, Elias and Bjørnstad, Agnar Martin and Ranjbar, Arian},
  journal      = {npj Digital Medicine},
  year         = {2026},
  doi          = {10.1038/s41746-025-02327-1},
  url          = {https://doi.org/10.1038/s41746-025-02327-1},
  shorttitle   = {Digitizing Paper {ECGs} at Scale}
}
```

If you use ECG-Image-Database, please also cite the dataset paper.

```bibtex
@article{reyna_ecg_image_database_2026,
  author  = {Reyna, Matthew A. and {Deepanshi} and Weigle, Audrey T. and Koscova, Zuzana and Campbell, Kiersten and Kodthalu Shivashankara, Kshama and Saghafi, Soheil and Nikookar, Sepideh and Motie-Shirazi, Mohsen and Kiarashi, Yashar and Seyedi, Salman and Hassannia, Mohammadsina and Bj{\o}rnstad, Agnar M. and Stenhede, Elias and Ranjbar, Arian and Clifford, Gari D. and Sameni, Reza},
  title   = {{ECG}-image-database: large-scale paired {ECG} images and time-series with real-world artifacts; a foundation for computerized {ECG} digitization and analysis},
  journal = {Physiological Measurement},
  year    = {2026},
  volume  = {47},
  number  = {7},
  pages   = {075015},
  doi     = {10.1088/1361-6579/ae85b2}
}
```
