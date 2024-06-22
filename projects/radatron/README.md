# Radatron
Radatron: Accurate Detecton using Multi-resolution Cascaded MIMO Radar, ECCV 2022.

## Prerequisites

- Python 3.6
- Pytorch 1.7.1
- Detectron2
- Pycocotools

Radatron uses a slightly old version of Detectron2 (i.e., [0.4](https://github.com/facebookresearch/detectron2/releases/tag/v0.4)). To download and install the compatible version, follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 

Install Radatron
```
git clone https://github.com/waleedillini/radatron.git
cd radatron && pip install -e .
```

## Prepare Data

Download the processed Radatron dataset [here](https://uofi.box.com/v/radatrondataset). After extracting the files, the directory should look like this:
```
# Radatron Dataset Record
|-- radatron_dataset
    |-- day1
        |-- RGB
        |-- GT
        |-- heatmap_HighRes
        |-- heatmap_LowRes
        |-- heatmap_NoFix
        |-- heatmap_1chip
    |-- day2
        |-- RGB
        |-- GT
        |-- heatmap_HighRes
        |-- heatmap_LowRes
        |-- heatmap_NoFix
        |-- heatmap_1chip
    |-- day3
        |-- RGB
        |-- GT
        |-- heatmap_HighRes
        |-- heatmap_LowRes
        |-- heatmap_NoFix
        |-- heatmap_1chip
    |-- day4
        |-- RGB
        |-- GT
        |-- heatmap_HighRes
        |-- heatmap_LowRes
        |-- heatmap_NoFix
        |-- heatmap_1chip
```

More information about the dataset can be found at the dataset documentation page [here](https://github.com/waleedillini/radatronDataset).

## Train Radatron
```
python ./tools/train.py --config ./configs/radatron_config.yaml
```

## Evaluate Radatron
```
python ./tools/eval.py --config ./configs/radatron_config.yaml
```
