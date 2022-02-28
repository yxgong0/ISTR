# Irregular Scene Text Rectifier

This software implements the Irregular Scene Text Rectifier, an unattached irregular text rectification network with refined objective. For details, please refer to [our paper](https://www.sciencedirect.com/science/article/pii/S0925231221012315).

## Requirements
* [Python 3.7](https://www.python.org/)
* [PyTorch 1.7](https://pytorch.org/)
* [TorchVision](https://pypi.org/project/torchvision/)
* [Numpy](https://pypi.org/project/numpy/)
* [Pillow](https://pypi.org/project/Pillow/) 

## Data Preparation
Please arrange your own training dataset to a folder /path/to/train/data, which contains two sub-folders 'distorted' and 'corrected'. In the 'distorted' sub-folder, there are images which contain irregular text, and in the 'corrected' sub-folder, there are images which contain regular text. The corresponding images should have the same file name.

Then arrange your validating dataset to a folder /path/to/val/data, which directly contains irregular text images.

## Train and Test
To train a model, simply execute python3 train.py --dataPath /path/to/train/data --val_path /path/to/val/data --cuda. If you need to set other parameters, explore train.py for details.

## Citation
@article{gong2021unattached,   
  title={Unattached irregular scene text rectification with refined objective},   
  author={Gong, Yanxiang and Deng, Linjie and Zhang, Zhiqiang and Duan, Guozhen and Ma, Zheng and Xie, Mei},   
  journal={Neurocomputing},   
  volume={463},   
  pages={101--108},   
  year={2021},   
  publisher={Elsevier}   
}
