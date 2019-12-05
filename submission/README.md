# Universal Style Transfer via Feature Transforms

In this project, we experiment a universal style transfer method proposed for transferring arbitrary visual styles to content images. We conduct an Encoder-Decoder training on Microsoft COCO dataset. Afterwards, we apply the whitening and coloring methods introduced in the paper for style transfer. Experimental results demonstrate our performance. And we demonstrate the effectiveness of our algorithm by generating high-quality style transferred images with comparisons the results in the original paper.

## Requirements
* Python 3
* Tensorflow
* scikit-image

Requirements can be found in file `requirement.txt`. To install required packages, please run

```pip install -r requirement.txt```

## Style Transform
```python stylize.py --target_layers relu1_1 relu2_1 relu3_1 relu4_1 relu5_1 --alpha 0.6 --style-path 'sample --content-path /path/to/content_img --output-path /path/to/output_img```

* `--content-path` Path to content image 
* `--style-path` Path to sytle image
* `--output-path` Path to output folder
* `--target-layers` Layers in VGG 19 model for WCT transfroms
* `--alpha`  [0,1] Strength of feature transform
* `--checkpoint-dir` Path to checkpoint folder

The detailed instuction can be found in Jupyter Notebook [Universal Style Transfer via Feature Transforms](Universal_Style_Transfer_via_Feature_Transforms.ipynb)

## Files and Folders
There are four main Python files for the Style Transfer:
* `model.py` The Encoder_decoder class in this file loads VGG 19 model and generates encoders and decoders
* `train.py` This file contents functions for train the decoders 
* `utils.py` This file contents functions for image opeations and feature transfer
* `stylize.py` This file contents functions for image stylization

There are several folders:
* `models` This folder contents VGG 19 model(whill be download automatically) and checkpoints for pre-trained decoders
* `train_images` This folder contents images for training set
* `val_images` This folder contents images for validation set
* `test_images` This folder contents images for test set
* `samples` This folder contents samples for content and style images 
* `output` This is the default folder for output images
* `ckpts` Checkpoint folder for training

## Training decoders
### Download data 

To train the decoders, please download [MS COCO images](http://mscoco.org/dataset/#download) for content data first.
```shell
cd train_images 
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
cd ..
```

```shell
cd test_images 
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip
cd ..
```

```shell
cd val_images 
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
cd ..
```

Then use the function `train_decoder` from `train.py` to train the decoder. Or we can train the decoder by:

```python stylize.py --target-decoder relu3_1 --checkpoint-dir ckpts```