## About The Project

The goal of this research is to develop a DeepLabV3+ model with a ResNet50 backbone to perform binary segmentation on plant image datasets. Based on the presence or absence of a certain object or characteristic, binary segmentation entails splitting an image into discrete subgroups known as image segments which helps to simplify processing or analysis of the image by reducing the complexity of the image. Labeling pixels is a step in the segmentation process. Each pixel or piece of a picture assigned to the same category has a unique label. 


Plant pictures with ground truth binary mask labels make up the training and validation dataset. The project uses PyTorch, a well-known deep learning library, for model development, training, and evaluation.[^1] During the training process, the model is optimized using strategies like the Dice Loss, Adam optimizer, Reducing LR on Pleateau and Early Stopping. All the while, important metrics like Intersection over Union (IoU), Pixel Accuracy, and Dice Coefficient are kept track of.

[^1]: A Tensorflow implementation can be found [here](https://github.com/mukund-ks/DeepLabV3-Segmentation). 
 
  The Computer Vision Problems in Plant Phenotyping (CVPPP) Leaf Counting Challenge (LCC) 2017 Dataset provides 27 images of tobacco and 783 Arabidopsis images in separate folders. Using a camera with a single plant in its range of view, tobacco photos were gathered. Images of the Arabidopsis plant were taken with a camera that had a wider field of view and were later cropped.  The photographs were shot over a period of days from mutants or wild types, and they came from two different experimental settings, where the field of vision was different.

  Additionally, certain plants are slightly out of focus than others due to the wider range of view. Though, the backgrounds of most photographs are straightforward and static, occasionally, moss growth or the presence of water in the growing tray complicates the scene.
  For the purpose of obtaining ground truth masks for every leaf/plant in the picture, each image was manually labeled.


The ultimate objective of the project is to develop a strong model that can accurately segment plant-related regions inside photographs, which can have applications in a variety of fields, such as agriculture, botany, and environmental sciences. The included code demonstrates how to prepare the data, create the model's architecture, train it on the dataset, and assess the model's effectiveness using a variety of metrics.

## Working

The objective of binary segmentation, often referred to as semantic binary segmentation, is to categorize each pixel in an image into one of two groups: the foreground (object of interest), or the background. A powerful Encoder-Decoder based architecture for solving binary segmentation challenges, DeepLabV3+ with ResNet50 or ResNet101 as the backbone offers great accuracy and spatial precision.


### DeepLabV3+

Known for its precise pixel-by-pixel image segmentation skills, DeepLabV3+ is a powerful semantic segmentation model. It combines a robust feature extractor, such as ResNet50 or ResNet101, with an effective decoder. This architecture does a great job of capturing both local and global context information, which makes it suitable for tasks where accurate object boundaries and fine details are important. A crucial part is the Atrous Spatial Pyramid Pooling (ASPP) module, which uses several dilated convolutions to collect data on multiple scales. The decoder further improves the output by fusing high-level semantic features with precise spatial data. Highly precise segmentations across a variety of applications are made possible by this fusion of context and location awareness.

### ResNet Backbone

Residual Networks, often known as ResNets, are a class of deep neural network architectures created to address the vanishing gradient problem that can arise in very deep networks. They were first presented in the 2015 publication [*Deep Residual Learning for Image Recognition*](https://ieeexplore.ieee.org/document/7780459) by **Kaiming He et al**. ResNets have been extensively used for a number of tasks, including image classification, object recognition, and segmentation.

The main novelty in ResNets is the introduction of residual blocks, which allow for the training of extremely deep networks by providing shortcut connections (skip connections) that omit one or more layers. Through the use of these connections, gradients can pass directly through the network without disappearing or blowing up, enabling the training of far more complex structures.

ResNets are available in a range of depths, designated as ResNet-XX, where XX is the number of layers. The ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152 are popular variations. The performance of the deeper variations is better, but they also use up more processing resources.


### Modules used

* ***Encoder:*** The ResNet backbone's early layers are typically where the encoder is implemented. With growing receptive fields, it has numerous convolutional layers. These layers take the input image and extract low-level and mid-level information. The ASPP module is then given the feature maps the encoder produced.
  
  For pixel-wise predictions, the Encoder is essential in converting raw pixel data into abstract representations. It consists of several layers of convolutional and pooling procedures, arranged into blocks, that gradually increase the number of channels while decreasing the input's spatial dimensions. Because of its hierarchical structure, the model may capture aspects with varied levels of complexity, from simple edges and textures to complicated object semantics.

* ***Atrous Spatial Pyramid Pooling (ASPP):*** The ASPP module executes many convolutions with various dilation rates following the encoder. This records contextual data at various scales. Concatenated and processed outputs from several atrous convolutions are then used to create context-rich features.
  
  By gathering data from diverse scales and viewpoints, the ASPP module improves the network's comprehension of the items in a scene. It is especially useful for overcoming the challenges presented by items with varying sizes and spatial distributions. 

* ***Decoder:*** Through skip connections, the decoder module combines low-level features from the encoder with high-level features from the ASPP module. This method aids in recovering spatial data and producing fine-grained segmentation maps.
  
  This Module enables the network to generate precise and contextually rich segmentation maps by including skip links and mixing data from various scales. This is crucial for tasks like semantic segmentation, where accurate delineation of object boundaries is necessary for producing high-quality results.

* ***Squeeze & Excitation (SE):*** It is a mechanism made to increase the convolutional neural networks' representational strength by explicitly modeling channel-wise interactions. **Jie Hu et al.** first discussed it in their publication [*Squeeze-and-Excitation Networks*](https://ieeexplore.ieee.org/document/8578843) published in 2018. In order to enable the model to focus greater attention on crucial features, the SE Module seeks to selectively emphasize informative channels while suppressing less critical ones within the network. 
  
  By computing the average value of each channel across all spatial dimensions, the global average pooling method is used. The end result is a channel-wise descriptor that accurately reflects the significance of each channel in relation to the overall feature map. 

  The channels are then adaptively recalibrated using the squeezed information. Two fully connected layers are utilized for this. A non-linear activation function, also known as ReLU, is added after the first layer, which minimizes the dimensionality of the squeezed descriptor. A set of channel-wise excitation weights is produced after the second layer returns the dimensionality to the original number of channels. Each channel's weights indicate how much it should be boosted or muted.

  | ![Squeeze & Excite Diagram](diagrams/Squeeze-Excitation-Diagram.png) |
  | :------------------------------------------------------------------: |
  |                    *Squeeze & Excitation Module*                     |



## Getting Started

To get a local copy of this project up and running on your machine, follow these simple steps.

* Clone a copy of this Repository on your machine.
```console
git clone https://github.com/mukund-ks/DeepLabV3Plus-PyTorch.git
```

### Prerequisites

* Python 3.9 or above.

```console
python -V
Python 3.9.13
```

* CUDA 11.2 or above.

```console
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_22:08:44_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.2, V11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
```

### Installation

1. Move into the cloned repo.
```console
cd DeepLabV3Plus-PyTorch
```

2. Setup a Virutal Environment

```console
python -m venv env
```

3. Activate the Virutal Environment
```console
env/Scripts/activate
```

4. Install Dependencies

```console
pip install -r requirements.txt
```

> **Note**
> You can deactivate the Virtual Environment by using
> ```env/Scripts/deactivate```
 

## Usage

The Model can be trained on the data aforementioned in the [**About**](#about-the-project) section or on your own data.

* To train the model, use [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py)
```console
python train.py --help
```
```console
Usage: train.py [OPTIONS]

  Training Script for DeepLabV3+ with ResNet50 Encoder for Binary
  Segmentation.

  Please make sure your data is structured according to the folder structure
  specified in the Github Repository.

  See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

  Refer to the Options below for usage.

Options:
  -D, --data-dir TEXT        Path for Data Directory  [required]
  -E, --num-epochs INTEGER   Number of epochs to train the model for. Default
                             - 25
  -L, --learning-rate FLOAT  Learning Rate for model. Default - 1e-4
  -B, --batch-size INTEGER   Batch size of data for training. Default - 4
  -P, --pre-split BOOLEAN    Opt-in to split data into Training and Validaton
                             set.  [required]
  -A, --augment BOOLEAN      Opt-in to apply augmentations to training set.
                             Default - True
  -S, --early-stop BOOLEAN   Stop training if val_loss hasn't improved for a
                             certain no. of epochs. Default - True
  --help                     Show this message and exit.
```

* For Evaluation, use [`evaluation.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/evaluation.py)
```console
python evaluation.py --help
```
```console
Usage: evaluation.py [OPTIONS]

  Evaluation Script for DeepLabV3+ with ResNet50 Encoder for Binary
  Segmentation.

  Please make sure your evaluation data is structured according to the folder
  structure specified in the Github Repository.

  See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

  Refer to the Option(s) below for usage.

Options:
  -D, --data-dir TEXT  Path for Data Directory  [required]
  --help               Show this message and exit.
```
* An Example
```console
python train.py --data-dir data --num-epochs 80 --pre-split False --early-stop False
```
```console
python evaluation.py --data-dir eval_data
```

## Folder Structure

The folder structure will alter slightly depending on whether or not your training data has already been divided into a training and testing set.

- If the data is not already seperated, it should be in a directory called `data` that is further subdivided into `Image` and `Mask` subdirectories.
  
  - [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py) should be run with `--pre-split` option as `False` in this case.
  
    Example: ```python train.py --data-dir data --pre-split False```

> **Note**
> [`dataset.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/dataset.py) will split the data into training and testing set with a ratio of 0.2

```console
$ tree -L 2
.
├── data
│   ├── Image
│   └── Mask
└── eval_data
    ├── Image
    └── Mask
```

- If the data has already been separated, it should be in a directory called `data` that is further subdivided into the subdirectories `Train` and `Test`, both of which contain the subdirectories `Image` and `Mask`.

  - [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py) should be run with `--pre-split` option as `True` in this case.
  
    Example: ```python train.py --data-dir data --pre-split True```

```console
$ tree -L 3
.
├── data
│   ├── Test
│   │   ├── Image
│   │   └── Mask
│   └── Train
│       ├── Image
│       └── Mask
└── eval_data
    ├── Image
    └── Mask
```
* The structure of `eval_data` remains the same in both cases, holding `Image` and `Mask` sub-directories.

> **Note**
> The directory names are case-sensitive.
## Roadmap

See the [open issues](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b MyBranch`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push -u origin myBranch`)
5. Open a Pull Request

## License

Distributed under the Apache 2.0 License. See [LICENSE](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/LICENSE) for more information.

