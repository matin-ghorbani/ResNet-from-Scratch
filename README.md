# ResNet Implementation from Scratch using PyTorch

This repository contains an implementation of the Residual Network (ResNet) architecture from scratch using PyTorch. ResNet is a deep convolutional neural network that won the ImageNet competition in 2015 and introduced the concept of residual connections to address the problem of vanishing gradients in very deep networks.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [References](#references)

## Overview

ResNet is a highly influential architecture that allows the training of very deep neural networks by introducing residual blocks. These blocks use skip connections (also known as identity mappings) to allow gradients to flow through the network more easily, mitigating the vanishing gradient problem that occurs when training deep networks.

![ResNet Architecture](https://media.geeksforgeeks.org/wp-content/uploads/20200424011138/ResNet.PNG)

![Model Values](https://pytorch.org/assets/images/resnet.png)

## Architecture

This implementation supports the following ResNet variants:

- ResNet-50
- ResNet-101
- ResNet-152

Each variant differs in the number of layers (blocks) used in the network:

- **ResNet-50**: 50 layers deep (3, 4, 6, 3 blocks per layer)
- **ResNet-101**: 101 layers deep (3, 4, 23, 3 blocks per layer)
- **ResNet-152**: 152 layers deep (3, 4, 36, 3 blocks per layer)

The basic building block of ResNet is a residual block, which consists of three convolutional layers with batch normalization and ReLU activation functions. The key feature is the skip connection that bypasses the block, adding the input directly to the output, which helps in training deep networks by preserving gradient flow.

![Residual Block Architecture](https://linkinpark213.com/images/resnet/residual_blocks.png)

## Code Explanation

### Residual Block

The `Block` class defines a residual block. Each block contains:

1. A 1x1 convolution layer for reducing the dimensionality.
2. A 3x3 convolution layer for processing the feature map.
3. A 1x1 convolution layer for restoring the dimensionality.
4. Batch normalization and ReLU activation after each convolution.
5. An optional identity downsample layer, used when the input and output dimensions do not match.

```python
class Block(nn.Module):
    ...
```

### ResNet Class

The `ResNet` class defines the full network architecture. It begins with a standard convolutional layer and a max-pooling layer, followed by four main layers, each containing several residual blocks. The network ends with an average pooling layer and a fully connected layer for classification.

```python
class ResNet(nn.Module):
    ...
```

### ResNet Variants

Three functions are provided to create different versions of the ResNet architecture:

- `resnet50()`
- `resnet101()`
- `resnet152()`

```python
def resnet50(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 6, 3], img_channels, num_classes)

def resnet101(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 23, 3], img_channels, num_classes)

def resnet152(img_channels=3, num_classes=1000):
    return ResNet([3, 4, 36, 3], img_channels, num_classes)
```

## Usage

To run the network, simply execute the main() function in the script. This will create an instance of the ResNet-152 model, pass a random tensor through it, and print the output size.

***First you need to install PyTorch:***

```bash
pip install torch
```

then you can run the following command:

```bash
python ResNet.py
```

This indicates that the model has processed two input images (batch size = 2) and produced a vector of size 1000 for each image, corresponding to the 1000 classes in the dataset.

## References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - The original paper introducing ResNet.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Official PyTorch documentation.
- [ResNet on Towards Data Science](https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8) - Article explaining ResNet in depth.
