# Self-supervised Learning for Video Correspondence Flow

This repository contains the code (in PyTorch) for the model introduced in the following paper

[Self-supervised Learning for Video Correspondence Flow](https://arxiv.org/abs/1905.00875)

by Zihang Lai, Weidi Xie

![Figure](figures/layout.png)

### Citation
```
@article{Lai19,
  title={Self-supervised Learning for Video Correspondence Flow},
  author={Lai, Z. and Xie, W.},
  journal={arXiv preprint arXiv:1905.00875},
  year={2019}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

The objective of this paper is self-supervised learning of feature embeddings from videos, 
suitable for correspondence flow, i.e. matching correspondences between frames over the video. We leverage the natural spatial-temporal coherence of appearance in videos, to create a pointer model that learns to reconstruct a target frame by copying colors from a reference frame.

We make three contributions:
_First_, we introduce a simple information bottleneck that enforces the model to learn robust features for correspondence matching, and avoids it learning trivial solutions, e.g. matching based on low-level color information.
_Second_, we propose to train the model over a long temporal window in videos. To make the model more robust to complex object deformation, occlusion, the problem of tracker drifting,
we formulate a recursive model, trained with scheduled sampling and cycle consistency.
_Third_, we evaluate the approach by first training on the Kinetics dataset using self-supervised learning, and then directly applied for DAVIS video segmentation and JHMDB keypoint tracking.
On both tasks, our approach has achieved state-of-the-art performance, especially on segmentation, we outperform all previous methods by a significant margin.


## Usage

### Dependencies

- [Python3.5](https://www.python.org/downloads/)
- [PyTorch(1.0.0)](http://pytorch.org)
- CUDA 9.0/9.2
- [Kinetics dataset](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)
- [DAVIS-2017](https://davischallenge.org/davis2017/code.html)
- [JHMDB](http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets)

### Train
To be released soon.

### Pretrained model

## Results
![DAVIS-2017 Results](figures/results.png) 
