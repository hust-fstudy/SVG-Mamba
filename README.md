# SVG-Mamba: An Efficient Local-to-Global Sparse Voxel Graph Mamba for Event Stream Recognition

The complete code will be released after the paper is published.

## Overview

We propose a sparse voxel graph Mamba (SVG-Mamba), which combines message-passing graph neural networks (GNNs) with state space models (SSMs) to capture both local and global spatiotemporal relationships among event voxels. First, we convert event data into sparse voxel graphs to preserve its sparsity. To extract robust spatial semantics and motion cues, we employ vertex feature encoding (VFE) to perform linear integration of internal event points in each graph vertex along three temporal directions. Subsequently, we design a multi-stage local feature aggregation (LFA) block that integrates point-wise spatial encoding  (PWSE), attention pooling units (APU), and dilated residual connections (DRC), facilitating the network to focus on fine-grained spatiotemporal features. Finally, we implement long-range temporal modeling that expands from local vertex receptive fields to global context through hierarchical dynamic farthest vertex sampling (DFVS) and Mamba-based global attention. This lightweight model is designed for event-based object classification and action recognition, achieving a trade-off between efficiency and accuracy.

![Framework](./assets/Framework.svg)

## Performance

Extensive experimental results show that our proposed SVG-Mamba not only achieves higher accuracy but also exhibits significant advantages in efficiency for event-based visual tasks such as object classification and action recognition.

![VisPer](./assets/VisPer.svg)

## Installation

### Requirements

All the codes are tested in the following environment:

- Linux (Ubuntu 20.04)
- Python 3.12
- PyTorch 2.4.0
- CUDA 11.8

### Dataset Preparation

All datasets should be downloaded and placed within the `dataset` directory, adhering to the folder naming rules and structure specified for the examples (`N-Caltech101` and `DvsGesture` datasets) as provided in the project.

## Quick Start

Clone the repository to your local machine:

```
git clone https://github.com/hust-fstudy/SVG-Mamba
cd SVG-Mamba
```

Once the dataset is specified in the `dataset_dict` dictionary within the `main` function of the `run_recognition.py` file, we can train and test it using the following command:

```bash
python run_recognition.py
```

