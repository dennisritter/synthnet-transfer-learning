<div align="center">

# SynthNet Transfer Learning

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

<!-- [![Paper](img)](link)
[![Conference](img)](link) -->

</div>

## Description

SynthNet Transfer Learning aims to provide a generic Framework to perform transfer learning on pretrained image classification networks.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11
conda activate myenv

install pytorch (v2.0) according to instructions
https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

- Get the Visda2017 Dataset from ![the official website](http://ai.bu.edu/visda-2017/) and store it under `data/visda2017`.
  Ensure the following directory structure: `data/visda2017/<split>/<class>/<images>`
- Logging is setup for ![Weights & Biases](https://wandb.com)

```bash
# Test run using reduced dataset to ensure everything works
python python src/train.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
