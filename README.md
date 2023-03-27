<div align="center">

# SynthNet Transfer Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

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
