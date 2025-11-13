# Reconstruction of [HEART](https://www.sciencedirect.com/science/article/pii/S153204642400159X)

This repository contains the implementation for pre-training and fine-tuning the **HEART** model on the **MIMIC-III** and **eICU** clinical datasets.

---

## Project Structure

- `finetune.py`: the pipeline for finetuning the model on downstream tasks  
- `pretrain.py`: the pipeline for pretraining the model on the pretraining task  
- `dataset/`: the code and folders for data processing  
  - `eICU-raw/`: raw eICU `.csv.gz` files  
  - `MIMIC-III-raw/`: raw MIMIC-III `.csv.gz` files  
  - `eICU.ipynb`: dataset preprocessing notebook for eICU  
  - `MIMIC-III.ipynb`: dataset preprocessing notebook for MIMIC-III  
- `models/`: model implementations  
  - `gnn.py`: implementation of the graph attention for the encounter-level attention  
  - `HEART.py`: implementation of the pretraining and finetuning model  
  - `transformer_rel.py`: implementation of the transformer with heterogeneous relations  
  - `transformer.py`: implementation of the transformer  
- `utils/`: utility functions including data loading pipeline
## 1. Setup and Installation

### 1.1 Environment Setup

Clone the repository:

```bash
git clone https://github.com/juyujing/new-HEART.git
cd new-HEART
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate heart_env
```

### Hardware Requirement

This project was developed and tested on a machine equipped with:

- **NVIDIA GPU** (CUDA-capable)
- **CUDA Driver Version: 12.8**

Please ensure that your GPU driver and CUDA runtime are compatible with the required versions.

---

## 1.2 Data Preparation

### Step 1. Download Raw Data

You must obtain access and download the raw datasets from their official sources:

- **MIMIC-III:** https://mimic.mit.edu/docs/iii/  
- **eICU:** https://eicu-crd.mit.edu/about/eicu/

Place the downloaded `.csv.gz` files into:

```bash
dataset/MIMIC-III.ipynb
dataset/eICU.ipynb
```

---

### Step 2. Download GAMENet Data

GAMENet provides additional necessary data processing resources.

```bash
cd dataset/
git clone https://github.com/sjy1203/GAMENet.git
cd ..
```

---

### Step 3. Run Preprocessing Notebooks

Run the preprocessing notebooks to generate training-ready files:

```bash
dataset/MIMIC-III.ipynb
dataset/eICU.ipynb
```

## 2. How to Run

The workflow consists of two main stages:

- **Pre-training**
- **Fine-tuning**

---

## 2.1 Pre-training

To run the default pre-training pipeline on the MIMIC-III dataset:

```bash
python pretrain.py
```

### Example — Pre-training on eICU

Run pre-training on the **eICU** dataset, using **GPU 1**, with a **0.4 mask rate**, for **100 epochs**:

```bash
python pretrain.py --dataset eicu --device 1 --mask_rate 0.4 --epochs 100
```

Additional hyperparameters such as learning rate, optimizer type, batch size, hidden dimensions, dropout rate, etc., can be specified using command-line flags.

## 2.2 Fine-tuning
To fine-tune the model on downstream tasks (default: mortality prediction on MIMIC-III, loading the pre-trained model from epoch 30):

```bash
python finetune.py
```

### Example — Fine-tuning for 12-month Re-diagnosis
Fine-tune the model on the next 12-month re-diagnosis task, using the model pre-trained for 50 epochs, with a learning rate of 1e-5:

```bash
python finetune.py --task next_diag_12m --pretrain_epoch 50 --lr 1e-5
```

### Example — Evaluation Only (No Training)
Evaluate an existing fine-tuned model (e.g., 10.pt) for the readmission task without performing training:

```bash
python finetune.py --task readmission --eval
```

You may also adjust flags for batch size, evaluation mode, logging, or checkpoint paths as needed.

---

# Configuration & Arguments
Both pretrain.py and finetune.py support comprehensive command-line argument configurations through argparse.

To list all available options, run:

```bash
python pretrain.py --help
python finetune.py --help
```
