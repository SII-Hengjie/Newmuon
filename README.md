# Newmuon

This repository contains the implementation of **NewMuon** and **Muon** optimizers for large-scale deep learning tasks. The optimizers are tested on various models, and this repository provides scripts to run experiments with Adam, Muon, and NewMuon optimizers.

## Requirements

To get started, you will need to install the necessary dependencies. The required packages are specified in `requirements.txt`. You can install them by running:

```bash
pip install -r requirements.txt
````

The required libraries are:

* `torch`
* `transformer_engine>=2.0.0`
* `numpy`
* `transformers`
* `scipy`
* `torchgpipe`
* `pandas`
* `matplotlib`
* `tiktoken`

## Datasets

You need to prepare the datasets manually. The dataset path must be specified in the scripts. Ensure you have the dataset downloaded and accessible on your system.

## Running the Scripts

There are three scripts provided to run experiments with different optimizers. Make sure you have set the dataset paths correctly in each script.

### 1. Run with Adam Optimizer

To run the experiment with the **Adam** optimizer, use the script `run_1B_adam.sh`. It will run the model with the Adam optimizer on the 1B parameter model.

```bash
bash run_1B_adam.sh
```

### 2. Run with Muon Optimizer

To run the experiment with the **Muon** optimizer, use the script `run_1B_muon.sh`. This will run the model with the Muon optimizer on the 1B parameter model.

```bash
bash run_1B_muon.sh
```

### 3. Run with NewMuon (with momentum smoothing)

To run the experiment with **NewMuon**, use the script `run_1B_newmuon_sm.sh`. This will run the model with the NewMuon optimizer (with momentum smoothing) on the 1B parameter model.

```bash
bash run_1B_newmuon_sm.sh
```

