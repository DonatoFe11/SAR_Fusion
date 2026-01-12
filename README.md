# RGBT-Fusion-for-SAR

## Overview
This repository provides tools for processing and analyzing SAR (Search and Rescue) data, leveraging RGBT (RGB-Thermal) data fusion for enhanced performance. It includes scripts for training machine learning models, dataset preprocessing, and running a FastAPI server to interact with the models.

## Features
- **FastAPI server**: Deploy the application for model inference.
- **Dataset preprocessing**: Scripts to prepare WiSARD and SARDATA datasets.
- **Model training**: Train a pose classifier and YOLOv10 for object detection.

---

## Run the FastAPI server

### With Docker

Build and run the Docker container to start the FastAPI server:

```sh
docker build -t sarfusion .
docker run -it --rm -p 8000:8000 sarfusion
```

### Without Docker

Run the FastAPI server manually after setting up the environment:

```sh
python main.py app
```

#### Prepare the conda environment:

Set up and activate the conda environment as follows:

```bash
conda env create -f environment.yml
conda activate sarfusion
```

---

## Train the model

### Download and Prepare Datasets

#### Download the WiSARD dataset:

Download the dataset from the link below:

[WiSARD Dataset](https://drive.google.com/file/d/1PKjGCqUszHH1nMbXUBTwPSDqRabAt_ht)

#### Extract and preprocess the WiSARD dataset:

1. Unzip the dataset:
    ```bash
    unzip dataset/WiSARDv1.zip -d dataset/WiSARD
    ```
2. Preprocess the dataset:
    ```bash
    python3 main.py preprocess_wisard
    ```

#### Extract classification patches from the SARDATA dataset:

Prepare classification patches using:

```bash
python3 main.py preprocess_classification
```

---

### Train Models

#### Train the Pose Classifier:

Train the pose classifier using the following command:

```bash
python3 main.py experiment --parameters="parameters/SARD_pose/parameters.yaml"
```

#### Annotate the WiSARD dataset with the pose classifier:

1. Move the pose classifier checkpoint to the `checkpoints` folder.
2. Preprocess Wisard
    ```bash
    python3 main.py preprocess_wisard
    ```
3. Annotate the dataset using:
    ```bash
    python3 main.py annotate_wisard --model-yaml parameters/WiSARD_pose/parameters.yaml
    ```

#### Train the FusionDETR Model:

Train the FusionDETR model using the following command:

```bash
python main.py experiment --parameters "parameters/DETR/fusiondetr.yaml"
```

#### Test the FusionDETR Model:

Ater moving the trained checkpoint to the `checkpoints` folder, ensuring that their names match in the .yaml file, run the following command to test the model:


```bash
python main.py experiment --parameters "parameters/DETR/fusion_test.yaml"
```