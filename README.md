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
## RT-DETR environments and configurations

See [the RT-DETR configuration guide](parameters/RTDETR/README.md) for the
reference experiments and setup commands for `sarfusion` and
`sarfusion-rtdetrv2`. YOLO26 uses a separate `sarfusion-yolo26` environment,
created by cloning `sarfusion` and installing `requirements-yolo26.txt`.

## Local data and thesis materials

`dataset/`, checkpoints, W&B runs and `notes/` are local files excluded from Git.
A fresh clone contains the implementations and reference configurations.
Replaying completed experiments requires their original data and checkpoints;
reports that compare against thesis results also require the relevant CSV/JSON
files under `notes/Thesis/results/`.

Tests of those local datasets and historical results are skipped when the files
are absent. Model, loss, configuration and metric tests still run. Operational
YOLO26 source manifests cover repository files and do not require local notes.
Archived manifests retain the hashes recorded for the completed experiments.

## Tests

Run from the repository root in the corresponding environments. The historical
YOLOv10 fork and current YOLO26 package require separate environments.

```bash
conda activate sarfusion
python - <<'PY'
from pathlib import Path
import unittest

suite = unittest.TestSuite()
for path in sorted(Path("tests").glob("test_*.py")):
    if not path.name.startswith("test_yolo26"):
        suite.addTests(unittest.defaultTestLoader.discover("tests", pattern=path.name))
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
PY

conda run -n sarfusion-rtdetrv2 python -m unittest discover -s tests -p 'test_rtdetr_v2*.py'
conda run -n sarfusion-yolo26 python -m pytest -q tests/test_yolo26.py tests/test_yolo26_repair_protocol.py tests/test_yolo26_five_seed_v2.py
```

## Post-thesis maintenance

The final review corrected box clipping at the left/top tile boundaries and the
DCNv2 `(dy, dx)` interpretation in FAM diagnostic plots. Re-running tiled
experiments or generating diagnostic plots uses these corrections. Existing
thesis results and figures have not been regenerated. The pre-correction
implementation is available at commit `bb6be50`.

Training source hashes include comments and docstrings. Any edit to a covered
file requires updating the operational hash before starting a new run; archived
experiment manifests continue to identify their original sources.
