# CNN Food Recognition

Local-first canteen checkout for 11 UEH dishes. The runtime uses one convolutional neural network classifier and a calibrated five-compartment tray grid.

## Pipeline

1. Prepare reviewed classification images.
2. Build train, validation, and test splits.
3. Train the dish classifier with PyTorch.
4. Crop each tray using the fixed five-compartment template.
5. Classify every crop, price the recognized dishes, and create a bill.

## Project layout

```text
canteen_checkout/                 shared Python package
configs/                          query and evaluation configuration
data/classification/              generated classifier dataset
models/dish_classifier.pt         CNN checkpoint downloaded from Drive
models/class_names.json           class order
scripts/apps/01_demo_checkout_app.py
scripts/apps/02_data_ide.py
scripts/cli/01_crop_tray.py
scripts/cli/02_demo_checkout.py
scripts/data/                      classification data tools
scripts/train/01_train_classifier.py
static/ and templates/            browser demo
tests/                             automated tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the demo

```powershell
.\.venv\Scripts\python.exe scripts\apps\01_demo_checkout_app.py --host 127.0.0.1 --port 7863
```

Open `http://127.0.0.1:7863`. The demo crops the tray with the calibrated grid and sends each crop only through `models/dish_classifier.pt`.

## Restore data and model from Drive

GitHub stores code and history only. Keep datasets and PyTorch checkpoints in Google Drive, then restore them after cloning:

```text
Google Drive/cnn-food-recognition/data/   -> <repo>/data/
Google Drive/cnn-food-recognition/models/dish_classifier.pt
                                          -> <repo>/models/dish_classifier.pt
```

The transfer ZIP contains `local-data.zip` and `cnn-models-for-drive.zip` for the initial Drive upload. The demo requires `models/dish_classifier.pt` at runtime; model files remain ignored by Git.

Run the Data IDE:

```powershell
.\.venv\Scripts\python.exe scripts\apps\02_data_ide.py --host 127.0.0.1 --port 7864
```

Run checkout from the terminal:

```powershell
.\.venv\Scripts\python.exe scripts\cli\02_demo_checkout.py --image data\demo\tray.jpg
```

## Build and train

```powershell
.\.venv\Scripts\python.exe scripts\data\01_build_classification_dataset.py --clear
.\.venv\Scripts\python.exe scripts\data\02_audit_dataset_conflicts.py --root data\classification --phash-threshold 4
.\.venv\Scripts\python.exe scripts\data\03_package_classification_dataset.py
.\.venv\Scripts\python.exe scripts\train\01_train_classifier.py --data data\classification --epochs 20 --batch-size 32 --patience 3
```

The Colab notebook mirrors this classifier-only workflow. Prices live in `prices.csv`; loyalty, vouchers, paid bills, and ratings are stored in `outputs/canteen_engagement.sqlite3`.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
